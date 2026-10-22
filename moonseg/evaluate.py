"""Evaluate scene 5 locally without reading any held-out test labels."""
from __future__ import annotations

import argparse

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from .data import MoonDataset, collate_samples
from .metrics import SegmentationMetrics
from .predict import WEIGHTS_KEYS, predict_scores, select_weights, semantic_scores
from .utils import (amp_context, configure_torch_backends, init_distributed, load_checkpoint, load_config,
                    load_partial_state_dict, unwrap_model, write_json)

# Seeds the criterion's random point sampling: the validation loss is reproducible, every epoch
# draws the same points for a sample, and evaluation never advances the caller's RNG streams.
VALIDATION_LOSS_SEED = 0


def validation_indices(dataset, stride=1):
    """Every stride-th record in (scene, sample_id) order; stride 1 returns all records."""
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("Validation stride must be a positive integer")
    order = sorted(range(len(dataset)), key=lambda index: (int(dataset.records[index]["scene"]),
                                                           str(dataset.records[index]["sample_id"])))
    return order[::stride]


def _single_forward(inference, image):
    """True when predict_scores reduces to exactly one model(image) call (no window, TTA or padding)."""
    sliding = bool(inference.get("sliding_window", True)) and inference.get("crop_size") is not None
    scales = [float(scale) for scale in inference.get("scales", [1.0])]
    return (not sliding and scales == [1.0] and not inference.get("flip", False)
            and image.shape[-2] % 32 == 0 and image.shape[-1] % 32 == 0)


@torch.inference_mode()
def evaluate_model(model, dataset, device, inference, amp="none", rank=0, world_size=1, num_workers=2,
                   indices=None, criterion=None):
    """Evaluate a raw or DDP-wrapped model on dataset[indices] (default: all records).

    With a criterion, result["val_loss"] is its total training loss (final plus auxiliary layers,
    batch 1, same autocast mode as training) averaged over the evaluated samples of all ranks;
    without one it is None.
    """
    model = unwrap_model(model)
    was_training = model.training
    model.eval()
    indices = list(range(len(dataset))) if indices is None else list(indices)
    # A strided subset has no repeated padding samples, unlike DistributedSampler.
    loader = DataLoader(Subset(dataset, indices[rank::world_size]), batch_size=1,
                        num_workers=num_workers, collate_fn=collate_samples, pin_memory=device.type == "cuda")
    metrics = SegmentationMetrics(device=device)
    scenes = sorted(set(int(record["scene"]) for record in dataset.records))
    scene_metrics = {scene: SegmentationMetrics(device=device) for scene in scenes}
    loss_totals = torch.zeros(2, device=device, dtype=torch.float64)  # loss sum, sample count
    sync_num_masks = criterion.sync_num_masks if criterion is not None else None
    try:
        if criterion is not None:
            # Ranks evaluate different sample counts (e.g. 800/799 of 1599), so the criterion's
            # per-forward mask-count all_reduce would deadlock; each sample uses its own count.
            criterion.sync_num_masks = False
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.random.default_generator.manual_seed(VALIDATION_LOSS_SEED)
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(VALIDATION_LOSS_SEED)
            for batch in loader:
                image = batch["images"].to(device, non_blocking=True)
                target = batch["targets"].to(device, non_blocking=True)
                if criterion is not None and _single_forward(inference, image):
                    # The plain inference path is this one forward, so its output also gives the loss.
                    with amp_context(device, amp):
                        outputs = model(image)
                        loss = criterion(outputs, target)["loss"]
                    prediction = semantic_scores(outputs, image.shape[-2:]).argmax(1)
                else:
                    prediction = predict_scores(model, image, inference, amp).argmax(1)
                    if criterion is not None:
                        with amp_context(device, amp):
                            loss = criterion(model(image), target)["loss"]
                if criterion is not None:
                    loss_totals[0] += loss.double()
                    loss_totals[1] += 1
                metrics.update(prediction, target)
                scene_metrics[batch["scenes"][0]].update(prediction, target)
    finally:
        if criterion is not None:
            criterion.sync_num_masks = sync_num_masks
        model.train(was_training)
    metrics.synchronize()
    for metric in scene_metrics.values():
        metric.synchronize()
    if criterion is not None and dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_totals)
    result = metrics.compute()
    result["scenes"] = {str(scene): metric.compute() for scene, metric in scene_metrics.items()}
    result["evaluated_samples"] = len(indices)
    result["val_loss"] = (loss_totals[0] / loss_totals[1]).item() if loss_totals[1] > 0 else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True,
                        help="Builds the model and, from its loss section, the criterion for val_loss")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--weights", choices=WEIGHTS_KEYS, default="auto",
                        help="auto = model; ema_model only for legacy checkpoints from runs with the removed train.ema")
    parser.add_argument("--output", default="runs/local_metrics.json")
    args = parser.parse_args()
    config = load_config(args.config)
    configure_torch_backends(config.get("train", {}))
    device, rank, world_size = init_distributed()
    from .models import build_criterion, build_model
    dataset = MoonDataset(config["data"], "val")
    model = build_model(config, load_backbone=True).to(device)
    criterion = build_criterion(config).to(device)
    checkpoint = load_checkpoint(args.checkpoint)
    state, weights, message = select_weights(checkpoint, args.weights)
    if rank == 0:
        print(f"{message}; checkpoint={args.checkpoint}", flush=True)
    load_partial_state_dict(model, state, weights)
    result = evaluate_model(model, dataset, device, config.get("inference", {}),
                            config.get("train", {}).get("amp", "none"), rank, world_size,
                            config.get("train", {}).get("num_workers", 2), criterion=criterion)
    result.update({"checkpoint": str(args.checkpoint), "weights": weights,
                   "selected_by": checkpoint.get("selected_by"),
                   "checkpoint_epoch": checkpoint.get("epoch", -1) + 1})
    if rank == 0:
        write_json(result, args.output)
        print(result, flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
