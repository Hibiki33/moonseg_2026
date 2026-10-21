"""Evaluate scene 5 locally without reading any held-out test labels."""
from __future__ import annotations

import argparse

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from .data import MoonDataset, collate_samples
from .metrics import SegmentationMetrics
from .predict import predict_scores
from .utils import (load_partial_state_dict, configure_torch_backends, init_distributed, load_checkpoint, load_config,
                    unwrap_model, write_json)


def validation_indices(dataset, stride=1):
    """Every stride-th record in (scene, sample_id) order; stride 1 returns all records."""
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("Validation stride must be a positive integer")
    order = sorted(range(len(dataset)), key=lambda index: (int(dataset.records[index]["scene"]),
                                                           str(dataset.records[index]["sample_id"])))
    return order[::stride]


@torch.inference_mode()
def evaluate_model(model, dataset, device, inference, amp="none", rank=0, world_size=1, num_workers=2,
                   indices=None):
    """Evaluate a raw, DDP-wrapped or EMA model on dataset[indices] (default: all records)."""
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
    for batch in loader:
        image = batch["images"].to(device, non_blocking=True)
        target = batch["targets"].to(device, non_blocking=True)
        prediction = predict_scores(model, image, inference, amp).argmax(1)
        metrics.update(prediction, target)
        scene_metrics[batch["scenes"][0]].update(prediction, target)
    metrics.synchronize()
    for metric in scene_metrics.values():
        metric.synchronize()
    model.train(was_training)
    result = metrics.compute()
    result["scenes"] = {str(scene): metric.compute() for scene, metric in scene_metrics.items()}
    result["evaluated_samples"] = len(indices)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--weights", choices=("auto", "model", "ema_model"), default="auto",
                        help="auto uses the checkpoint's best_source (raw model when absent)")
    parser.add_argument("--output", default="runs/local_metrics.json")
    args = parser.parse_args()
    config = load_config(args.config)
    configure_torch_backends(config.get("train", {}))
    device, rank, world_size = init_distributed()
    from .models import build_model
    dataset = MoonDataset(config["data"], "val")
    model = build_model(config, load_backbone=True).to(device)
    checkpoint = load_checkpoint(args.checkpoint)
    weights = (checkpoint.get("best_source") or "model") if args.weights == "auto" else args.weights
    if weights not in checkpoint:
        raise KeyError(f"Checkpoint has no {weights!r} weights (train.ema.enabled was false?)")
    load_partial_state_dict(model, checkpoint[weights], weights)
    result = evaluate_model(model, dataset, device, config.get("inference", {}),
                            config.get("train", {}).get("amp", "none"), rank, world_size,
                            config.get("train", {}).get("num_workers", 2))
    result.update({"checkpoint": str(args.checkpoint), "weights": weights,
                   "checkpoint_epoch": checkpoint.get("epoch", -1) + 1})
    if rank == 0:
        write_json(result, args.output)
        print(result, flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
