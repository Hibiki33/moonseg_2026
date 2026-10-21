"""Evaluate scene 5 locally without reading any held-out test labels."""
from __future__ import annotations

import argparse

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from .data import MoonDataset, collate_samples
from .metrics import SegmentationMetrics
from .predict import predict_scores
from .utils import init_distributed, load_checkpoint, load_config, unwrap_model, write_json


@torch.inference_mode()
def evaluate_model(model, dataset, device, inference, amp="none", rank=0, world_size=1, num_workers=2):
    model = unwrap_model(model)
    was_training = model.training
    model.eval()
    # A strided subset has no repeated padding samples, unlike DistributedSampler.
    loader = DataLoader(Subset(dataset, list(range(rank, len(dataset), world_size))), batch_size=1,
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
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="runs/local_metrics.json")
    args = parser.parse_args()
    config = load_config(args.config)
    device, rank, world_size = init_distributed()
    from .models import build_model
    dataset = MoonDataset(config["data"], "val")
    model = build_model(config, load_backbone=False).to(device)
    model.load_state_dict(load_checkpoint(args.checkpoint)["model"], strict=True)
    result = evaluate_model(model, dataset, device, config.get("inference", {}),
                            config.get("train", {}).get("amp", "none"), rank, world_size,
                            config.get("train", {}).get("num_workers", 2))
    if rank == 0:
        write_json(result, args.output)
        print(result, flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
