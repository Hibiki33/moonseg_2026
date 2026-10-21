"""Automatic single-image, sliding-window and optional multiscale/flip inference."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image

from .data.manifest import TEST_SCENES, load_manifest, read_allowlist, require_complete_allowlist
from .utils import amp_context, init_distributed, load_checkpoint, load_config, write_json


def semantic_scores(outputs, size):
    class_prob = outputs["pred_logits"].float().softmax(-1)[..., :5]
    mask_prob = outputs["pred_masks"].float().sigmoid()
    scores = torch.einsum("bqc,bqhw->bchw", class_prob, mask_prob)
    return F.interpolate(scores, size=size, mode="bilinear", align_corners=False)


@torch.inference_mode()
def _forward_scores(model, image, mode="none", multiple=32):
    height, width = image.shape[-2:]
    image = F.pad(image, (0, (-width) % multiple, 0, (-height) % multiple))
    with amp_context(image.device, mode):
        output = model(image)
    return semantic_scores(output, image.shape[-2:])[..., :height, :width]


def _positions(length, crop, stride):
    if stride <= 0 or stride > crop:
        raise ValueError("Sliding-window stride must be in (0, crop_size]")
    last = max(0, length - crop)
    return list(range(0, last, stride)) + [last]


@torch.inference_mode()
def predict_scores(model, image, inference, amp="none"):
    """image: normalized BCHW with batch=1; returns Bx5xHxW scores."""
    if image.shape[0] != 1:
        raise ValueError("Sliding-window/TTA inference accepts one image at a time")
    original_size = image.shape[-2:]
    scales = inference.get("scales", [1.0])
    if not scales or any(float(scale) <= 0 for scale in scales):
        raise ValueError("inference.scales must be a nonempty list of positive values")
    combined = torch.zeros((1, 5, *original_size), device=image.device, dtype=torch.float32)
    count = 0
    for scale in scales:
        scaled_size = tuple(max(1, round(length * float(scale))) for length in original_size)
        scaled = F.interpolate(image, size=scaled_size, mode="bilinear", align_corners=False)
        for flip in ([False, True] if inference.get("flip", False) else [False]):
            source = scaled.flip(-1) if flip else scaled
            crop_size = inference.get("crop_size") if inference.get("sliding_window", True) else None
            if crop_size is None:
                scores = _forward_scores(model, source, amp)
            else:
                crop_h, crop_w = map(int, crop_size)
                stride_h, stride_w = map(int, inference.get("stride", [crop_h * 2 // 3, crop_w * 2 // 3]))
                scores = torch.zeros((1, 5, *scaled_size), device=image.device, dtype=torch.float32)
                coverage = torch.zeros((1, 1, *scaled_size), device=image.device, dtype=torch.float32)
                for top in _positions(scaled_size[0], crop_h, stride_h):
                    for left in _positions(scaled_size[1], crop_w, stride_w):
                        bottom, right = min(top + crop_h, scaled_size[0]), min(left + crop_w, scaled_size[1])
                        tile = _forward_scores(model, source[..., top:bottom, left:right], amp)
                        scores[..., top:bottom, left:right] += tile
                        coverage[..., top:bottom, left:right] += 1
                scores /= coverage
            if flip:
                scores = scores.flip(-1)
            combined += F.interpolate(scores, size=original_size, mode="bilinear", align_corners=False)
            count += 1
    return combined / count


def normalize_image(image, data):
    pixels = np.asarray(image.convert("RGB"), np.float32) / 255.0
    mean = np.asarray(data.get("mean", [0.485, 0.456, 0.406]), np.float32)
    std = np.asarray(data.get("std", [0.229, 0.224, 0.225]), np.float32)
    return torch.from_numpy(((pixels - mean) / std).transpose(2, 0, 1).copy()).unsqueeze(0)


def prediction_path(record):
    sample_id = str(record["sample_id"])
    return Path("test") / f"Moon_{record['scene']}" / "submission" / f"{sample_id}.png"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True, help="Complete unlabeled official test manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--official-test-allowlist", required=True)
    parser.add_argument("--device", help="Explicit device for single-process inference; incompatible with multiple torchrun ranks")
    args = parser.parse_args()
    if args.device is not None and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        parser.error("--device cannot be used with multiple torchrun ranks; each rank uses its LOCAL_RANK device")
    config = load_config(args.config)
    if args.device is None:
        device, rank, world_size = init_distributed()
    else:
        device, rank, world_size = torch.device(args.device), 0, 1
        if device.type == "cuda":
            torch.cuda.set_device(device)
    records = load_manifest(args.manifest, config["data"]["root"], TEST_SCENES,
                            args.official_test_allowlist, need_label=False)
    require_complete_allowlist(records, read_allowlist(args.official_test_allowlist))
    from .models import build_model
    model = build_model(config, load_backbone=False).to(device)
    checkpoint = load_checkpoint(args.checkpoint)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    predictions = []
    # Strided assignment has no repeated padding samples and gives each PNG one writer.
    for index in range(rank, len(records), world_size):
        record = records[index]
        with Image.open(Path(config["data"]["root"]) / record["image"]) as handle:
            image = handle.convert("RGB")
        tensor = normalize_image(image, config["data"]).to(device)
        scores = predict_scores(model, tensor, config.get("inference", {}), config.get("train", {}).get("amp", "none"))
        mask = scores.argmax(1)[0].cpu().numpy().astype(np.uint8)
        relative = prediction_path(record)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(target)
        predictions.append((index, {"scene": record["scene"], "sample_id": record["sample_id"],
                                   "prediction": str(relative), "width": image.width, "height": image.height}))
        print(f"[rank {rank}; sample {index + 1}/{len(records)}] {target}", flush=True)
    if world_size > 1:
        gathered = [None] * world_size if rank == 0 else None
        dist.gather_object(predictions, gathered, dst=0)
        if rank == 0:
            predictions = [item for shard in gathered for item in shard]
    if rank == 0:
        predictions.sort(key=lambda item: item[0])
        if [index for index, _ in predictions] != list(range(len(records))):
            raise RuntimeError("Distributed prediction index has missing or duplicated samples")
        write_json({"kind": "prediction_index", "layout": "test/Moon_i/submission/<sample_id>.png",
                    "packaging_status": "Official prediction layout; validate complete files with tools/package_submission.py before submitting.",
                    "predictions": [record for _, record in predictions]}, output / "prediction_index.json")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
