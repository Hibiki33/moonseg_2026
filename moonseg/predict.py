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

from .data.manifest import TEST_SCENES, load_manifest, read_allowlist, require_complete_allowlist, scene_number
from .utils import load_partial_state_dict, amp_context, init_distributed, load_checkpoint, load_config, write_json

WEIGHTS_KEYS = ("auto", "model", "ema_model")
SCORE_POSTPROCESS = ("pred_masks logits bilinearly upsampled to the padded input size first, then sigmoid, "
                     "then einsum with softmax class probabilities[..., :num_classes] (DINOv3 m2f predict/"
                     "Mask2Former semantic inference order)")


def score_postprocess(config):
    architecture = config.get("model", {}).get("architecture", "dinov3_mask2former")
    if architecture == "sam3":
        return "Official per-prompt dense semantic logits resized to input size, then sigmoid; five-class argmax."
    if architecture in {"segformer_b5", "deeplabv3plus_mobilenetv3"}:
        return "Dense semantic logits bilinearly resized to input size, then class softmax."
    return SCORE_POSTPROCESS


def semantic_scores(outputs, size, num_classes=5):
    """Dense or Mask2Former semantic scores at `size`.

    Upsample mask logits before the sigmoid, as upstream DINOv3 (heads/mask2former_head.py
    predict() + eval/segmentation/inference.py) and Mask2Former sem_seg_postprocess do.
    The trailing no-object class is dropped by keeping the first num_classes probabilities.
    """
    dense_keys = {"semantic_logits", "semantic_scores"} & outputs.keys()
    if dense_keys:
        if len(dense_keys) != 1 or {"pred_masks", "pred_logits"} & outputs.keys():
            raise ValueError("Model output must have one unambiguous semantic representation")
        key = next(iter(dense_keys))
        scores = outputs[key].float()
        if scores.ndim != 4 or scores.shape[1] != num_classes:
            raise ValueError(f"Dense scores must have BCHW shape and {num_classes} channels")
        if tuple(scores.shape[-2:]) != tuple(size):
            scores = F.interpolate(scores, size=tuple(size), mode="bilinear", align_corners=False)
        return scores.softmax(dim=1) if key == "semantic_logits" else scores
    mask_logits = outputs["pred_masks"].float()
    if tuple(mask_logits.shape[-2:]) != tuple(size):
        mask_logits = F.interpolate(mask_logits, size=tuple(size), mode="bilinear", align_corners=False)
    class_prob = outputs["pred_logits"].float().softmax(-1)[..., :num_classes]
    return torch.einsum("bqc,bqhw->bchw", class_prob, mask_logits.sigmoid())


@torch.inference_mode()
def _forward_scores(model, image, mode="none", multiple=32, num_classes=5):
    height, width = image.shape[-2:]
    image = F.pad(image, (0, (-width) % multiple, 0, (-height) % multiple))
    with amp_context(image.device, mode):
        output = model(image)
    return semantic_scores(output, image.shape[-2:], num_classes)[..., :height, :width]


def _positions(length, crop, stride):
    if stride <= 0 or stride > crop:
        raise ValueError("Sliding-window stride must be in (0, crop_size]")
    last = max(0, length - crop)
    return list(range(0, last, stride)) + [last]


@torch.inference_mode()
def predict_scores(model, image, inference, amp="none", num_classes=5):
    """image: normalized BCHW with batch=1; returns B x num_classes x H x W scores."""
    if image.shape[0] != 1:
        raise ValueError("Sliding-window/TTA inference accepts one image at a time")
    num_classes = int(num_classes)
    original_size = image.shape[-2:]
    scales = inference.get("scales", [1.0])
    if not scales or any(float(scale) <= 0 for scale in scales):
        raise ValueError("inference.scales must be a nonempty list of positive values")
    combined = torch.zeros((1, num_classes, *original_size), device=image.device, dtype=torch.float32)
    count = 0
    for scale in scales:
        scaled_size = tuple(max(1, round(length * float(scale))) for length in original_size)
        scaled = F.interpolate(image, size=scaled_size, mode="bilinear", align_corners=False)
        for flip in ([False, True] if inference.get("flip", False) else [False]):
            source = scaled.flip(-1) if flip else scaled
            crop_size = inference.get("crop_size") if inference.get("sliding_window", True) else None
            if crop_size is None:
                scores = _forward_scores(model, source, amp, num_classes=num_classes)
            else:
                crop_h, crop_w = map(int, crop_size)
                stride_h, stride_w = map(int, inference.get("stride", [crop_h * 2 // 3, crop_w * 2 // 3]))
                scores = torch.zeros((1, num_classes, *scaled_size), device=image.device, dtype=torch.float32)
                coverage = torch.zeros((1, 1, *scaled_size), device=image.device, dtype=torch.float32)
                for top in _positions(scaled_size[0], crop_h, stride_h):
                    for left in _positions(scaled_size[1], crop_w, stride_w):
                        bottom, right = min(top + crop_h, scaled_size[0]), min(left + crop_w, scaled_size[1])
                        tile = _forward_scores(model, source[..., top:bottom, left:right], amp, num_classes=num_classes)
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
    return Path("test") / f"Moon_{scene_number(record['scene'])}" / "submission" / f"{sample_id}.png"


def select_weights(checkpoint, weights_key="auto"):
    """Return (state_dict, used_key, message).

    auto always resolves to 'model'. EMA was removed from training: a legacy checkpoint's
    'ema_model' loads only when requested explicitly, and its best_source='ema_model' is ignored
    with a note in the message. An explicit 'model' or 'ema_model' must exist in the checkpoint.
    """
    if weights_key not in WEIGHTS_KEYS:
        raise ValueError(f"weights key must be one of {WEIGHTS_KEYS}")
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary with a 'model' state_dict")
    if weights_key == "auto":
        used = "model"
        message = "weights: 'model' (auto)"
        legacy = checkpoint.get("best_source")
        if legacy not in (None, "model"):
            message += (f"; note: legacy best_source={legacy!r} ignored because EMA was removed; "
                        f"request {legacy!r} explicitly to load those weights")
    else:
        used = weights_key
        message = f"weights: '{used}' (explicit)"
    if not isinstance(checkpoint.get(used), dict):
        raise KeyError(f"Checkpoint has no '{used}' state_dict; available keys: {sorted(checkpoint)}")
    return checkpoint[used], used, message


def _json_scalar(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return value.item() if hasattr(value, "item") else str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True, help="Complete unlabeled official test manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--official-test-allowlist", required=True)
    parser.add_argument("--weights-key", choices=WEIGHTS_KEYS, default="auto",
                        help="auto = model; ema_model only for legacy checkpoints from runs with the removed train.ema")
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
    data_root = Path(config["data"]["root"]).resolve()
    records = load_manifest(args.manifest, data_root, TEST_SCENES,
                            args.official_test_allowlist, need_label=False)
    require_complete_allowlist(records, read_allowlist(args.official_test_allowlist))
    from .models import build_model
    num_classes = int(config.get("model", {}).get("num_classes", 5))
    model = build_model(config, load_backbone=True).to(device)
    checkpoint = load_checkpoint(args.checkpoint)
    state, weights_key, message = select_weights(checkpoint, args.weights_key)
    if rank == 0:
        print(f"{message}; checkpoint={args.checkpoint}", flush=True)
    load_partial_state_dict(model, state, weights_key)
    model.eval()
    inference = config.get("inference", {})
    amp = config.get("train", {}).get("amp", "none")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    predictions = []
    # Strided assignment has no repeated padding samples and gives each PNG one writer.
    for index in range(rank, len(records), world_size):
        record = records[index]
        with Image.open(data_root / record["image"]) as handle:
            image = handle.convert("RGB")
        tensor = normalize_image(image, config["data"]).to(device)
        scores = predict_scores(model, tensor, inference, amp, num_classes)
        mask = scores.argmax(1)[0].cpu().numpy().astype(np.uint8)
        relative = prediction_path(record)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(target)
        predictions.append((index, {"scene": scene_number(record["scene"]), "sample_id": record["sample_id"],
                                   "prediction": relative.as_posix(), "width": image.width, "height": image.height}))
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
        settings = {"sliding_window": bool(inference.get("sliding_window", True)) and inference.get("crop_size") is not None,
                    "crop_size": inference.get("crop_size"), "stride": inference.get("stride"),
                    "scales": [float(scale) for scale in inference.get("scales", [1.0])],
                    "flip": bool(inference.get("flip", False)), "amp": amp, "num_classes": num_classes,
                    "pad_multiple": 32, "score_postprocess": score_postprocess(config), "world_size": world_size}
        write_json({"kind": "prediction_index", "layout": "test/Moon_i/submission/<sample_id>.png",
                    "packaging_status": "Official prediction layout; validate complete files with tools/package_submission.py before submitting.",
                    "checkpoint": str(args.checkpoint), "weights_key": weights_key,
                    "weights_key_requested": args.weights_key, "selected_by": _json_scalar(checkpoint.get("selected_by")),
                    "best_source": _json_scalar(checkpoint.get("best_source")),
                    "checkpoint_epoch": _json_scalar(checkpoint.get("epoch")), "inference": settings,
                    "predictions": [record for _, record in predictions]}, output / "prediction_index.json")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
