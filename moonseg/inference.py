"""Single-image, sliding-window and multiscale/flip score inference.

This module only turns one normalized image into class scores. Checkpoint loading and
the prediction CLI stay outside it. The scale, flip and window order below is the
order the recorded narrowflip result used.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .utils import amp_context

PROFILES = {
    "single": {"sliding_window": False, "scales": [1.0], "flip": False},
    "narrowflip": {"sliding_window": False, "scales": [0.9, 1.0, 1.1], "flip": True},
}

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


def inference_padding(model):
    """Record common input padding separately from padding owned by a model."""
    settings = {"pad_multiple": int(getattr(model, "inference_input_multiple", 32))}
    internal = getattr(model, "inference_internal_padding", None)
    if internal is not None:
        settings["model_internal_padding"] = dict(internal)
    return settings


@torch.inference_mode()
def _forward_scores(model, image, mode="none", multiple=None, num_classes=5):
    height, width = image.shape[-2:]
    if multiple is None:
        multiple = inference_padding(model)["pad_multiple"]
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
