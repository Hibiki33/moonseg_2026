"""Source-defined SAM3 semantic postprocessing, without model/framework wrappers.

SegEarth-OV-3 87e5cc18: q=sigmoid(object)*sigmoid(presence), I=max(q*M),
F=presence*max(D,I). ``masks_logits`` in its processor are probabilities;
this module instead accepts raw Meta forward_grounding outputs explicitly.
No ground truth or category-presence labels enter these functions.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


MODE_SPECS = {
    "legacy_dense": {"formula": "sigmoid(resize(semantic_seg.float()))", "instance_threshold": None},
    "dense_presence": {"formula": "p * D", "instance_threshold": None},
    "segearth_t050": {"formula": "p * max(D, max_i(q_i*M_i))", "instance_threshold": 0.5},
    "segearth_t030": {"formula": "p * max(D, max_i(q_i*M_i))", "instance_threshold": 0.3},
    "single_presence_t050": {"formula": "max(p*D, max_i(q_i*M_i))", "instance_threshold": 0.5},
}
INSTANCE_SPECS = {"instance_t050": 0.5, "instance_t030": 0.3}
SEG_EARTH_COMMIT = "87e5cc18c8bc9c0b4713b975587281ac1c97c79a"


def mask_probability_chunks(raw_masks, size, chunk_size=8, indices=None):
    """Resize native raw masks before sigmoid, in bounded original-coordinate chunks."""
    if raw_masks.ndim != 3 or type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("Require QxHxW raw masks and a positive integer mask chunk size")
    if indices is None:
        indices = torch.arange(raw_masks.shape[0], device=raw_masks.device)
    for start in range(0, indices.numel(), chunk_size):
        ids = indices[start:start + chunk_size]
        # Deliberately retain native dtype/operation order, as in the pinned
        # author processor. Legacy dense alone explicitly casts before resize.
        probabilities = F.interpolate(raw_masks[ids, None], size=size, mode="bilinear",
                                      align_corners=False)[:, 0].sigmoid()
        yield ids, probabilities


@torch.inference_mode()
def prompt_semantic_maps(outputs, size, chunk_size=8):
    """Reduce one prompt into all modes; empty instance sets yield zero I/support."""
    dense = outputs["semantic_seg"]
    masks = outputs["pred_masks"]
    logits = outputs["pred_logits"]
    presence_logit = outputs["presence_logit_dec"]
    if (dense.ndim != 4 or tuple(dense.shape[:2]) != (1, 1)
            or masks.ndim != 4 or masks.shape[0] != 1
            or tuple(logits.shape) != (1, masks.shape[1], 1) or presence_logit.numel() != 1):
        raise ValueError("Expected one image/prompt and native dense, query, mask and presence logits")
    if any(not torch.isfinite(value).all() for value in (dense, masks, logits, presence_logit)):
        raise ValueError("Nonfinite native SAM3 output")
    p = presence_logit.sigmoid().reshape(())
    r = logits.sigmoid().reshape(-1)
    q = r * p
    legacy = F.interpolate(dense.float(), size=size, mode="bilinear", align_corners=False)[0, 0].sigmoid()
    native_dense = F.interpolate(dense, size=size, mode="bilinear", align_corners=False)[0, 0].sigmoid()
    instances = {name: torch.zeros(size, dtype=torch.float32, device=dense.device) for name in INSTANCE_SPECS}
    support = {name: torch.zeros(size, dtype=torch.bool, device=dense.device) for name in INSTANCE_SPECS}
    selected = (q > min(INSTANCE_SPECS.values())).nonzero(as_tuple=True)[0]
    mask_dtype = None
    for ids, probabilities in mask_probability_chunks(masks[0], size, chunk_size, selected):
        mask_dtype = str(probabilities.dtype)
        # The author multiplies each HxW probability map by a 0-D score.
        # Preserve scalar type promotion when mask and score dtypes differ.
        weighted = torch.stack([probabilities[j] * q[i] for j, i in enumerate(ids)])
        for name, threshold in INSTANCE_SPECS.items():
            keep = q[ids] > threshold
            if keep.any():
                instances[name] = torch.maximum(instances[name], weighted[keep].amax(0))
                support[name] |= (probabilities[keep] > 0.5).any(0)
    # Author seg_logits accumulator is FP32, including when dense is BF16.
    D = native_dense.float()
    scores = {
        "legacy_dense": legacy,
        "dense_presence": D * p,
        "segearth_t050": torch.maximum(D, instances["instance_t050"]) * p,
        "segearth_t030": torch.maximum(D, instances["instance_t030"]) * p,
        "single_presence_t050": torch.maximum(D * p, instances["instance_t050"]),
    }
    metadata = {"presence": float(p), "queries": q.numel(),
                "conditional_score_max": float(r.max()) if r.numel() else None,
                "joint_score_max": float(q.max()) if q.numel() else None,
                "filter_counts": {f"t{int(t*100):03d}": {
                    "r_above": int((r > t).sum()), "q_above": int((q > t).sum())}
                    for t in INSTANCE_SPECS.values()},
                "dtypes": {"raw_dense": str(dense.dtype), "raw_masks": str(masks.dtype),
                           "presence": str(p.dtype), "r": str(r.dtype), "q": str(q.dtype),
                           "resized_native_dense": str(native_dense.dtype),
                           "resized_retained_masks": mask_dtype, "accumulator": str(D.dtype)}}
    return {"scores": scores, "instances": instances, "support": support, "metadata": metadata}


def class_prediction(scores):
    """Five-class argmax, with explicit failure instead of a zero/nonfinite fallback."""
    if scores.ndim != 3 or scores.shape[0] != 5 or not torch.isfinite(scores).all():
        raise ValueError("Full semantic scores must be finite 5xHxW")
    maxima, prediction = scores.max(0)
    if (scores < 0).any() or (maxima <= 0).any():
        raise ValueError("Full semantic scores contain negative scores or uncovered/all-zero pixels")
    ties = int(((scores == maxima[None]).sum(0) > 1).sum())
    # torch.max returns the first (lowest fixed class ID) maximum.
    return prediction, ties


def instance_prediction(scores, support):
    """Compete only within binary-mask coverage; reject=5 never means regolith=0."""
    if scores.shape != support.shape or scores.ndim != 3 or scores.shape[0] != 5:
        raise ValueError("Instance scores/support must both have shape 5xHxW")
    if not torch.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Nonfinite or negative instance score")
    prediction = scores.masked_fill(~support, -torch.inf).argmax(0)
    prediction[~support.any(0)] = 5
    return prediction
