"""Optimizer groups, polynomial schedule and complete-batch counting.

Parameter groups use the same names, decay rules and backbone learning-rate multiplier
as the training runs already recorded. This module does not step the optimizer or
touch a checkpoint.
"""
from __future__ import annotations

import math

import torch

from .utils import unwrap_model

# Embedding-like tensors get no weight decay, like Mask2Former's WEIGHT_DECAY_EMBED=0. DINOv3 names
# its register tokens storage_tokens; register_tokens is kept for other ViT implementations.
NO_DECAY_NAMES = ("pos_embed", "cls_token", "register_tokens", "storage_tokens", "mask_token",
                  "level_embed", "level_encoding", "query_feat", "query_embed", "camera_token")


def make_optimizer(model, settings):
    # Only the native pretrained encoder gets the multiplier; new fusion/decoder modules use full LR.
    base_lr = float(settings.get("lr", 1e-4))
    groups = {}
    backbone_prefix = getattr(unwrap_model(model), "native_backbone_prefix", "backbone.backbone.")
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        backbone = name.startswith(backbone_prefix)
        no_decay = parameter.ndim <= 1 or name.endswith("bias") or any(part in name for part in NO_DECAY_NAMES)
        key = (backbone, no_decay)
        groups.setdefault(key, []).append(parameter)
    if not groups:
        raise ValueError("Model has no trainable parameters")
    return torch.optim.AdamW([
        {"params": parameters,
         "lr": base_lr * (float(settings.get("backbone_lr_multiplier", 0.1)) if backbone else 1.0),
         "weight_decay": 0.0 if no_decay else float(settings.get("weight_decay", 0.05))}
        for (backbone, no_decay), parameters in groups.items()
    ], betas=tuple(settings.get("betas", [0.9, 0.999])))


def make_scheduler(optimizer, total_steps, settings):
    warmup = min(int(settings.get("warmup_iters", 1500)), max(0, total_steps - 1))
    power = float(settings.get("poly_power", 0.9))
    warmup_ratio = float(settings.get("warmup_ratio", 0.001))

    def factor(step):
        if step < warmup:
            return warmup_ratio + (1 - warmup_ratio) * step / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return max(0.0, 1.0 - progress) ** power

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def training_batch_plan(dataset_size, world_size, settings):
    """Count complete optimizer updates when a minimum effective batch is required."""
    batch_size = int(settings.get("batch_size", 1))
    accumulation = int(settings.get("accumulation_steps", 1))
    minimum = int(settings.get("min_effective_batch", 0))
    if batch_size < 1 or accumulation < 1 or minimum < 0 or world_size < 1:
        raise ValueError("batch_size, accumulation_steps and world_size must be positive; min_effective_batch cannot be negative")
    effective = batch_size * world_size * accumulation
    if effective < minimum:
        raise ValueError(f"Effective batch {effective} is below train.min_effective_batch={minimum}; "
                         "increase batch/accumulation or launch the configured number of GPUs")
    strict = minimum > 0
    samples_per_rank = dataset_size // world_size if strict else math.ceil(dataset_size / world_size)
    loader_batches = samples_per_rank // batch_size if strict else math.ceil(samples_per_rank / batch_size)
    updates = loader_batches // accumulation if strict else math.ceil(loader_batches / accumulation)
    if updates < 1:
        raise ValueError(f"Training split with {dataset_size} samples cannot form one complete "
                         f"effective batch of {effective} under the current tail policy")
    batches = updates * accumulation if strict else loader_batches
    return {"batch_per_gpu": batch_size, "world_size": world_size, "accumulation_steps": accumulation,
            "effective_batch": effective, "minimum_effective_batch": minimum,
            "drop_incomplete_updates": strict, "microbatches_per_epoch": batches,
            "optimizer_steps_per_epoch": updates,
            "samples_used_per_epoch": updates * effective if strict else samples_per_rank * world_size,
            "dropped_samples_per_epoch": dataset_size - updates * effective if strict else 0,
            "tail_policy": "drop incomplete shuffled optimizer batch" if strict else "allow partial final update"}
