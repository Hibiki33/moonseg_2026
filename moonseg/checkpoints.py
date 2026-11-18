"""Checkpoint IO and the frozen-backbone state policy.

Training, evaluation and prediction all load weights through this module. It does not
build models, run inference, or choose a learning rate.
"""
from __future__ import annotations

from pathlib import Path

import torch

WEIGHTS_KEYS = ("auto", "model", "ema_model")
FROZEN_BACKBONE_PREFIX = "backbone.backbone."


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def _trainable_names(model):
    return {name for name, parameter in model.named_parameters() if parameter.requires_grad}


def partial_state_dict(model):
    """Complete baseline state, or omit frozen pretrained DINOv3 tensors.

    Every tensor under ``backbone.backbone.`` that is not a trainable parameter (frozen weights and
    the deterministic RoPE/bias-mask buffers) is omitted; they are restored from
    ``model.backbone_weights`` when the model is rebuilt. Trainable ViT tensors
    (``model.unfreeze_last_blocks``/``freeze_backbone: false``) are kept.
    """
    model = _unwrap(model)
    if getattr(model, "checkpoint_full_state", False):
        # Dense baselines need the entire encoder, including mutable BatchNorm
        # statistics. Buffers are not named parameters and must not be dropped.
        return dict(model.state_dict())
    trainable = _trainable_names(model)
    return {key: value for key, value in model.state_dict().items()
            if not key.startswith(FROZEN_BACKBONE_PREFIX) or key in trainable}


def load_partial_state_dict(model, state, label="model"):
    """Load a checkpoint saved by partial_state_dict into a model built with the pretrained backbone.

    Only frozen ``backbone.backbone.`` tensors may be missing; any other missing key, any
    unexpected key, or a missing trainable tensor is an error. Full legacy checkpoints load unchanged.
    Returns the number of tolerated missing (frozen) keys.
    """
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint '{label}' is not a state dict")
    model = _unwrap(model)
    trainable = _trainable_names(model)
    expected = model.state_dict().keys()
    # Checked before loading so a mismatched checkpoint never leaves a half-loaded model.
    tolerated = [key for key in expected if key not in state]
    missing = [key for key in tolerated if getattr(model, "checkpoint_full_state", False)
               or not key.startswith(FROZEN_BACKBONE_PREFIX) or key in trainable]
    unexpected = [key for key in state if key not in expected]
    if missing or unexpected:
        raise ValueError(f"Checkpoint '{label}' does not match the model built from this config "
                         f"(check model.* such as unfreeze_last_blocks/freeze_backbone/num_classes): "
                         f"{len(missing)} missing, e.g. {missing[:8]}; {len(unexpected)} unexpected, "
                         f"e.g. {unexpected[:8]}")
    model.load_state_dict(state, strict=False)
    return len(tolerated)


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load_checkpoint(path, device="cpu"):
    # Optimizer/RNG states require full Python loading. Only load your own trusted checkpoints.
    return torch.load(path, map_location=device, weights_only=False)


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
