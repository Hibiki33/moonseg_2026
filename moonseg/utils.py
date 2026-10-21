"""Configuration, reproducibility, distributed training, EMA and checkpoint helpers."""
from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml


def load_config(path):
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    return config


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch_backends(settings):
    """Apply train.tf32 / train.cudnn_benchmark (both default true) and return the applied values."""
    tf32, benchmark = settings.get("tf32", True), settings.get("cudnn_benchmark", True)
    if not isinstance(tf32, bool) or not isinstance(benchmark, bool):
        raise ValueError("train.tf32 and train.cudnn_benchmark must be true or false")
    # TF32 affects FP32 matmuls/convolutions outside autocast (e.g. matching costs), not BF16 kernels.
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.set_float32_matmul_precision("high" if tf32 else "highest")
    torch.backends.cudnn.benchmark = benchmark
    return {"tf32": tf32, "cudnn_benchmark": benchmark}


def init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world_size > 1:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    return device, rank, world_size


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


class ModelEMA:
    """Exponential moving average of the trainable parameters; buffers are copied, not averaged.

    Frozen parameters are shared with the source model (never copied), so the extra memory is the
    trainable parameters plus buffers. Its state dict omits the frozen ViT tensors like the model's.
    Build it from the DDP-synchronized model so every rank starts from identical weights.
    """

    def __init__(self, model, decay):
        if isinstance(decay, bool) or not 0 < float(decay) < 1:
            raise ValueError("train.ema.decay must be in (0, 1)")
        model = unwrap_model(model)
        self.decay = float(decay)
        self.names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        shared = {id(parameter): parameter for parameter in model.parameters() if not parameter.requires_grad}
        self.module = copy.deepcopy(model, shared).eval()
        for name, parameter in self.module.named_parameters():
            if name in self.names:
                parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        source = dict(unwrap_model(model).named_parameters())
        target = dict(self.module.named_parameters())
        for name in self.names:
            target[name].lerp_(source[name].detach(), 1.0 - self.decay)
        self.copy_buffers(model)

    @torch.no_grad()
    def copy_buffers(self, model):
        source = dict(unwrap_model(model).named_buffers())
        for name, buffer in self.module.named_buffers():
            buffer.copy_(source[name])

    # Every EMA parameter has requires_grad=False, so pass the source model's trainable names:
    # unfrozen ViT blocks (model.unfreeze_last_blocks) must be saved and required like the model's.
    def state_dict(self):
        return partial_state_dict(self.module, self.names)

    def load_state_dict(self, state):
        load_partial_state_dict(self.module, state, "ema_model", self.names)


FROZEN_BACKBONE_PREFIX = "backbone.backbone."


def _trainable_names(model, trainable=None):
    if trainable is None:
        return {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return set(trainable)


def partial_state_dict(model, trainable=None):
    """State dict without the frozen pretrained DINOv3 tensors.

    Every tensor under ``backbone.backbone.`` that is not a trainable parameter (frozen weights and
    the deterministic RoPE/bias-mask buffers) is omitted; they are restored from
    ``model.backbone_weights`` when the model is rebuilt. Trainable ViT tensors
    (``model.unfreeze_last_blocks``/``freeze_backbone: false``) are kept. ``trainable`` overrides
    the parameter names treated as trainable (default: those with requires_grad, e.g. not for EMA).
    """
    model = unwrap_model(model)
    trainable = _trainable_names(model, trainable)
    return {key: value for key, value in model.state_dict().items()
            if not key.startswith(FROZEN_BACKBONE_PREFIX) or key in trainable}


def load_partial_state_dict(model, state, label="model", trainable=None):
    """Load a checkpoint saved by partial_state_dict into a model built with the pretrained backbone.

    Only frozen ``backbone.backbone.`` tensors may be missing; any other missing key, any
    unexpected key, or a missing trainable tensor is an error. Full legacy checkpoints load unchanged.
    Returns the number of tolerated missing (frozen) keys.
    """
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint '{label}' is not a state dict")
    model = unwrap_model(model)
    trainable = _trainable_names(model, trainable)
    expected = model.state_dict().keys()
    # Checked before loading so a mismatched checkpoint never leaves a half-loaded model.
    tolerated = [key for key in expected if key not in state]
    missing = [key for key in tolerated if not key.startswith(FROZEN_BACKBONE_PREFIX) or key in trainable]
    unexpected = [key for key in state if key not in expected]
    if missing or unexpected:
        raise ValueError(f"Checkpoint '{label}' does not match the model built from this config "
                         f"(check model.* such as unfreeze_last_blocks/freeze_backbone/num_classes): "
                         f"{len(missing)} missing, e.g. {missing[:8]}; {len(unexpected)} unexpected, "
                         f"e.g. {unexpected[:8]}")
    model.load_state_dict(state, strict=False)
    return len(tolerated)


def rng_state():
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load_checkpoint(path, device="cpu"):
    # Optimizer/RNG states require full Python loading. Only load your own trusted checkpoints.
    return torch.load(path, map_location=device, weights_only=False)


def amp_context(device, mode):
    if mode not in ("bf16", "fp16", "none"):
        raise ValueError("train.amp must be bf16, fp16, or none")
    enabled = mode != "none" and device.type == "cuda"
    if enabled and mode == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support bf16; set train.amp=fp16")
    return torch.autocast(device_type=device.type, enabled=enabled,
                          dtype=torch.bfloat16 if mode == "bf16" else torch.float16)


def write_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
