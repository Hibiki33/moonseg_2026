"""Configuration, reproducibility and distributed runtime.

Checkpoint loading is implemented in ``checkpoints`` and re-exported here so existing
callers keep working.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import torch
import torch.distributed as dist
import yaml


def matches_github_repository(value, repository):
    """Check the HTTPS origin recorded in a local third-party source manifest."""
    if not isinstance(value, str):
        return False
    return urlsplit(value) == ("https", "github.com", "/" + repository, "", "")


def load_config(path):
    with open(path, encoding="utf-8") as stream:
        # Resolved JSON must retain JSON numeric types (YAML reads 2e-05 as a string).
        config = json.load(stream) if Path(path).suffix.lower() == ".json" else yaml.safe_load(stream)
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


from .checkpoints import (  # noqa: E402
    FROZEN_BACKBONE_PREFIX,
    atomic_torch_save,
    load_checkpoint,
    load_partial_state_dict,
    partial_state_dict,
)
