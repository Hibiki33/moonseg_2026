"""Configuration, reproducibility, distributed training and checkpoint helpers."""
from __future__ import annotations

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
