"""DDP training with AMP, polynomial scheduling and complete epoch-boundary resume."""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
from itertools import islice
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .data import MoonDataset, collate_samples
from .evaluate import evaluate_model
from .utils import (amp_context, atomic_torch_save, init_distributed, load_checkpoint,
                    load_config, restore_rng, rng_state, seed_everything, seed_worker,
                    unwrap_model, write_json)


def make_optimizer(model, settings):
    # Only the original DINO transformer gets the multiplier. The ViT-Adapter trains at full LR.
    base_lr = float(settings.get("lr", 1e-4))
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        backbone = name.startswith("backbone.backbone.")
        no_decay = parameter.ndim <= 1 or name.endswith("bias") or any(
            part in name for part in ("pos_embed", "cls_token", "register_tokens", "mask_token"))
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


def _gather_rng(world_size):
    local = rng_state()
    if world_size == 1:
        return [local]
    result = [None for _ in range(world_size)]
    dist.all_gather_object(result, local)
    return result


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", help="Trusted full checkpoint; resumes at the following epoch")
    args = parser.parse_args()
    config = load_config(args.config)
    settings = config["train"]
    device, rank, world_size = init_distributed()
    seed = int(settings.get("seed", 42))
    seed_everything(seed + rank)
    train_dataset = MoonDataset(config["data"], "train")
    val_dataset = MoonDataset(config["data"], "val") if settings.get("validate", True) else None
    data_snapshot = {"train_samples": train_dataset.records,
                     "validation_samples": val_dataset.records if val_dataset else [],
                     "official_allowlist": Path(config["data"]["official_allowlist"]).read_text(encoding="utf-8")}
    if val_dataset is not None:
        train_ids = {str(row["sample_id"]) for row in train_dataset.records}
        val_ids = {str(row["sample_id"]) for row in val_dataset.records}
        if train_ids & val_ids:
            raise ValueError("Training and local validation manifests overlap")
        if set(config["data"].get("train_scenes", [])) & set(config["data"].get("val_scenes", [5])):
            raise ValueError("Scene-level train/validation split must be disjoint")
    batch_plan = training_batch_plan(len(train_dataset), world_size, settings)
    batch_size = batch_plan["batch_per_gpu"]
    accumulation = batch_plan["accumulation_steps"]
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank,
                                 shuffle=True, seed=seed, drop_last=batch_plan["drop_incomplete_updates"])
    workers = int(settings.get("num_workers", 4))
    loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                        num_workers=workers, collate_fn=collate_samples, pin_memory=device.type == "cuda",
                        worker_init_fn=seed_worker, persistent_workers=False,
                        drop_last=batch_plan["drop_incomplete_updates"])
    resume_path = args.resume or settings.get("resume")
    from .models import build_criterion, build_model
    model = build_model(config, load_backbone=not bool(resume_path)).to(device)
    criterion = build_criterion(config).to(device)
    optimizer = make_optimizer(model, settings)
    epochs = int(settings.get("epochs", 80))
    total_steps = epochs * batch_plan["optimizer_steps_per_epoch"]
    scheduler = make_scheduler(optimizer, total_steps, settings)
    amp = settings.get("amp", "bf16")
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp == "fp16")
    start_epoch, global_step, best_miou = 0, 0, -1.0
    if resume_path:
        checkpoint = load_checkpoint(resume_path)
        if checkpoint["world_size"] != world_size:
            raise ValueError("Epoch-boundary resume requires the checkpoint's WORLD_SIZE")
        if checkpoint["total_steps"] != total_steps:
            raise ValueError("Epoch-boundary resume requires unchanged epochs, batch size, accumulation and dataset length")
        if checkpoint.get("batch_plan", batch_plan) != batch_plan:
            raise ValueError("Epoch-boundary resume requires an unchanged effective batch and incomplete-update policy")
        for section in ("model", "loss", "data"):
            if checkpoint["config"].get(section) != config.get(section):
                raise ValueError(f"Epoch-boundary resume requires unchanged {section} configuration")
        if checkpoint["data_snapshot"] != data_snapshot:
            raise ValueError("Manifest sample membership or official allowlist changed since checkpoint")
        transient = {"resume", "output_dir", "log_every", "save_every"}
        old_settings = {key: value for key, value in checkpoint["config"]["train"].items() if key not in transient}
        new_settings = {key: value for key, value in settings.items() if key not in transient}
        if old_settings != new_settings:
            raise ValueError("Epoch-boundary resume requires unchanged training configuration (except resume/output/log/save paths)")
        if checkpoint["config"].get("inference") != config.get("inference"):
            raise ValueError("Epoch-boundary resume requires unchanged validation inference settings")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        global_step, best_miou = checkpoint["global_step"], checkpoint["best_miou"]
        restore_rng(checkpoint["rng_states"][rank])
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None,
                                        find_unused_parameters=False, broadcast_buffers=True)
    output = Path(settings.get("output_dir", "runs/moonseg"))
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        write_json(config, output / "resolved_config.json")
        write_json(batch_plan, output / "batch_plan.json")
        write_json({**data_snapshot,
                    "note": "Only explicit official allowlist samples admitted; public discovery grants no training permission."},
                   output / "data_provenance.json")
        print(f"Training {len(train_dataset)} official samples; world_size={world_size}; "
              f"effective_batch={batch_size * world_size * accumulation}; device={device}; amp={amp}", flush=True)
        print(json.dumps(batch_plan), flush=True)
    for epoch in range(start_epoch, epochs):
        sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = torch.zeros(2, device=device, dtype=torch.float64)
        started = time.monotonic()
        batches_per_epoch = batch_plan["microbatches_per_epoch"]
        for index, batch in enumerate(islice(loader, batches_per_epoch)):
            images = batch["images"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            group_start = (index // accumulation) * accumulation
            group_size = min(accumulation, batches_per_epoch - group_start)
            update = (index + 1) % accumulation == 0 or index + 1 == batches_per_epoch
            sync = contextlib.nullcontext() if update or world_size == 1 else model.no_sync()
            with sync:
                with amp_context(device, amp):
                    losses = criterion(model(images), targets)
                    loss = losses["loss"]
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at epoch={epoch} batch={index}: {loss.item()}")
                scaler.scale(loss / group_size).backward()
            epoch_loss[0] += loss.detach().double() * images.shape[0]
            epoch_loss[1] += images.shape[0]
            if update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings.get("grad_clip", 0.1)))
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
                    global_step += 1
                optimizer.zero_grad(set_to_none=True)
                if rank == 0 and global_step % int(settings.get("log_every", 20)) == 0:
                    print(json.dumps({"epoch": epoch + 1, "step": global_step,
                                      "loss": float(loss.detach()), "lr": scheduler.get_last_lr(),
                                      "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
        if dist.is_initialized():
            dist.all_reduce(epoch_loss)
            # Adapter BatchNorm updates running statistics locally. Validate and save
            # one consistent model; unwrapped validation forwards contain no collectives.
            for buffer in unwrap_model(model).buffers():
                dist.broadcast(buffer, src=0)
        result = None
        improved = False
        if val_dataset is not None and ((epoch + 1) % int(settings.get("eval_every", 1)) == 0 or epoch + 1 == epochs):
            result = evaluate_model(model, val_dataset, device, config.get("inference", {}), amp,
                                    rank, world_size, workers)
            if result["mIoU"] is not None and result["mIoU"] > best_miou:
                best_miou, improved = result["mIoU"], True
        states = _gather_rng(world_size)
        if rank == 0:
            summary = {"epoch": epoch + 1, "global_step": global_step,
                       "train_loss": (epoch_loss[0] / epoch_loss[1].clamp_min(1)).item(),
                       "seconds": time.monotonic() - started, "validation": result}
            with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(summary, ensure_ascii=False) + "\n")
            print(json.dumps(summary, ensure_ascii=False), flush=True)
            state = {"model": unwrap_model(model).state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                     "rng_states": states, "epoch": epoch, "global_step": global_step,
                     "best_miou": best_miou, "world_size": world_size, "total_steps": total_steps,
                     "config": config, "sampler_epoch": epoch, "data_snapshot": data_snapshot, "batch_plan": batch_plan,
                     "resume_scope": "Complete state at epoch boundary; use same config/world size. CUDA grid_sample backward is nondeterministic; bitwise replay is not guaranteed."}
            atomic_torch_save(state, output / "last.pt")
            if improved:
                atomic_torch_save(state, output / "best.pt")
            if (epoch + 1) % int(settings.get("save_every", 10)) == 0:
                atomic_torch_save(state, output / f"epoch_{epoch + 1:03d}.pt")
        if dist.is_initialized():
            dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
