"""DDP training with AMP, polynomial scheduling, per-epoch validation (mIoU and loss) and complete epoch-boundary resume."""
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
from .data.sampler import build_rare_class_sampler
from .evaluate import evaluate_model, validation_indices
from .utils import (amp_context, atomic_torch_save, configure_torch_backends, init_distributed,
                    load_partial_state_dict, partial_state_dict,
                    load_checkpoint, load_config, restore_rng, rng_state, seed_everything, seed_worker,
                    unwrap_model, write_json)

# Embedding-like tensors get no weight decay, like Mask2Former's WEIGHT_DECAY_EMBED=0. DINOv3 names
# its register tokens storage_tokens; register_tokens is kept for other ViT implementations.
NO_DECAY_NAMES = ("pos_embed", "cls_token", "register_tokens", "storage_tokens", "mask_token",
                  "level_embed", "level_encoding", "query_feat", "query_embed")
TRAIN_DEFAULTS = {"tf32": True, "cudnn_benchmark": True, "eval_stride": 1,
                  "sampling": {"mode": "uniform", "temperature": 0.01, "min_pixels": 1}}
# Only last.pt (and epoch_XXX.pt) hold these; best_miou.pt/best_loss.pt are inference-only.
RESUME_KEYS = ("optimizer", "scheduler", "scaler", "rng_states")
# Selection state saved in every checkpoint and restored on resume; recent_miou feeds miou_ma3.
BEST_KEYS = ("best_miou", "best_miou_epoch", "best_loss", "best_loss_epoch", "recent_miou")
MIOU_WINDOW = 3
DEVELOPMENT_NOTE = ("development only; final submission model must be trained on the organizer's "
                    "official sample list")


def make_optimizer(model, settings):
    # Only the original DINO transformer gets the multiplier. The ViT-Adapter trains at full LR.
    base_lr = float(settings.get("lr", 1e-4))
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        backbone = name.startswith("backbone.backbone.")
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


def resolve_train_settings(settings):
    """Fill defaults in place, so resolved_config.json and checkpoints record the values used."""
    if "ema" in settings:
        raise ValueError("train.ema was removed: training no longer keeps an EMA model (see docs/training.md). "
                         "Delete the train.ema block from the config.")
    for key, default in TRAIN_DEFAULTS.items():
        if isinstance(default, dict):
            value = settings.get(key)
            value = {} if value is None else value
            if not isinstance(value, dict) or set(value) - set(default):
                raise ValueError(f"train.{key} must be a mapping with keys among {sorted(default)}")
            settings[key] = {**default, **value}
        else:
            settings.setdefault(key, default)
    stride = settings["eval_stride"]
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("train.eval_stride must be a positive integer")
    if settings["sampling"]["mode"] not in ("uniform", "rare_class"):
        raise ValueError("train.sampling.mode must be uniform or rare_class")
    if float(settings.get("grad_clip", 0.1)) < 0:
        raise ValueError("train.grad_clip must be nonnegative; 0 disables clipping")
    return settings


def initial_best():
    return {"best_miou": None, "best_miou_epoch": None, "best_loss": None, "best_loss_epoch": None,
            "recent_miou": []}


def update_best(best, epoch, miou, val_loss):
    """Pure best-checkpoint bookkeeping for one validated epoch (1-based).

    Returns (new_best, improved, miou_ma3). improved lists the snapshots to write now:
    "best_miou" when mIoU is strictly higher than every earlier one, "best_loss" when the
    validation loss is strictly lower; a missing or non-finite value never counts. miou_ma3 is the
    mean mIoU of the last MIOU_WINDOW validations (None until there are that many), for
    information only: it never selects a checkpoint.
    """
    best = {**best, "recent_miou": list(best["recent_miou"])}
    improved, miou_ma3 = [], None
    if miou is not None and math.isfinite(miou):
        if best["best_miou"] is None or miou > best["best_miou"]:
            best["best_miou"], best["best_miou_epoch"] = miou, epoch
            improved.append("best_miou")
        best["recent_miou"] = (best["recent_miou"] + [miou])[-MIOU_WINDOW:]
        if len(best["recent_miou"]) == MIOU_WINDOW:
            miou_ma3 = sum(best["recent_miou"]) / MIOU_WINDOW
    if val_loss is not None and math.isfinite(val_loss):
        if best["best_loss"] is None or val_loss < best["best_loss"]:
            best["best_loss"], best["best_loss_epoch"] = val_loss, epoch
            improved.append("best_loss")
    return best, improved, miou_ma3


def resumed_best(checkpoint):
    """Selection state from last.pt; a legacy (EMA-era) checkpoint restarts it with a note."""
    if all(key in checkpoint for key in BEST_KEYS):
        return {key: checkpoint[key] for key in BEST_KEYS}, None
    legacy = {key: checkpoint.get(key) for key in ("best_miou", "best_source", "best_epoch")}
    return initial_best(), (f"legacy checkpoint ({legacy}): best_miou/best_loss tracking restarts at the next "
                            "validation; its EMA weights are dropped and an existing best.pt is left untouched")


def gradient_norm(parameters, max_norm):
    """Pre-clipping total L2 gradient norm; clips only when max_norm > 0."""
    parameters = [parameter for parameter in parameters if parameter.grad is not None]
    if max_norm > 0:
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    return torch.nn.utils.get_total_norm([parameter.grad for parameter in parameters])


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
    settings = resolve_train_settings(config["train"])
    backends = configure_torch_backends(settings)
    device, rank, world_size = init_distributed()
    seed = int(settings.get("seed", 42))
    seed_everything(seed + rank)
    data_config = config["data"]
    train_dataset = MoonDataset(data_config, "train")
    val_dataset = MoonDataset(data_config, "val") if settings.get("validate", True) else None
    # The dataset derives provenance from the manifest kind; data.development_pool is the opt-in flag.
    provenance = getattr(train_dataset, "provenance", "public_development_pool"
                         if data_config.get("development_pool", False) else "official_allowlist")
    development = provenance == "public_development_pool"
    allowlist = data_config.get("official_allowlist")
    data_snapshot = {"train_samples": train_dataset.records,
                     "validation_samples": val_dataset.records if val_dataset else [],
                     "official_allowlist": Path(allowlist).read_text(encoding="utf-8") if allowlist else None,
                     "provenance": provenance}
    if val_dataset is not None:
        train_ids = {str(row["sample_id"]) for row in train_dataset.records}
        val_ids = {str(row["sample_id"]) for row in val_dataset.records}
        if train_ids & val_ids:
            raise ValueError("Training and local validation manifests overlap")
        if set(data_config.get("train_scenes", [])) & set(data_config.get("val_scenes", [5])):
            raise ValueError("Scene-level train/validation split must be disjoint")
    batch_plan = training_batch_plan(len(train_dataset), world_size, settings)
    batch_size = batch_plan["batch_per_gpu"]
    accumulation = batch_plan["accumulation_steps"]
    drop_last = batch_plan["drop_incomplete_updates"]
    if settings["sampling"]["mode"] == "rare_class":
        sampler, batch_plan["sampling"] = build_rare_class_sampler(
            train_dataset.records, data_config.get("label_stats"), settings["sampling"],
            int(config.get("model", {}).get("num_classes", 5)), world_size, rank, seed, drop_last)
    else:
        sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank,
                                     shuffle=True, seed=seed, drop_last=drop_last)
        batch_plan["sampling"] = {"mode": "uniform"}
    workers = int(settings.get("num_workers", 4))
    loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                        num_workers=workers, collate_fn=collate_samples, pin_memory=device.type == "cuda",
                        worker_init_fn=seed_worker, persistent_workers=False, drop_last=drop_last)
    resume_path = args.resume or settings.get("resume")
    from .models import build_criterion, build_model
    # The pretrained backbone always comes from model.backbone_weights; checkpoints omit frozen tensors.
    model = build_model(config, load_backbone=True).to(device)
    criterion = build_criterion(config).to(device)
    optimizer = make_optimizer(model, settings)
    epochs = int(settings.get("epochs", 80))
    total_steps = epochs * batch_plan["optimizer_steps_per_epoch"]
    scheduler = make_scheduler(optimizer, total_steps, settings)
    amp = settings.get("amp", "bf16")
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp == "fp16")
    grad_clip = float(settings.get("grad_clip", 0.1))
    start_epoch, global_step, best = 0, 0, initial_best()
    if resume_path:
        checkpoint = load_checkpoint(resume_path)
        absent = [key for key in RESUME_KEYS if key not in checkpoint]
        if absent:
            raise ValueError(f"{resume_path} cannot be resumed: it has no {', '.join(absent)} state. best_miou.pt and "
                             "best_loss.pt (and the legacy best.pt) are inference-only snapshots; resume from last.pt.")
        if checkpoint["world_size"] != world_size:
            raise ValueError("Epoch-boundary resume requires the checkpoint's WORLD_SIZE")
        if checkpoint["total_steps"] != total_steps:
            raise ValueError("Epoch-boundary resume requires unchanged epochs, batch size, accumulation and dataset length")
        if checkpoint.get("batch_plan", batch_plan) != batch_plan:
            raise ValueError("Epoch-boundary resume requires an unchanged effective batch, incomplete-update policy and sampling")
        for section in ("model", "loss", "data"):
            if checkpoint["config"].get(section) != config.get(section):
                raise ValueError(f"Epoch-boundary resume requires unchanged {section} configuration")
        if checkpoint["data_snapshot"] != data_snapshot:
            raise ValueError("Manifest sample membership, official allowlist or data provenance changed since checkpoint")
        # Legacy checkpoints also recorded the removed train.ema; its weights are not restored.
        transient = {"resume", "output_dir", "log_every", "save_every", "ema"}
        old_settings = {key: value for key, value in checkpoint["config"]["train"].items() if key not in transient}
        new_settings = {key: value for key, value in settings.items() if key not in transient}
        if old_settings != new_settings:
            raise ValueError("Epoch-boundary resume requires unchanged training configuration (except resume/output/log/save paths)")
        if checkpoint["config"].get("inference") != config.get("inference"):
            raise ValueError("Epoch-boundary resume requires unchanged validation inference settings")
        load_partial_state_dict(model, checkpoint["model"], "model")
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
        best, legacy_note = resumed_best(checkpoint)
        if legacy_note and rank == 0:
            print(f"NOTE: {resume_path} is a {legacy_note}.", flush=True)
        restore_rng(checkpoint["rng_states"][rank])
        del checkpoint
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None,
                                        find_unused_parameters=False, broadcast_buffers=True)
    output = Path(settings.get("output_dir", "runs/moonseg"))
    if rank == 0:
        runtime = {"deformable_attention": unwrap_model(model).deformable_attention,
                   "torch": torch.__version__, **backends}
        output.mkdir(parents=True, exist_ok=True)
        write_json({**config, "runtime": runtime}, output / "resolved_config.json")
        write_json(batch_plan, output / "batch_plan.json")
        note = (DEVELOPMENT_NOTE if development else
                "Only explicit official allowlist samples admitted; public discovery grants no training permission.")
        write_json({**data_snapshot, "note": note}, output / "data_provenance.json")
        if development:
            print(f"WARNING: data.development_pool=true, training on the public development pool: {DEVELOPMENT_NOTE}.",
                  flush=True)
        print(f"Training {len(train_dataset)} {'development-pool' if development else 'official'} samples; "
              f"world_size={world_size}; effective_batch={batch_size * world_size * accumulation}; device={device}; "
              f"amp={amp}; deformable_attention={runtime['deformable_attention']}; tf32={backends['tf32']}; "
              f"cudnn_benchmark={backends['cudnn_benchmark']}; "
              f"sampling={settings['sampling']['mode']}; eval_stride={settings['eval_stride']}", flush=True)
        print(json.dumps(batch_plan), flush=True)
    for epoch in range(start_epoch, epochs):
        sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_loss = torch.zeros(2, device=device, dtype=torch.float64)
        grad_stats = torch.zeros(3, device=device, dtype=torch.float64)  # sum, max, count of applied updates
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
                # DDP has averaged gradients, so the norm is identical on every rank.
                norm = gradient_norm(model.parameters(), grad_clip).detach().double()
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
                    global_step += 1
                    grad_stats[0] += norm
                    grad_stats[1] = torch.maximum(grad_stats[1], norm)
                    grad_stats[2] += 1
                optimizer.zero_grad(set_to_none=True)
                if rank == 0 and global_step % int(settings.get("log_every", 20)) == 0:
                    print(json.dumps({"epoch": epoch + 1, "step": global_step,
                                      "loss": float(loss.detach()), "grad_norm": float(norm),
                                      "lr": scheduler.get_last_lr(),
                                      "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
            # Loop locals otherwise keep the previous batch and its loss graph alive during
            # the next forward, and keep the final batch throughout validation/checkpointing.
            del batch, images, targets, losses, loss
        if dist.is_initialized():
            dist.all_reduce(epoch_loss)
            # Adapter BatchNorm updates running statistics locally. Validate and save one consistent
            # model; unwrapped validation forwards and the unsynchronized validation loss contain
            # no collectives (evaluate_model reduces its sums once at the end).
            for buffer in unwrap_model(model).buffers():
                dist.broadcast(buffer, src=0)
        result = stride = evaluated = miou_ma3 = None
        improved = []
        if val_dataset is not None and ((epoch + 1) % int(settings.get("eval_every", 1)) == 0 or epoch + 1 == epochs):
            # Intermediate validations may use every k-th record; the final epoch evaluates all.
            stride = 1 if epoch + 1 == epochs else settings["eval_stride"]
            indices = validation_indices(val_dataset, stride)
            evaluated = len(indices)
            inference = config.get("inference", {})
            result = evaluate_model(model, val_dataset, device, inference, amp, rank, world_size, workers, indices,
                                    criterion)
            best, improved, miou_ma3 = update_best(best, epoch + 1, result["mIoU"], result["val_loss"])
        grad_sum, grad_max, grad_count = grad_stats.tolist()
        states = _gather_rng(world_size)
        if rank == 0:
            summary = {"epoch": epoch + 1, "global_step": global_step,
                       "train_loss": (epoch_loss[0] / epoch_loss[1].clamp_min(1)).item(),
                       "grad_norm_mean": grad_sum / grad_count if grad_count else None,
                       "grad_norm_max": grad_max if grad_count else None,
                       "seconds": time.monotonic() - started, "eval_stride": stride, "evaluated_samples": evaluated,
                       "val_loss": result["val_loss"] if result else None, "miou_ma3": miou_ma3,
                       **{key: best[key] for key in BEST_KEYS if key != "recent_miou"},
                       "saved_best": improved, "validation": result}
            with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(summary, ensure_ascii=False) + "\n")
            print(json.dumps(summary, ensure_ascii=False), flush=True)
            state = {"model": partial_state_dict(model), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                     "rng_states": states, "epoch": epoch, "global_step": global_step, **best,
                     "world_size": world_size, "total_steps": total_steps,
                     "config": config, "sampler_epoch": epoch, "data_snapshot": data_snapshot, "batch_plan": batch_plan,
                     "frozen_backbone": {"weights": config["model"].get("backbone_weights"),
                                         "note": "Frozen DINOv3 tensors are not stored; rebuild the model with "
                                                 "model.backbone_weights before loading these weights."},
                     "resume_scope": "Complete state at epoch boundary; use same config/world size. "
                                     "CUDA deformable-attention and grid_sample backward are nondeterministic; "
                                     "bitwise replay is not guaranteed."}
            atomic_torch_save(state, output / "last.pt")
            # Inference-only snapshots (no optimizer/scheduler/scaler/RNG): best_miou.pt, best_loss.pt.
            inference_state = {key: value for key, value in state.items() if key not in RESUME_KEYS}
            for name in improved:
                atomic_torch_save({**inference_state, "selected_by": name,
                                   "resume_scope": "Weights only (model); resume training from last.pt."},
                                  output / f"{name}.pt")
            save_every = int(settings.get("save_every", 0))
            if save_every > 0 and (epoch + 1) % save_every == 0:
                atomic_torch_save(state, output / f"epoch_{epoch + 1:03d}.pt")
            # state_dict tensors alias live model/optimizer storage; do not retain a snapshot's
            # references after saving (including buffers that a later forward may replace).
            del inference_state, state
        if dist.is_initialized():
            dist.barrier()
        if device.type == "cuda":
            # Validation uses batch 1, training uses larger batches: release unused cached
            # blocks at the phase boundary so they do not fragment the next training forward.
            memory = {"allocated": torch.cuda.memory_allocated(device),
                      "reserved_before_release": torch.cuda.memory_reserved(device),
                      "peak_allocated": torch.cuda.max_memory_allocated(device),
                      "peak_reserved": torch.cuda.max_memory_reserved(device)}
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info(device)
            memory.update(reserved=torch.cuda.memory_reserved(device), free=free, total=total)
            print(json.dumps({"epoch_boundary": epoch + 1, "rank": rank,
                              "cuda_memory_bytes": memory}), flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
