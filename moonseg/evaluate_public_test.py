"""Offline research metrics on public LuSNAR scenes 2/4/9 (not an official score).

This dataset is deliberately separate from MoonDataset and its train/val admission.
Labels reach only the confusion metrics; there is no loss, adaptation or training.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .data.labels import decode_label
from .evaluate import evaluate_model
from .predict import select_weights
from .utils import configure_torch_backends, load_checkpoint, load_config, load_partial_state_dict

MANIFEST_KIND = "public_test_research_manifest"
SCENES = (2, 4, 9)
EXPECTED_SAMPLES = 5004
SCENE_COUNTS = {"2": 1597, "4": 1320, "9": 2087}
PROFILES = {
    "single": {"sliding_window": False, "scales": [1.0], "flip": False},
    "narrowflip": {"sliding_window": False, "scales": [0.9, 1.0, 1.1], "flip": True},
}
CHECKPOINT_POLICIES = ("validation_selected", "final_epoch")
FULLTRAIN_SCENES = [1, 3, 5, 6, 7, 8]
FULLTRAIN_SAMPLES = 8002
FULLTRAIN_UPDATES_PER_EPOCH = 125


def fulltrain_data_snapshot(config):
    """Read the exact six-scene training membership, without constructing a val dataset."""
    from .data.manifest import (is_allowed, load_manifest, manifest_provenance,
                                read_allowlist, require_complete_allowlist)

    data = config["data"]
    if (config["train"].get("validate") is not False or data.get("val_manifest") is not None
            or data.get("val_scenes") != [] or data.get("train_scenes") != FULLTRAIN_SCENES):
        raise ValueError("No-validation full training requires six train scenes and no validation split")
    records = load_manifest(data["train_manifest"], Path(data["root"]).resolve(), FULLTRAIN_SCENES,
                            data.get("official_allowlist"), need_label=True,
                            allow_development_pool=data.get("development_pool", False))
    if len(records) != FULLTRAIN_SAMPLES or sorted({record["scene"] for record in records}) != FULLTRAIN_SCENES:
        raise ValueError("No-validation full training requires all 8002 six-scene samples")
    # The public-pool config retains its provenance; membership is independently
    # checked against the organizer's downloaded training list.
    official = read_allowlist("data/LuSNAR_official/official_train_allowlist.json")
    require_complete_allowlist(records, official)
    if any(not is_allowed(record, official) for record in records):
        raise ValueError("Full-training membership differs from the official training allowlist")
    stats = json.loads(Path(data["label_stats"]).read_text())
    if (stats.get("num_samples") != FULLTRAIN_SAMPLES or stats.get("scenes") != FULLTRAIN_SCENES
            or Path(stats.get("manifest", "")).resolve() != Path(data["train_manifest"]).resolve()):
        raise ValueError("Full-training label statistics must cover the same 8002 six-scene samples")
    allowlist = data.get("official_allowlist")
    return {"train_samples": records, "validation_samples": [],
            "official_allowlist": Path(allowlist).read_text(encoding="utf-8") if allowlist else None,
            "provenance": manifest_provenance(data["train_manifest"])}


def validate_no_validation_history(rows, epochs):
    """Every epoch must have training updates and null validation/selection fields."""
    if [row.get("epoch") for row in rows] != list(range(1, epochs + 1)):
        raise ValueError(f"No-validation history requires exactly epochs 1..{epochs}")
    null_fields = ("validation", "eval_stride", "evaluated_samples", "val_loss", "miou_ma3",
                   "best_miou", "best_miou_epoch", "best_loss", "best_loss_epoch")
    for epoch, row in enumerate(rows, 1):
        if type(row.get("global_step")) is not int or row["global_step"] != epoch * FULLTRAIN_UPDATES_PER_EPOCH:
            raise ValueError("No-validation history must contain exactly 125 optimizer updates per epoch")
        if any(key not in row or row[key] is not None for key in null_fields) or row.get("saved_best") != []:
            raise ValueError("No-validation training history contains validation or checkpoint selection")


def validate_no_validation_checkpoint(config, path, checkpoint, rows):
    """Require complete training and data receipts for the fixed full-training recipe."""
    from .train import training_batch_plan

    settings = config["train"]
    snapshot = fulltrain_data_snapshot(config)
    expected_batch = training_batch_plan(FULLTRAIN_SAMPLES, 2, settings)
    if settings.get("sampling", {}).get("mode", "uniform") != "uniform":
        raise ValueError("No-validation full training requires uniform sampling")
    expected_batch["sampling"] = {"mode": "uniform"}
    if (expected_batch["optimizer_steps_per_epoch"] != FULLTRAIN_UPDATES_PER_EPOCH
            or expected_batch["effective_batch"] != 64 or checkpoint.get("world_size") != 2
            or checkpoint.get("batch_plan") != expected_batch):
        raise ValueError("No-validation checkpoint must preserve the two-GPU batch64/125-update recipe")
    validate_no_validation_history(rows, settings["epochs"])
    if checkpoint.get("total_steps") != settings["epochs"] * FULLTRAIN_UPDATES_PER_EPOCH:
        raise ValueError("No-validation checkpoint has an incorrect total optimizer-step budget")
    if checkpoint.get("data_snapshot") != snapshot:
        raise ValueError("No-validation checkpoint data provenance differs from the full training manifest")
    if (settings.get("resume") is not None or settings.get("initialize_from") is not None
            or checkpoint.get("stage_initialization") is not None or checkpoint.get("selected_by") is not None):
        raise ValueError("No-validation runs must start independently from native pretraining with fresh Adapter/head")
    if (any(checkpoint.get(key) is not None for key in
            ("best_miou", "best_miou_epoch", "best_loss", "best_loss_epoch"))
            or checkpoint.get("recent_miou") != []):
        raise ValueError("No-validation checkpoint contains validation-based selection state")
    directory = Path(path).parent
    if any((directory / name).exists() for name in ("best.pt", "best_miou.pt", "best_loss.pt", "stage_initialization.json")):
        raise ValueError("No-validation run contains an unexpected selected/stage checkpoint")
    provenance = json.loads((directory / "data_provenance.json").read_text())
    if {key: provenance.get(key) for key in snapshot} != snapshot:
        raise ValueError("No-validation run data_provenance.json differs from checkpoint provenance")
    if json.loads((directory / "batch_plan.json").read_text()) != expected_batch:
        raise ValueError("No-validation run batch_plan.json differs from checkpoint batch plan")


def checkpoint_policy_metadata(policy):
    if policy not in CHECKPOINT_POLICIES:
        raise ValueError(f"Unknown checkpoint policy: {policy}")
    selection = (
        "Fixed-budget final checkpoint after all planned training epochs; no checkpoint score selection."
        if policy == "final_epoch" else
        "Existing local-validation checkpoint, fixed before public-test evaluation; "
        "public-test metrics do not select checkpoints.")
    return {"checkpoint_policy": policy, "checkpoint_selection": selection,
            "comparison_scope": "Public development comparison; not an independent holdout."}


def validate_final_checkpoint(config, checkpoint_path, checkpoint, expected_epochs=None):
    """Validate trusted CPU checkpoint metadata before model construction or CUDA use."""
    path = Path(checkpoint_path)
    if path.name != "last.pt":
        raise ValueError("final_epoch policy requires last.pt")
    epochs = config.get("train", {}).get("epochs")
    if type(epochs) is not int or epochs < 1 or (expected_epochs is not None and epochs != expected_epochs):
        raise ValueError("Final checkpoint requires the planned resolved train.epochs")
    epoch, step = checkpoint.get("epoch"), checkpoint.get("global_step")
    if type(epoch) is not int or epoch + 1 != epochs or type(step) is not int or step < 1:
        raise ValueError("Final checkpoint epoch/global_step does not complete resolved train.epochs")
    if not isinstance(checkpoint.get("model"), dict):
        raise ValueError("Final checkpoint has no model weights")
    saved = checkpoint.get("config", {})
    for section in ("data", "model", "loss", "inference"):
        if not isinstance(saved, dict) or saved.get(section) != config.get(section):
            raise ValueError(f"Final checkpoint differs from resolved {section} configuration")
    transient = {"resume", "output_dir", "log_every", "save_every", "ema"}
    old_train = saved.get("train")
    if not isinstance(old_train, dict) or (
            {key: value for key, value in old_train.items() if key not in transient}
            != {key: value for key, value in config["train"].items() if key not in transient}):
        raise ValueError("Final checkpoint differs from resolved critical train configuration")
    rows = [json.loads(line) for line in (path.parent / "history.jsonl").read_text().splitlines() if line.strip()]
    if [row.get("epoch") for row in rows] != list(range(1, epochs + 1)):
        raise ValueError(f"Final checkpoint requires complete history epochs 1..{epochs}")
    final = rows[-1]
    if type(final.get("global_step")) is not int or final["global_step"] != step:
        raise ValueError("Final checkpoint global_step differs from final training history")
    if config["train"].get("validate", True) is False:
        validate_no_validation_checkpoint(config, path, checkpoint, rows)
    elif (final.get("evaluated_samples") != 1599 or final.get("eval_stride") != 1
          or not final.get("validation")):
        raise ValueError("Final checkpoint history requires full scene-5 validation")
    if "total_steps" in checkpoint and checkpoint["total_steps"] != step:
        raise ValueError("Final checkpoint has not completed its fixed optimizer-step budget")


def file_identity(path):
    stat = Path(path).stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino}


class PublicTestResearchDataset(Dataset):
    """Exact public RGB images and palette labels; normalize, never augment."""

    def __init__(self, root, manifest, normalization):
        self.root = Path(root).resolve()
        self.manifest_path = Path(manifest).resolve()
        value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if value.get("kind") != MANIFEST_KIND or value.get("scenes") != list(SCENES):
            raise ValueError("Expected an explicit public_test_research_manifest for scenes [2,4,9]")
        records = value.get("samples", value.get("records"))
        if not isinstance(records, list) or len(records) != EXPECTED_SAMPLES:
            raise ValueError(f"Research evaluation requires all {EXPECTED_SAMPLES} public samples")
        seen = set()
        for record in records:
            scene, sample = record["scene"], str(record["sample_id"])
            if scene not in SCENES or not sample or Path(sample).name != sample or sample in {".", ".."}:
                raise ValueError(f"Invalid research sample: {record}")
            key = (scene, sample)
            if key in seen:
                raise ValueError(f"Duplicate research sample: {key}")
            seen.add(key)
            for field, directory in (("image", "color"), ("label", "label")):
                relative = Path(f"Moon_{scene}/{directory}/{sample}.png")
                if Path(record[field]) != relative:
                    raise ValueError(f"Unexpected public sample path: {record[field]}")
                path = (self.root / relative).resolve()
                # RGB directories intentionally link to the existing public image pool.
                # Labels must remain within this separate research root.
                if (field == "label" and not path.is_relative_to(self.root)) or not path.is_file():
                    raise ValueError(f"Missing or external research sample: {path}")
        self.records = sorted(records, key=lambda item: (int(item["scene"]), str(item["sample_id"])))
        self.scene_counts = dict(Counter(str(item["scene"]) for item in records))
        if self.scene_counts != SCENE_COUNTS:
            raise ValueError(f"Expected complete public scene counts {SCENE_COUNTS}")
        self.mean = np.asarray(normalization.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        self.std = np.asarray(normalization.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)
        if (self.mean.shape != (3,) or self.std.shape != (3,) or not np.isfinite(self.mean).all()
                or not np.isfinite(self.std).all() or np.any(self.std <= 0)):
            raise ValueError("Normalization requires three finite means and positive standard deviations")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(self.root / record["image"]) as handle:
            pixels = np.asarray(handle.convert("RGB"), dtype=np.float32)
        with Image.open(self.root / record["label"]) as handle:
            target = decode_label(handle)
        if pixels.shape[:2] != target.shape:
            raise ValueError(f"Image/label dimensions differ: {record['sample_id']}")
        pixels = (pixels / 255.0 - self.mean) / self.std
        return {"image": torch.from_numpy(pixels.transpose(2, 0, 1).copy()),
                "target": torch.from_numpy(target.astype(np.int64)),
                "sample_id": str(record["sample_id"]), "scene": int(record["scene"])}


def research_result(metrics, dataset, checkpoint_policy="validation_selected"):
    metrics = dict(metrics)
    metrics.pop("val_loss", None)
    pooled = {key: value for key, value in metrics.items() if key not in {"scenes", "evaluated_samples"}}
    for scene, count in dataset.scene_counts.items():
        metrics["scenes"][scene]["evaluated_samples"] = count
    metrics.update(
        pooled=pooled,
        scene_equal_macro={key: sum(metrics["scenes"][scene][key] for scene in map(str, SCENES)) / 3
                           for key in ("mIoU", "mAcc")},
        scene_sample_counts=dataset.scene_counts,
        evaluation_kind=MANIFEST_KIND,
        official_score=False,
        metric_scope="Offline public-test research only. Pooled pixels and equal scene macro are reported; "
                     "neither is the official organizer-weighted total.",
        **checkpoint_policy_metadata(checkpoint_policy),
        data_root=str(dataset.root), manifest=str(dataset.manifest_path),
        label_usage="Confusion metrics only; no loss, training, BN adaptation or parameter updates.")
    return metrics


def write_new_json(path, value):
    """Publish a complete score exclusively; never replace a previous score."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temp.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Original resolved training config")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-policy", choices=CHECKPOINT_POLICIES, default="validation_selected")
    parser.add_argument("--weights", choices=["model"], default="model")
    parser.add_argument("--data-root", default="data/research_test/LuSNAR")
    parser.add_argument("--manifest", default="data/research_test/manifest.json")
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(f"Refusing to overwrite scores: {args.output}")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Research evaluator runs as one process per GPU")
    config = load_config(args.config)
    if args.checkpoint_policy == "final_epoch" and Path(args.checkpoint).name != "last.pt":
        raise ValueError("final_epoch policy requires last.pt")
    identity = file_identity(args.checkpoint)
    checkpoint = load_checkpoint(args.checkpoint)
    if file_identity(args.checkpoint) != identity:
        raise ValueError("Checkpoint changed while loading")
    if args.checkpoint_policy == "final_epoch":
        validate_final_checkpoint(config, args.checkpoint, checkpoint)
    configure_torch_backends(config.get("train", {}))
    dataset = PublicTestResearchDataset(args.data_root, args.manifest, config["data"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    from .models import build_model
    model = build_model(config, load_backbone=True).to(device)
    state, weights, message = select_weights(checkpoint, "model")
    load_partial_state_dict(model, state, weights)
    print(f"{message}; public-test RESEARCH ONLY; checkpoint={args.checkpoint}", flush=True)
    model.eval()
    metrics = evaluate_model(model, dataset, device, PROFILES[args.profile],
                             config.get("train", {}).get("amp", "none"),
                             num_workers=config.get("train", {}).get("num_workers", 2), criterion=None)
    if file_identity(args.checkpoint) != identity:
        raise ValueError("Checkpoint changed during evaluation")
    result = research_result(metrics, dataset, args.checkpoint_policy)
    result.update(checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_epoch=checkpoint.get("epoch", -1) + 1,
                  global_step=checkpoint.get("global_step"), checkpoint_file_identity=identity,
                  selected_by=checkpoint.get("selected_by"), weights=weights,
                  profile=args.profile, inference=PROFILES[args.profile])
    write_new_json(args.output, result)
    print(json.dumps({key: result[key] for key in ("evaluated_samples", "mIoU", "mAcc", "scene_equal_macro")}))


if __name__ == "__main__":
    main()
