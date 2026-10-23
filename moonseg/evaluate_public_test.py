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


def research_result(metrics, dataset):
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
        checkpoint_selection="Existing local-validation checkpoint, fixed before public-test evaluation; "
                             "public-test metrics do not select checkpoints.",
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
    configure_torch_backends(config.get("train", {}))
    dataset = PublicTestResearchDataset(args.data_root, args.manifest, config["data"])
    identity = file_identity(args.checkpoint)
    checkpoint = load_checkpoint(args.checkpoint)
    if file_identity(args.checkpoint) != identity:
        raise ValueError("Checkpoint changed while loading")
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
    result = research_result(metrics, dataset)
    result.update(checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_epoch=checkpoint.get("epoch", -1) + 1,
                  global_step=checkpoint.get("global_step"), checkpoint_file_identity=identity,
                  selected_by=checkpoint.get("selected_by"), weights=weights,
                  profile=args.profile, inference=PROFILES[args.profile])
    write_new_json(args.output, result)
    print(json.dumps({key: result[key] for key in ("evaluated_samples", "mIoU", "mAcc", "scene_equal_macro")}))


if __name__ == "__main__":
    main()
