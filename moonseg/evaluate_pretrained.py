"""Inference-only SAM3 research metrics on all public LuSNAR scenes 2/4/9."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch

from .data.labels import CLASS_NAMES
from .evaluate import evaluate_model
from .evaluate_public_test import (EXPECTED_SAMPLES, MANIFEST_KIND, PROFILES, SCENES, SCENE_COUNTS,
                                   PublicTestResearchDataset, file_identity, write_new_json)
from .metrics import SegmentationMetrics
from .models.sam3 import PROMPTS
from .utils import amp_context, configure_torch_backends, load_config

SOURCE_COMMIT = "f6e51f59500a87c576c2df2323ce56b9fd7a12de"
SOURCE_REPOSITORY = "https://github.com/facebookresearch/sam3"
ROOT = Path(__file__).resolve().parents[1]
PREPROCESSING = {
    "input": "Project-normalized RGB restored to rounded uint8 RGB before the original Sam3Processor.",
    "processor": "Official Sam3Processor",
    "internal_resize": [1008, 1008],
    "internal_mean": [0.5, 0.5, 0.5],
    "internal_std": [0.5, 0.5, 0.5],
    "score_postprocess": "Each fixed prompt's dense semantic_seg logits are bilinearly resized to input "
                         "height/width, then sigmoid; the five class scores are combined by argmax.",
}
POLICY = {
    "checkpoint_policy": "pretrained_only",
    "checkpoint_selection": "Original pretrained SAM3 weights; no training, adaptation or checkpoint score selection.",
    "comparison_scope": "Public development comparison; not an independent holdout.",
    "weights": "pretrained",
}


def validate_config(config):
    model = config.get("model", {})
    if (model.get("architecture") != "sam3" or model.get("num_classes", 5) != 5
            or model.get("prompts", list(PROMPTS)) != list(PROMPTS)
            or model.get("freeze_backbone", True) is not True):
        raise ValueError("Use the frozen official SAM3 model with the five fixed predeclared class prompts")
    if "train" in config or "loss" in config:
        raise ValueError("Inference-only SAM3 has no training/loss section; place execution settings in runtime")
    if config.get("inference") != PROFILES["single"]:
        raise ValueError("SAM3 pretrained evaluation admits only the single profile, without TTA")
    runtime = {"amp": "bf16", "num_workers": 2, "tf32": True, "cudnn_benchmark": True,
               **config.get("runtime", {})}
    if (set(runtime) != {"amp", "num_workers", "tf32", "cudnn_benchmark"}
            or runtime["amp"] not in {"none", "bf16", "fp16"}
            or type(runtime["num_workers"]) is not int or runtime["num_workers"] < 0
            or any(type(runtime[key]) is not bool for key in ("tf32", "cudnn_benchmark"))):
        raise ValueError("runtime requires amp, nonnegative num_workers and boolean torch backend settings")
    if not isinstance(config.get("data"), dict):
        raise ValueError("Record the project's input normalization in config.data")
    return runtime


def asset_metadata(config):
    """Read first-party provenance without constructing the model or touching CUDA."""
    model = config["model"]
    source = Path(model.get("sam3_repo", ROOT / "third_party/sam3")).resolve()
    manifest_path = source / "SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("repository") != SOURCE_REPOSITORY or manifest.get("revision") != SOURCE_COMMIT
            or manifest.get("modifications") != []):
        raise ValueError("Require the pinned, unmodified official facebookresearch/sam3 source")
    for name in manifest["files"]:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not (source / relative).is_file():
            raise ValueError(f"Missing or invalid original SAM3 source asset: {name}")
    weights = Path(model["backbone_weights"]).expanduser().resolve()
    identity = file_identity(weights)
    if identity["size"] <= 0:
        raise ValueError("SAM3 original pretrained weights must be a complete nonempty local file")
    return {"pretrained_weights": str(weights), "pretrained_file_identity": identity,
            "source_repository": SOURCE_REPOSITORY, "source_commit": SOURCE_COMMIT,
            "source_manifest": str(manifest_path), "source_manifest_file_identity": file_identity(manifest_path),
            "bpe_file_identity": file_identity(source / "sam3/assets/bpe_simple_vocab_16e6.txt.gz")}


def pretrained_result(metrics, dataset, config, config_path, config_identity, assets, runtime, smoke):
    result = dict(metrics)
    result.pop("val_loss", None)
    pooled = {key: value for key, value in result.items() if key not in {"scenes", "evaluated_samples"}}
    for scene, count in dataset.scene_counts.items():
        result["scenes"][scene]["evaluated_samples"] = count
    result.update(pooled=pooled,
        scene_equal_macro={key: sum(result["scenes"][scene][key] for scene in map(str, SCENES)) / 3
                           for key in ("mIoU", "mAcc")},
        scene_sample_counts=dataset.scene_counts, evaluation_kind=MANIFEST_KIND, official_score=False,
        metric_scope="Offline public-test research only. Pooled pixels and equal scene macro are reported; "
                     "neither is the official organizer-weighted total.",
        data_root=str(dataset.root), manifest=str(dataset.manifest_path),
        label_usage="Confusion metrics only; no loss, training, BN adaptation or parameter updates.",
        config_path=str(Path(config_path).resolve()), config_file_identity=config_identity, resolved_config=config,
        profile="single", inference=PROFILES["single"], class_names=list(CLASS_NAMES), prompts=list(PROMPTS),
        preprocessing=PREPROCESSING, runtime=runtime, single_image_smoke=smoke,
        protocol="Project adaptation of original SAM3 text-conditioned dense semantic scores; "
                 "not a reproduction of an undisclosed LuSNAR SAM3 protocol.", **POLICY, **assets)
    return result


def validate_pretrained_result(value, config, config_path, data_root, manifest, assets=None):
    """Strict standalone result contract, independent of trained-checkpoint rules."""
    runtime = validate_config(config)
    assets = asset_metadata(config) if assets is None else assets
    expected = {**POLICY, **assets, "resolved_config": config, "profile": "single", "inference": PROFILES["single"],
                "class_names": list(CLASS_NAMES), "prompts": list(PROMPTS), "preprocessing": PREPROCESSING,
                "runtime": runtime, "evaluation_kind": MANIFEST_KIND, "official_score": False,
                "evaluated_samples": EXPECTED_SAMPLES, "scene_sample_counts": SCENE_COUNTS,
                "config_path": str(Path(config_path).resolve()), "config_file_identity": file_identity(config_path),
                "data_root": str(Path(data_root).resolve()), "manifest": str(Path(manifest).resolve())}
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise ValueError("Pretrained SAM3 scores differ from the frozen config/assets/profile/public-data provenance")
    forbidden = {"checkpoint", "checkpoint_epoch", "epoch", "global_step", "selected_by", "val_loss",
                 "best_miou", "best_loss", "checkpoint_file_identity"}
    if forbidden.intersection(value):
        raise ValueError("Pretrained-only scores must not contain training checkpoint, epoch, selection or loss metadata")
    if set(value.get("scenes", {})) != set(SCENE_COUNTS):
        raise ValueError("Pretrained-only scores must contain exactly the complete public scenes 2/4/9")
    matrices = []
    for name, metrics in [("pooled", value), *value["scenes"].items()]:
        matrix = metrics.get("confusion_matrix", [])
        if (not isinstance(matrix, list) or len(matrix) != 5
                or any(not isinstance(row, list) or len(row) != 5 for row in matrix)
                or any(type(count) is not int or count < 0 for row in matrix for count in row)):
            raise ValueError(f"Invalid five-class confusion counts: {name}")
        recomputed = SegmentationMetrics()
        recomputed.confusion = torch.tensor(matrix, dtype=torch.int64)
        actual = recomputed.compute()
        for key in ("mIoU", "mAcc"):
            metric = metrics.get(key)
            if (isinstance(metric, bool) or not isinstance(metric, (float, int)) or not math.isfinite(metric)
                    or not 0 <= metric <= 1 or actual[key] is None or not math.isclose(metric, actual[key])):
                raise ValueError(f"Invalid or inconsistent {name} {key}")
        ious = metrics.get("class_IoU", {})
        if set(ious) != set(CLASS_NAMES) or any(
                (ious[key] is not None if reference is None else
                 isinstance(ious[key], bool) or not isinstance(ious[key], (float, int))
                 or not math.isfinite(ious[key]) or not math.isclose(ious[key], reference))
                for key, reference in actual["class_IoU"].items()):
            raise ValueError(f"Inconsistent five-class IoU: {name}")
        if name != "pooled":
            if metrics.get("evaluated_samples") != SCENE_COUNTS[name]:
                raise ValueError(f"Incomplete pretrained evaluation of scene {name}")
            matrices.append(recomputed.confusion)
    if not torch.equal(sum(matrices), torch.tensor(value["confusion_matrix"], dtype=torch.int64)):
        raise ValueError("Pooled pretrained confusion counts differ from the three scenes")
    if value.get("pooled", {}).get("confusion_matrix") != value["confusion_matrix"]:
        raise ValueError("Pooled pretrained metric record differs from the complete confusion counts")
    for key in ("mIoU", "mAcc"):
        macro = value.get("scene_equal_macro", {}).get(key)
        if (value["pooled"].get(key) != value[key] or isinstance(macro, bool)
                or not isinstance(macro, (float, int)) or not math.isfinite(macro)
                or not math.isclose(macro, sum(value["scenes"][scene][key] for scene in SCENE_COUNTS) / 3)):
            raise ValueError("Inconsistent pooled or equal-scene pretrained metrics")
    smoke = value.get("single_image_smoke", {})
    shape = smoke.get("input_shape", [])
    if (smoke.get("passed") is not True or smoke.get("scene") != 2
            or not isinstance(smoke.get("sample_id"), str) or not smoke["sample_id"]
            or not isinstance(shape, list) or len(shape) != 4 or shape[:2] != [1, 3]
            or any(type(length) is not int or length <= 0 for length in shape)
            or smoke.get("output_shape") != [1, 5, *shape[2:]]):
        raise ValueError("Missing successful real first-image five-class inference smoke")
    return value


@torch.inference_mode()
def smoke_first_image(model, dataset, device, amp):
    sample = dataset[0]
    image = sample["image"].unsqueeze(0).to(device)
    with amp_context(device, amp):
        outputs = model(image)
    scores = outputs.get("semantic_scores")
    expected = (1, 5, *image.shape[-2:])
    if (scores is None or tuple(scores.shape) != expected or not torch.isfinite(scores).all()
            or scores.min() < 0 or scores.max() > 1):
        raise ValueError("First-image SAM3 smoke requires finite sigmoid scores with shape 1x5xHxW")
    result = {"passed": True, "sample_id": sample["sample_id"], "scene": sample["scene"],
              "input_shape": list(image.shape), "output_shape": list(scores.shape)}
    print(json.dumps({"single_image_smoke": result}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default="data/research_test/LuSNAR")
    parser.add_argument("--manifest", default="data/research_test/manifest.json")
    parser.add_argument("--profile", choices=["single"], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(f"Refusing to overwrite pretrained scores: {args.output}")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Pretrained research evaluation uses exactly one process/GPU")
    identity = file_identity(args.config)
    config = load_config(args.config)
    runtime = validate_config(config)
    assets = asset_metadata(config)
    if file_identity(args.config) != identity:
        raise ValueError("Pretrained config changed while reading it")
    if not torch.cuda.is_available():
        raise RuntimeError("Official SAM3 inference requires an available CUDA GPU. CPU-only source/weight "
                           "audits remain supported; the original geometry encoder uses CUDA pinned memory.")
    configure_torch_backends(runtime)
    dataset = PublicTestResearchDataset(args.data_root, args.manifest, config["data"])
    from .models import build_model
    model = build_model(config, load_backbone=True)
    if not getattr(model, "inference_only", False) or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Pretrained-only evaluation requires an entirely frozen inference-only model")
    device = torch.device("cuda:0")
    model = model.to(device).eval()
    smoke = smoke_first_image(model, dataset, device, runtime["amp"])
    metrics = evaluate_model(model, dataset, device, PROFILES["single"], runtime["amp"],
                             num_workers=runtime["num_workers"], criterion=None)
    if asset_metadata(config) != assets or file_identity(args.config) != identity:
        raise ValueError("Original pretrained weights/source/config changed during evaluation")
    result = pretrained_result(metrics, dataset, config, args.config, identity, assets, runtime, smoke)
    validate_pretrained_result(result, config, args.config, args.data_root, args.manifest, assets)
    write_new_json(args.output, result)
    print(json.dumps({key: result[key] for key in ("evaluated_samples", "mIoU", "mAcc", "scene_equal_macro")}))


if __name__ == "__main__":
    main()
