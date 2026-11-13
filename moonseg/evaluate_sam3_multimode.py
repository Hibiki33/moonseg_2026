"""One-pass frozen SAM3 public development evaluation: five semantic modes + two instance diagnostics."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import time

import torch
from torch.utils.data import DataLoader, Subset

from .data.labels import CLASS_NAMES
from .evaluate_pretrained import POLICY, PREPROCESSING, asset_metadata, validate_config
from .evaluate_public_test import (EXPECTED_SAMPLES, MANIFEST_KIND, PROFILES, SCENE_COUNTS,
                                   PublicTestResearchDataset, file_identity, write_new_json)
from .metrics import SegmentationMetrics
from .models.sam3 import PROMPTS, build_sam3
from .models.sam3_semantic import (INSTANCE_SPECS, MODE_SPECS, SEG_EARTH_COMMIT,
                                   class_prediction, instance_prediction,
                                   mask_probability_chunks, prompt_semantic_maps)
from .utils import amp_context, configure_torch_backends, load_config

ROOT = Path(__file__).resolve().parents[1]
GPU_UUID = "GPU-aec3a52c-8629-106e-3b66-a374628971a8"
AUDIT_SAMPLES = ROOT / "docs/evidence/sam3_protocol_audit_20261110.json"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def gpu_snapshot():
    row = subprocess.check_output([
        "nvidia-smi", "--id=2", "--query-gpu=index,uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True).strip().split(",")
    if len(row) != 4 or row[0].strip() != "2" or row[1].strip() != GPU_UUID:
        raise RuntimeError(f"Physical GPU2 identity mismatch: {row}")
    apps = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"], text=True)
    processes = [line.strip() for line in apps.splitlines() if line.split(",")[0].strip() == GPU_UUID]
    return {"physical_index": 2, "uuid": GPU_UUID, "memory_used_MiB": int(row[2]),
            "utilization_percent": int(row[3]), "compute_processes": processes, "at": utc_now()}


@contextmanager
def exclusive_gpu2():
    """Own both shared lock namespaces before checking idleness or initializing CUDA."""
    if torch.cuda.is_initialized():
        raise RuntimeError("GPU2 must be reserved before CUDA initialization")
    with ExitStack() as stack:
        lock_paths = [ROOT / "runs/.training_gpu_2.lock", ROOT / "runs/evaluations_2026-10-23/gpu_2.lock"]
        for path in lock_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = stack.enter_context(path.open("a"))
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"GPU2 shared lock is owned by another task: {path}") from exc
        snapshot = gpu_snapshot()
        if (snapshot["compute_processes"] or snapshot["utilization_percent"] != 0
                or snapshot["memory_used_MiB"] > 256):
            raise RuntimeError(f"GPU2 is not idle; refusing to start: {snapshot}")
        os.environ["CUDA_VISIBLE_DEVICES"] = GPU_UUID
        os.environ["NCCL_P2P_DISABLE"] = "1"
        snapshot["lock_paths"] = list(map(str, lock_paths))
        snapshot["locks_held_by_pid"] = os.getpid()
        yield snapshot
        # Closing both handles releases the locks even when evaluation raises.


class RejectionMetrics:
    """5x6 confusion plus independent binary mask coverage; reject is an FN."""
    def __init__(self, device="cpu"):
        self.confusion = torch.zeros((5, 6), dtype=torch.int64, device=device)
        self.binary = torch.zeros((5, 3), dtype=torch.int64, device=device)  # TP, FP, FN
        self.overlap = 0
        self.empty_images = 0
        self.empty_class_counts = [0] * 5
        self.images = 0

    @torch.inference_mode()
    def update(self, prediction, support, target):
        valid = (target >= 0) & (target < 5)
        if ((prediction[valid] < 0) | (prediction[valid] > 5)).any():
            raise ValueError("Instance prediction must be class 0..4 or reject=5")
        self.confusion += torch.bincount(target[valid] * 6 + prediction[valid], minlength=30).reshape(5, 6)
        for c in range(5):
            pred = support[c] & valid
            gt = (target == c) & valid
            self.binary[c] += torch.stack(((pred & gt).sum(), (pred & ~gt).sum(), (~pred & gt).sum()))
            self.empty_class_counts[c] += int(not pred.any())
        self.overlap += int(((support.sum(0) > 1) & valid).sum())
        self.empty_images += int(not support[:, valid].any())
        self.images += 1

    def merge(self, other):
        self.confusion += other.confusion
        self.binary += other.binary
        self.overlap += other.overlap
        self.empty_images += other.empty_images
        self.empty_class_counts = [a + b for a, b in zip(self.empty_class_counts, other.empty_class_counts)]
        self.images += other.images

    def compute(self):
        matrix = self.confusion.cpu().double()
        gt = matrix.sum(1)
        tp = matrix[:, :5].diag()
        pred = matrix[:, :5].sum(0)
        valid = gt > 0
        iou = tp / (gt + pred - tp).clamp_min(1)
        recall = tp / gt.clamp_min(1)
        total = int(matrix.sum())
        binary = self.binary.cpu().tolist()
        by_class = {}
        for c, name in enumerate(CLASS_NAMES):
            btp, bfp, bfn = binary[c]
            by_class[name] = {"tp": btp, "fp": bfp, "fn": bfn,
                              "IoU": btp / (btp + bfp + bfn) if btp + bfp + bfn else None,
                              "precision": btp / (btp + bfp) if btp + bfp else None,
                              "recall": btp / (btp + bfn) if btp + bfn else None}
        return {"confusion_matrix_5x6": self.confusion.cpu().tolist(),
                "prediction_columns": [*CLASS_NAMES, "reject"], "reject_class_id": 5,
                "diagnostic_mean_IoU_with_reject": float(iou[valid].mean()) if valid.any() else None,
                "class_IoU_with_reject": {n: float(iou[c]) if valid[c] else None for c, n in enumerate(CLASS_NAMES)},
                "class_recall_with_reject": {n: float(recall[c]) if valid[c] else None for c, n in enumerate(CLASS_NAMES)},
                "gt_class_covered_pixels": {n: int(gt[c] - matrix[c, 5]) for c, n in enumerate(CLASS_NAMES)},
                "gt_class_uncovered_pixels": {n: int(matrix[c, 5]) for c, n in enumerate(CLASS_NAMES)},
                "coverage_fraction": (total - int(matrix[:, 5].sum())) / total if total else None,
                "overlap_fraction": self.overlap / total if total else None,
                "independent_binary_masks": by_class, "all_reject_images": self.empty_images,
                "empty_class_images": dict(zip(CLASS_NAMES, self.empty_class_counts)),
                "evaluated_samples": self.images,
                "metric_scope": "Diagnostic only: unobserved pixels are reject/FN, not regolith; independent binary masks may overlap. Not a complete five-class competition score."}


@torch.inference_mode()
def crater_localization(outputs, target, chunk_size):
    """GT is used only after grounding, to measure localization upper bounds."""
    gt = target == 2
    valid = (target >= 0) & (target < 5)
    count = int(gt.sum())
    r = outputs["pred_logits"].sigmoid().reshape(-1)
    p = outputs["presence_logit_dec"].sigmoid().reshape(())
    q = r * p
    unions = {name: torch.zeros_like(gt) for name in ("all_queries", "r_t050", "q_t050", "r_t030", "q_t030")}
    best_iou, best_index = -1.0, None
    for ids, probabilities in mask_probability_chunks(outputs["pred_masks"][0], target.shape, chunk_size):
        masks = (probabilities > 0.5) & valid
        unions["all_queries"] |= masks.any(0)
        for name, score, threshold in (("r_t050", r, 0.5), ("q_t050", q, 0.5),
                                       ("r_t030", r, 0.3), ("q_t030", q, 0.3)):
            selected = score[ids] > threshold
            if selected.any():
                unions[name] |= masks[selected].any(0)
        if count:
            inter = (masks & gt).sum((1, 2))
            ious = inter.double() / (masks.sum((1, 2)) + count - inter).clamp_min(1)
            value, index = ious.max(0)
            if float(value) > best_iou:
                best_iou, best_index = float(value), int(ids[index])
    return {"gt_crater_pixels": count,
            "best_raw_query_IoU_against_crater_semantic_union": best_iou if count else None,
            "best_raw_query_index": best_index,
            "localization": {name: {"union_pixels": int(mask.sum()), "covered_crater_pixels": int((mask & gt).sum()),
                                    "crater_coverage": int((mask & gt).sum()) / count if count else None}
                             for name, mask in unions.items()},
            "scope": "Post-hoc GT oracle localization diagnostic; never used for prediction, points or class admission."}


def select_indices(dataset, subset):
    if subset is None:
        return list(range(len(dataset))), None
    audit = json.loads(AUDIT_SAMPLES.read_text())
    selected = {(int(record["scene"]), str(record["sample_id"])) for record in audit["samples"]}
    if len(selected) != 18:
        raise ValueError("The fixed audit18 source must contain exactly 18 unique samples")
    indices = [i for i, row in enumerate(dataset.records) if (row["scene"], str(row["sample_id"])) in selected]
    if len(indices) != 18:
        raise ValueError("Fixed audit sample IDs are missing from the complete public manifest")
    return indices, {"name": "audit18", "source": str(AUDIT_SAMPLES),
                     "source_file_identity": file_identity(AUDIT_SAMPLES),
                     "selection": audit["sample_selection"],
                     "label_fields_used_for_selection": False}


def summarize_modes(scene_metrics, counts, ties):
    result = {}
    for mode, metrics in scene_metrics.items():
        pooled = SegmentationMetrics()
        scenes = {}
        for scene, metric in metrics.items():
            pooled.confusion += metric.confusion.cpu()
            scenes[scene] = {**metric.compute(), "evaluated_samples": counts[scene], "tie_pixels": ties[mode][scene]}
        result[mode] = {"status": "complete", **MODE_SPECS[mode], "pooled": pooled.compute(), "scenes": scenes,
                        "scene_equal_macro": {key: sum(row[key] for row in scenes.values()) / len(scenes)
                                              for key in ("mIoU", "mAcc")},
                        "tie_policy": "Lowest fixed class ID among equal finite maxima; all-zero/nonfinite is an error."}
    return result


@torch.inference_mode()
def evaluate_multimode(model, dataset, indices, runtime, chunk_size=8):
    device = torch.device("cuda:0")
    counts = Counter(str(dataset.records[i]["scene"]) for i in indices)
    scene_metrics = {name: {scene: SegmentationMetrics(device=device) for scene in counts} for name in MODE_SPECS}
    instance_metrics = {name: {scene: RejectionMetrics(device=device) for scene in counts} for name in INSTANCE_SPECS}
    ties = {name: {scene: 0 for scene in counts} for name in MODE_SPECS}
    rows, actual_counts, pixel_counts = [], Counter(), Counter()
    calls = 0
    loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False,
                        num_workers=runtime["num_workers"], pin_memory=True)
    started = time.monotonic()
    for batch in loader:
        scene, sample_id = str(int(batch["scene"][0])), batch["sample_id"][0]
        images = batch["image"].to(device, non_blocking=True)
        # Labels never reach model.iter_grounding or any postprocessor input.
        target = batch["target"][0].to(device, non_blocking=True)
        valid = (target >= 0) & (target < 5)
        size = tuple(images.shape[-2:])
        maps = {name: [] for name in MODE_SPECS}
        inst_maps = {name: [] for name in INSTANCE_SPECS}
        supports = {name: [] for name in INSTANCE_SPECS}
        row = {"scene": int(scene), "sample_id": sample_id, "prompts": []}
        seen = []
        with amp_context(device, runtime["amp"]):
            for image_index, class_id, outputs in model.iter_grounding(images):
                if image_index != 0 or class_id != len(seen):
                    raise ValueError("Shared grounding must yield exactly the five fixed class IDs in order")
                seen.append(class_id)
                calls += 1
                reduced = prompt_semantic_maps(outputs, size, chunk_size)
                for name in maps:
                    maps[name].append(reduced["scores"][name])
                for name in supports:
                    inst_maps[name].append(reduced["instances"][name])
                    supports[name].append(reduced["support"][name])
                row["prompts"].append({"class_id": class_id, "prompt": PROMPTS[class_id], **reduced["metadata"]})
                if class_id == 2:
                    row["crater_raw_localization"] = crater_localization(outputs, target, chunk_size)
                del outputs, reduced
        if seen != list(range(5)):
            raise ValueError("Incomplete fixed-vocabulary grounding")
        gt_crater = target == 2
        crater_count = int(gt_crater.sum())
        row["crater_competition"] = {}
        for name in MODE_SPECS:
            prediction, tie_count = class_prediction(torch.stack(maps[name]))
            scene_metrics[name][scene].update(prediction, target)
            ties[name][scene] += tie_count
            by_pred = torch.bincount(prediction[gt_crater], minlength=5).cpu().tolist()
            row["crater_competition"][name] = {
                "gt_crater_pixels_by_predicted_class": dict(zip(CLASS_NAMES, by_pred)),
                "fraction_swallowed_by_other_classes": 1 - by_pred[2] / crater_count if crater_count else None}
        row["instance_coverage"] = {}
        for name in INSTANCE_SPECS:
            support = torch.stack(supports[name])
            prediction = instance_prediction(torch.stack(inst_maps[name]), support)
            instance_metrics[name][scene].update(prediction, support, target)
            covered = support.any(0)
            row["instance_coverage"][name] = {
                "covered_pixels": int((covered & valid).sum()),
                "reject_pixels": int((~covered & valid).sum()),
                "gt_class_covered_pixels": {n: int((covered & (target == c)).sum()) for c, n in enumerate(CLASS_NAMES)},
                "gt_class_uncovered_pixels": {n: int((~covered & (target == c)).sum()) for c, n in enumerate(CLASS_NAMES)}}
        rows.append(row)
        actual_counts[scene] += 1
        pixel_counts[scene] += int(valid.sum())
        if len(rows) == 1 or len(rows) % 100 == 0 or len(rows) == len(indices):
            print(json.dumps({"evaluated_samples": len(rows), "expected_samples": len(indices),
                              "grounding_calls": calls, "elapsed_seconds": round(time.monotonic() - started, 2)}), flush=True)
        del maps, inst_maps, supports, images, target, batch
    if actual_counts != counts or calls != len(indices) * 5:
        raise ValueError("Incomplete or repeated shared-forward evaluation")
    instances = {}
    for name, metrics in instance_metrics.items():
        pooled = RejectionMetrics(device=device)
        for metric in metrics.values():
            pooled.merge(metric)
        instances[name] = {"instance_threshold": INSTANCE_SPECS[name], "mask_support_threshold": 0.5,
                           "pooled": pooled.compute(), "scenes": {s: m.compute() for s, m in metrics.items()}}
    return {"evaluated_samples": len(rows), "scene_sample_counts": dict(actual_counts),
            "scene_valid_pixel_counts": dict(pixel_counts), "grounding_calls": calls,
            "modes": summarize_modes(scene_metrics, actual_counts, ties), "instance_diagnostics": instances,
            "per_image_diagnostics": rows, "elapsed_seconds": time.monotonic() - started}


def validate_result(result, expected_counts):
    """Recompute scores from matrices and require full/subset counts to remain explicit."""
    if result["scene_sample_counts"] != expected_counts or result["evaluated_samples"] != sum(expected_counts.values()):
        raise ValueError("Result count mismatch")
    if result["evaluation_scope"] == "full_public_development" and expected_counts != SCENE_COUNTS:
        raise ValueError("A full result must contain all 5004 public images")
    if set(result["modes"]) != set(MODE_SPECS) or set(result["instance_diagnostics"]) != set(INSTANCE_SPECS):
        raise ValueError("Missing or extra SAM3 modes")
    if result["grounding_calls"] != result["evaluated_samples"] * 5:
        raise ValueError("Expected exactly one grounding per image and original prompt")
    if len(result["per_image_diagnostics"]) != result["evaluated_samples"]:
        raise ValueError("Missing image diagnostics")
    for name, record in result["modes"].items():
        pooled = torch.zeros((5, 5), dtype=torch.int64)
        for scene, count in expected_counts.items():
            metrics = record["scenes"][scene]
            matrix = torch.tensor(metrics["confusion_matrix"], dtype=torch.int64)
            if matrix.shape != (5, 5) or (matrix < 0).any() or int(matrix.sum()) != result["scene_valid_pixel_counts"][scene]:
                raise ValueError(f"Invalid semantic confusion counts: {name}/{scene}")
            check = SegmentationMetrics()
            check.confusion = matrix
            recomputed = check.compute()
            for key in ("mIoU", "mAcc", "class_IoU", "class_recall"):
                actual, wanted = metrics[key], recomputed[key]
                pairs = zip(actual.values(), wanted.values()) if isinstance(wanted, dict) else [(actual, wanted)]
                if any((a is not b if a is None or b is None else not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12))
                       for a, b in pairs):
                    raise ValueError(f"Inconsistent semantic metrics: {name}/{scene}/{key}")
            if metrics["evaluated_samples"] != count:
                raise ValueError("Scene count mismatch")
            pooled += matrix
        if pooled.tolist() != record["pooled"]["confusion_matrix"]:
            raise ValueError("Pooled confusion mismatch")
    for name, record in result["instance_diagnostics"].items():
        pooled = torch.zeros((5, 6), dtype=torch.int64)
        for scene, count in expected_counts.items():
            metrics = record["scenes"][scene]
            matrix = torch.tensor(metrics["confusion_matrix_5x6"], dtype=torch.int64)
            if (matrix.shape != (5, 6) or (matrix < 0).any() or metrics["evaluated_samples"] != count
                    or int(matrix.sum()) != result["scene_valid_pixel_counts"][scene]):
                raise ValueError(f"Invalid reject confusion counts: {name}/{scene}")
            pooled += matrix
        if pooled.tolist() != record["pooled"]["confusion_matrix_5x6"]:
            raise ValueError("Pooled reject confusion mismatch")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sam3_pretrained.yaml")
    parser.add_argument("--data-root", default="data/research_test/LuSNAR")
    parser.add_argument("--manifest", default="data/research_test/manifest.json")
    parser.add_argument("--profile", choices=["single"], default="single")
    parser.add_argument("--subset", choices=["audit18"], default=None,
                        help="Use the pre-existing fixed audit IDs; output is explicitly a subset diagnostic.")
    parser.add_argument("--mask-chunk-size", type=int, default=8)
    parser.add_argument("--output", required=True, help="New combined scores JSON; existing scores are never overwritten.")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if not output.is_relative_to((ROOT / "runs").resolve()) or output.exists():
        raise ValueError("Output must be a new file under this repository's local runs directory")
    if args.mask_chunk_size < 1 or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Require positive mask chunk size and one process/GPU")
    identity = file_identity(args.config)
    config = load_config(args.config)
    runtime = validate_config(config)
    assets = asset_metadata(config)
    manifest_identity = file_identity(args.manifest)
    dataset = PublicTestResearchDataset(args.data_root, args.manifest, config["data"])
    indices, subset_metadata = select_indices(dataset, args.subset)
    expected = dict(Counter(str(dataset.records[i]["scene"]) for i in indices))
    started_at = utc_now()
    try:
        with exclusive_gpu2() as receipt:
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError("One CUDA-visible physical GPU2 is required")
            configure_torch_backends(runtime)
            model = build_sam3(config).to("cuda:0").eval()
            if any(p.requires_grad for p in model.parameters()):
                raise ValueError("SAM3 must remain fully frozen")
            metrics = evaluate_multimode(model, dataset, indices, runtime, args.mask_chunk_size)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            if (file_identity(args.config) != identity or asset_metadata(config) != assets
                    or file_identity(args.manifest) != manifest_identity):
                raise ValueError("SAM3 assets/config/manifest changed during evaluation")
            code_paths = [ROOT / "moonseg/models/sam3.py", ROOT / "moonseg/models/sam3_semantic.py", Path(__file__)]
            result = {**metrics, **POLICY, **assets, "status": "complete", "schema_version": 1,
                      "started_at": started_at, "completed_at": utc_now(),
                      "evaluation_scope": "subset_diagnostic" if args.subset else "full_public_development",
                      "evaluation_kind": MANIFEST_KIND, "official_score": False,
                      "subset": subset_metadata, "config_path": str(Path(args.config).resolve()),
                      "config_file_identity": identity, "resolved_config": config,
                      "manifest": str(dataset.manifest_path), "manifest_file_identity": manifest_identity,
                      "data_root": str(dataset.root), "profile": args.profile, "inference": PROFILES["single"],
                      "class_names": list(CLASS_NAMES), "prompts": list(PROMPTS),
                      "preprocessing": {**PREPROCESSING, "score_postprocess": "Five shared-forward modes; see postprocessing and each mode formula."},
                      "pretrained_mirror": {"repository": "ModelScope facebook/sam3",
                                            "revision": "f8ee5d16d5c30d0e87b1d22cbd358ce97761df56"},
                      "runtime": {**runtime, "mask_chunk_size": args.mask_chunk_size,
                                  "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                                  "nccl_p2p_disable": os.environ["NCCL_P2P_DISABLE"],
                                  "max_memory_allocated_bytes": peak},
                      "gpu_reservation": receipt,
                      "implementation_file_identities": {str(p.resolve()): file_identity(p) for p in code_paths},
                      "postprocessing": {"segearth_source_commit": SEG_EARTH_COMMIT,
                          "segearth_source": "https://github.com/earth-insights/SegEarth-OV-3",
                          "resize": "Bilinear raw logits to original H/W before sigmoid, align_corners=False.",
                          "dtype": "Legacy dense explicitly FP32 before resize, unchanged. Other modes retain native output dtype through sigmoid/resize and scalar score weighting; FP32 per-class accumulator, matching pinned author order. Actual dtypes are recorded per prompt.",
                          "semantic_full_results": 5, "instance_diagnostic_results": 2,
                          "instance_support": "q>t and resized mask probability>0.5; uncovered pixels reject=5.",
                          "single_presence_ablation_threshold": 0.5},
                      "label_usage": "After each grounding only: confusion, coverage and oracle localization diagnostics. Never model input, prompt/point/box generation or class-presence admission.",
                      "protocol": "Pinned SegEarth fusion algorithm on existing Meta SAM3 runtime; Moon five-class/reject diagnostics; not LuSNAR or Meta F.1 environment reproduction."}
            validate_result(result, expected)
            write_new_json(output, result)
            print(json.dumps({"output": str(output), "scope": result["evaluation_scope"],
                              "evaluated_samples": result["evaluated_samples"],
                              "scene_equal_macro": {n: r["scene_equal_macro"] for n, r in result["modes"].items()}}), flush=True)
    except Exception as exc:
        failure = output.with_suffix(".failure.json")
        if not failure.exists():
            write_new_json(failure, {"status": "failed", "started_at": started_at, "failed_at": utc_now(),
                                     "exception_type": type(exc).__name__, "exception": str(exc),
                                     "evaluation_scope": "subset_diagnostic" if args.subset else "full_public_development",
                                     "expected_scene_counts": expected})
        raise


if __name__ == "__main__":
    main()
