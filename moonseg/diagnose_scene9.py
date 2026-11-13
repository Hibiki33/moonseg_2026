"""Record paired scene-9 predictions for analysis of completed full-training runs.

Uses the existing single-scale BF16 predictor without adaptation. Ground truth is
used only for diagnostic counts; predictions are persisted before label analysis.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset

from .evaluate_public_test import (PublicTestResearchDataset, PROFILES, file_identity,
                                  inference_runtime, validate_final_checkpoint)
from .predict import predict_scores
from .utils import configure_torch_backends, load_config, load_partial_state_dict


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def confusion(prediction, target):
    return np.bincount((target.astype(np.int64) * 5 + prediction).ravel(), minlength=25).reshape(5, 5)


def metric(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    support, predicted = matrix.sum(1), matrix.sum(0)
    intersection = matrix.diagonal()
    valid = support > 0
    iou = intersection / np.maximum(support + predicted - intersection, 1)
    return {"mIoU": float(iou[valid].mean()) if valid.any() else None,
            "class_IoU": [float(v) if ok else None for v, ok in zip(iou, valid)],
            "confusion_matrix": matrix.astype(np.int64).tolist()}


@torch.inference_mode()
def scan(epoch, destination):
    from .models import build_model

    run = Path(f"runs/dnv3_va_m2f_vitl16_fulltrain_joint_e{epoch}").resolve()
    config = load_config(run / "resolved_config.json")
    checkpoint_path = run / "last.pt"
    identity = file_identity(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    validate_final_checkpoint(config, checkpoint_path, checkpoint, expected_epochs=epoch)
    runtime = inference_runtime(config, "bf16")
    configure_torch_backends(runtime)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Assign exactly one available GPU to each diagnostic process")
    device = torch.device("cuda:0")
    dataset = PublicTestResearchDataset("data/research_test/LuSNAR", "data/research_test/manifest.json", config["data"])
    indices = [i for i, row in enumerate(dataset.records) if row["scene"] == 9]
    assert len(indices) == 2087
    model = build_model(config, load_backbone=True).to(device)
    load_partial_state_dict(model, checkpoint["model"])
    del checkpoint
    model.eval()
    loader = DataLoader(Subset(dataset, indices), batch_size=1, num_workers=2, pin_memory=True)
    output = Path(destination) / f"e{epoch}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "predictions").mkdir()
    write_json(output / "metadata.json", {
        "checkpoint": str(checkpoint_path), "checkpoint_file_identity": identity,
        "checkpoint_epoch": epoch, "global_step": epoch * 125, "scene": 9,
        "expected_samples": len(indices), "profile": "single", "inference": PROFILES["single"],
        "runtime": runtime, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE"),
        "gpu": torch.cuda.get_device_name(), "config": config,
        "luminance_bins": ["Y < 25", "25 <= Y < 100", "Y >= 100"],
        "row_bands": [[0, 256], [256, 512], [512, 768], [768, 1024]],
        "label_usage": "Post-prediction diagnostics only; no parameter or prediction updates",
    })
    total = np.zeros((5, 5), np.int64)
    luminance_total = np.zeros((3, 5, 5), np.int64)
    row_total = np.zeros((4, 5, 5), np.int64)
    started = time.monotonic()
    with (output / "frames.jsonl").open("x") as stream:
        for count, batch in enumerate(loader, 1):
            image = batch["image"].to(device, non_blocking=True)
            scores = predict_scores(model, image, PROFILES["single"], runtime["amp"])
            prediction = scores.argmax(1)[0].to(dtype=torch.uint8).cpu().numpy()
            sample = batch["sample_id"][0]
            Image.fromarray(prediction).save(output / "predictions" / f"{sample}.png")
            target = batch["target"][0].numpy().astype(np.uint8)
            # Reconstruct the exact stored uint8 RGB from dataset normalization.
            pixels = np.rint((batch["image"][0].numpy().transpose(1, 2, 0) * dataset.std + dataset.mean) * 255)
            luminance = pixels @ np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
            bins = np.digitize(luminance, [25, 100])
            matrix = confusion(prediction, target)
            by_light = np.stack([confusion(prediction[bins == b], target[bins == b]) for b in range(3)])
            by_row = np.stack([confusion(prediction[a:a + 256], target[a:a + 256]) for a in range(0, 1024, 256)])
            total += matrix
            luminance_total += by_light
            row_total += by_row
            record = {"index": count - 1, "sample_id": sample, **metric(matrix),
                      "luminance_confusion": by_light.tolist(), "row_confusion": by_row.tolist()}
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            if count == 1 or count % 100 == 0 or count == len(indices):
                stream.flush()
                status = {"status": "running", "epoch": epoch, "evaluated_samples": count,
                          "expected_samples": len(indices), "elapsed_seconds": time.monotonic() - started}
                write_json(output / "status.json", status)
                print(json.dumps(status), flush=True)
            del scores, image
    if file_identity(checkpoint_path) != identity:
        raise RuntimeError("Checkpoint changed during diagnosis")
    lane = "07" if epoch in (10, 25) else "45"
    source = Path(f"runs/plans/2026-11-09/fulltrain_joint/lane_{lane}.state/scores/{run.name}/last/single/scores.json")
    reference = json.loads(source.read_text())["scenes"]["9"]
    assert np.array_equal(total.sum(1), np.asarray(reference["confusion_matrix"]).sum(1))
    result = {"status": "completed", "epoch": epoch, "evaluated_samples": len(indices), **metric(total),
              "elapsed_seconds": time.monotonic() - started, "luminance_confusion": luminance_total.tolist(),
              "row_confusion": row_total.tolist(), "reference_scores": str(source.resolve())}
    result["reference_mIoU_difference_pp"] = (result["mIoU"] - reference["mIoU"]) * 100
    result["reference_class_IoU_difference_pp"] = [
        (a - b) * 100 for a, b in zip(result["class_IoU"], reference["class_IoU"].values())]
    write_json(output / "summary.json", result)
    write_json(output / "status.json", {k: result[k] for k in ("status", "epoch", "evaluated_samples", "elapsed_seconds")})
    print(json.dumps(result), flush=True)
    del model, loader, dataset
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", nargs="+", type=int, choices=[10, 15, 20, 25], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(set(args.epochs)) != len(args.epochs):
        raise ValueError("Do not repeat epochs")
    for epoch in args.epochs:
        scan(epoch, args.output)


if __name__ == "__main__":
    main()
