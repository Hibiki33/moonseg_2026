"""Write official-layout prediction PNGs for one release configuration."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from .checkpoints import WEIGHTS_KEYS, load_checkpoint, load_partial_state_dict, select_weights
from .data.manifest import TEST_SCENES, load_manifest, read_allowlist, require_complete_allowlist, scene_number
from .inference import PROFILES, inference_padding, predict_scores, score_postprocess
from .utils import configure_torch_backends, init_distributed, load_config, write_json


def normalize_image(image, data):
    pixels = np.asarray(image.convert("RGB"), np.float32) / 255.0
    mean = np.asarray(data.get("mean", [0.485, 0.456, 0.406]), np.float32)
    std = np.asarray(data.get("std", [0.229, 0.224, 0.225]), np.float32)
    return torch.from_numpy(((pixels - mean) / std).transpose(2, 0, 1).copy()).unsqueeze(0)


def prediction_path(record):
    sample_id = str(record["sample_id"])
    return Path("test") / f"Moon_{scene_number(record['scene'])}" / "submission" / f"{sample_id}.png"


def _json_scalar(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return value.item() if hasattr(value, "item") else str(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile", choices=PROFILES, help="Override config inference with the evaluation profile")
    parser.add_argument("--data-root", help="Override data.root for the official test manifest")
    parser.add_argument("--checkpoint", help="Trained last.pt. Omit only for the inference-only SAM3 config.")
    parser.add_argument("--manifest", required=True, help="Complete unlabeled official test manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--official-test-allowlist", required=True)
    parser.add_argument("--weights-key", choices=WEIGHTS_KEYS, default="auto",
                        help="auto = model; ema_model only for legacy checkpoints from runs with the removed train.ema")
    parser.add_argument("--device", help="Explicit device for single-process inference; incompatible with multiple torchrun ranks")
    args = parser.parse_args(argv)
    if args.device is not None and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        parser.error("--device cannot be used with multiple torchrun ranks; each rank uses its LOCAL_RANK device")
    config = load_config(args.config)
    if args.device is None:
        device, rank, world_size = init_distributed()
    else:
        device, rank, world_size = torch.device(args.device), 0, 1
        if device.type == "cuda":
            torch.cuda.set_device(device)
    data_root = Path(args.data_root or config["data"]["root"]).resolve()
    if args.profile:
        configure_torch_backends(config.get("train", config.get("runtime", {})))
    records = load_manifest(args.manifest, data_root, TEST_SCENES,
                            args.official_test_allowlist, need_label=False)
    require_complete_allowlist(records, read_allowlist(args.official_test_allowlist))
    from .models import build_model
    num_classes = int(config.get("model", {}).get("num_classes", 5))
    checkpoint = load_checkpoint(args.checkpoint) if args.checkpoint else {}
    full_state = checkpoint.get("full_model_state") is True
    model = build_model(config, load_backbone=not full_state).to(device)
    if getattr(model, "inference_only", False):
        if args.checkpoint is not None or args.weights_key != "auto":
            raise ValueError("This model loads its official weights from the config and does not take a training checkpoint")
        checkpoint, weights_key = {}, "auto"
        if rank == 0:
            print("weights: official config (inference-only)", flush=True)
    else:
        if args.checkpoint is None:
            raise ValueError("A trained checkpoint is required")
        state, weights_key, message = select_weights(checkpoint, args.weights_key)
        if rank == 0:
            print(f"{message}; checkpoint={args.checkpoint}", flush=True)
        if full_state:
            model.load_state_dict(state, strict=True)
        else:
            load_partial_state_dict(model, state, weights_key)
        del state
        checkpoint = {key: checkpoint.get(key) for key in ("epoch", "selected_by", "best_source")}
    model.eval()
    inference = PROFILES[args.profile] if args.profile else config.get("inference", {})
    amp = config.get("train", config.get("runtime", {})).get("amp", "none")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    predictions = []
    # Strided assignment has no repeated padding samples and gives each PNG one writer.
    for index in range(rank, len(records), world_size):
        record = records[index]
        with Image.open(data_root / record["image"]) as handle:
            image = handle.convert("RGB")
        tensor = normalize_image(image, config["data"]).to(device)
        scores = predict_scores(model, tensor, inference, amp, num_classes)
        mask = scores.argmax(1)[0].cpu().numpy().astype(np.uint8)
        relative = prediction_path(record)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(target)
        predictions.append((index, {"scene": scene_number(record["scene"]), "sample_id": record["sample_id"],
                                   "prediction": relative.as_posix(), "width": image.width, "height": image.height}))
        print(f"[rank {rank}; sample {index + 1}/{len(records)}] {target}", flush=True)
    if world_size > 1:
        gathered = [None] * world_size if rank == 0 else None
        dist.gather_object(predictions, gathered, dst=0)
        if rank == 0:
            predictions = [item for shard in gathered for item in shard]
    if rank == 0:
        predictions.sort(key=lambda item: item[0])
        if [index for index, _ in predictions] != list(range(len(records))):
            raise RuntimeError("Distributed prediction index has missing or duplicated samples")
        settings = {"sliding_window": bool(inference.get("sliding_window", True)) and inference.get("crop_size") is not None,
                    "crop_size": inference.get("crop_size"), "stride": inference.get("stride"),
                    "scales": [float(scale) for scale in inference.get("scales", [1.0])],
                    "flip": bool(inference.get("flip", False)), "amp": amp, "num_classes": num_classes,
                    **inference_padding(model), "score_postprocess": score_postprocess(config),
                    "world_size": world_size}
        write_json({"kind": "prediction_index", "layout": "test/Moon_i/submission/<sample_id>.png",
                    "packaging_status": "Official prediction layout; validate complete files before submitting.",
                    "checkpoint": None if args.checkpoint is None else str(args.checkpoint), "weights_key": weights_key,
                    "weights_key_requested": args.weights_key, "selected_by": _json_scalar(checkpoint.get("selected_by")),
                    "best_source": _json_scalar(checkpoint.get("best_source")),
                    "checkpoint_epoch": _json_scalar(checkpoint.get("epoch")), "inference": settings,
                    "predictions": [record for _, record in predictions]}, output / "prediction_index.json")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
