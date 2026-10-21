#!/usr/bin/env python3
"""Validate complete predictions and create the official prediction ZIP layout.

ZIP entries are test/Moon_i/submission/<sample_id>.png without an extra wrapper.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moonseg.data.manifest import TEST_SCENES, load_manifest, read_allowlist, require_complete_allowlist
from moonseg.predict import prediction_path
from moonseg.utils import write_json

ALLOWED_PREFIXES = tuple(f"test/Moon_{scene}/submission/" for scene in sorted(TEST_SCENES))


def check_entry_names(names):
    """Every entry must be a PNG file directly under test/Moon_{2,4,9}/submission/ (no directory entries)."""
    for name in names:
        if name.endswith("/") or "\\" in name:
            raise ValueError(f"ZIP must contain file entries only, found directory-like entry: {name!r}")
        prefix = next((value for value in ALLOWED_PREFIXES if name.startswith(value)), None)
        rest = name[len(prefix):] if prefix else ""
        if prefix is None or not rest.endswith(".png") or "/" in rest or rest in ("", ".png"):
            raise ValueError(f"ZIP entry outside test/Moon_{{2,4,9}}/submission/<sample_id>.png: {name!r}")


def verify_archive(archive, expected):
    with zipfile.ZipFile(archive) as stream:
        infos = stream.infolist()
        directories = [info.filename for info in infos if info.is_dir()]
        if directories:
            raise ValueError(f"ZIP contains directory entries: {directories[:8]}")
        names = [info.filename for info in infos]
        check_entry_names(names)
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError("ZIP entries do not match the validated prediction set exactly")
        bad = stream.testzip()
        if bad is not None:
            raise ValueError(f"Corrupt ZIP member: {bad}")
    return len(names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Dataset root containing official test RGB inputs")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--official-test-allowlist", required=True, help="Complete fixed official test sample list")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--report", default="runs/submission_validation.json")
    parser.add_argument("--archive", help="Optional prediction ZIP with the official test/Moon_i/submission layout")
    args = parser.parse_args()
    records = load_manifest(args.manifest, args.root, TEST_SCENES, args.official_test_allowlist, need_label=False)
    require_complete_allowlist(records, read_allowlist(args.official_test_allowlist))
    root, predictions = Path(args.root), Path(args.predictions)
    expected = {prediction_path(record).as_posix() for record in records}
    actual = {path.relative_to(predictions).as_posix() for path in predictions.rglob("*.png")}
    if actual != expected:
        raise ValueError(f"Prediction set mismatch: missing={sorted(expected-actual)[:8]}, extra={sorted(actual-expected)[:8]}")
    for record in records:
        path = predictions / prediction_path(record)
        with Image.open(root / record["image"]) as image:
            size = image.size
        with Image.open(path) as mask:
            if mask.format != "PNG" or mask.mode != "L" or mask.size != size:
                raise ValueError(f"Require original-size 8-bit single-channel PNG: {path}; got {mask.format}/{mask.mode}/{mask.size}, expected {size}")
            pixels = np.asarray(mask)
            if pixels.dtype != np.uint8 or pixels.min() < 0 or pixels.max() > 4:
                raise ValueError(f"Prediction contains invalid IDs: {path}")
    report = {"validated_samples": len(records), "complete_official_test_list": True,
              "output_encoding": "8-bit single-channel PNG, IDs 0..4, original RGB dimensions",
              "layout": "test/Moon_i/submission/<sample_id>.png", "official_archive_layout_verified": True,
              "rules_source": "https://insight.onescience.cn/competition/moonseg?event_id=c2f557a58ff2e9f4265d56f112a7e01&tab=info&section=validation",
              "note": "Validates the prediction package. Model weights, source, configuration, dependencies, README, training logs and model/data provenance must accompany the solution as required. No submission was sent."}
    check_entry_names(sorted(expected))
    if args.archive:
        archive = Path(args.archive)
        archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as stream:
            for name in sorted(expected):
                stream.write(predictions / name, name)
        try:
            report["archive_file_entries"] = verify_archive(archive, expected)
        except Exception:
            archive.unlink(missing_ok=True)  # never leave an invalid package behind
            raise
        report["archive_entries_verified"] = "file entries only, all under test/Moon_{2,4,9}/submission/"
        report["prediction_archive"] = str(archive)
    write_json(report, args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
