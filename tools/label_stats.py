#!/usr/bin/env python3
"""Per-sample class pixel counts for a labelled training manifest (multiprocess).

Reads labels only through moonseg.data.manifest.load_manifest, so only
official_train_manifest files (with --official-allowlist) or, with the explicit
--development-pool opt-in, public_development_manifest files of training scenes
1,3,5,6,7,8 are accepted. Test-scene labels are never read.

Output (JSON):
  {"kind": "label_stats", "num_classes": 5, "manifest": "<path>",
   "samples": {"<sample_id>": {"class_pixels": [5 ints], "total": N}},
   "class_pixel_fraction": [5 floats], "class_presence_fraction": [5 floats], ...}
class_pixel_fraction pools pixels over all samples; class_presence_fraction is the
fraction of samples containing at least one pixel of the class.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from moonseg.data.labels import CLASS_NAMES, decode_label
from moonseg.data.manifest import TRAIN_SCENES, load_manifest, manifest_kind, manifest_provenance
from moonseg.utils import write_json

NUM_CLASSES = len(CLASS_NAMES)


def _count(task):
    root, record = task
    with Image.open(root / record["label"]) as handle:
        label = decode_label(handle)
    counts = np.bincount(label.ravel(), minlength=NUM_CLASSES)
    if counts.size != NUM_CLASSES:
        raise ValueError(f"Label has IDs outside 0..{NUM_CLASSES - 1}: {record['sample_id']}")
    return str(record["sample_id"]), [int(value) for value in counts], int(label.size)


def label_stats(manifest, root, official_allowlist=None, development_pool=False, workers=64):
    records = load_manifest(manifest, root, TRAIN_SCENES, official_allowlist, need_label=True,
                            allow_development_pool=development_pool)
    root = Path(root)
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as pool:
        results = list(pool.map(_count, [(root, record) for record in records], chunksize=16))
    samples = {sample_id: {"class_pixels": counts, "total": total} for sample_id, counts, total in results}
    pixels = np.array([value["class_pixels"] for value in samples.values()], dtype=np.int64)
    total = int(sum(value["total"] for value in samples.values()))
    if pixels.sum() != total:
        raise RuntimeError("Class pixel counts do not sum to the label area")
    return {"kind": "label_stats", "num_classes": NUM_CLASSES, "manifest": str(manifest),
            "manifest_kind": manifest_kind(manifest), "provenance": manifest_provenance(manifest),
            "class_names": list(CLASS_NAMES), "num_samples": len(samples),
            "scenes": sorted({int(record["scene"]) for record in records}),
            "total_pixels": total, "class_pixels_total": [int(value) for value in pixels.sum(0)],
            "class_pixel_fraction": [float(value) for value in pixels.sum(0) / max(total, 1)],
            "class_presence_fraction": [float(value) for value in (pixels > 0).mean(0)],
            "samples": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="Labelled training manifest (official or development pool)")
    parser.add_argument("--root", default="data/LuSNAR")
    parser.add_argument("--output", help="Default: <manifest dir>/label_stats_<manifest stem>.json")
    parser.add_argument("--official-allowlist", help="Organizer training list; required for official_train_manifest")
    parser.add_argument("--development-pool", action="store_true",
                        help="Explicit opt-in for public_development_manifest files (method development only)")
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()
    manifest = Path(args.manifest)
    output = Path(args.output) if args.output else manifest.parent / f"label_stats_{manifest.stem}.json"
    stats = label_stats(manifest, args.root, args.official_allowlist, args.development_pool, args.workers)
    write_json(stats, output)
    summary = {key: value for key, value in stats.items() if key != "samples"}
    summary["output"] = str(output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
