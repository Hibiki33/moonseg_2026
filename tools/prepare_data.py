#!/usr/bin/env python3
"""Discover left RGB frames, then admit only samples on explicit official lists.

Without --official-allowlist this produces exploration manifests only. They cannot
be used by the training/evaluation loaders. Never derive an "official" allowlist
from the public archives: obtain the fixed sample list from the organizers.

--development-pool is a separate, explicit opt-in. It writes
public_development_manifest files from the public LuSNAR training-scene pool
(scenes 1,3,5,6,7,8 only) for method development before the organizer's list
arrives. Loaders accept them only with data.development_pool: true. Test scenes
2/4/9 never receive labels, and the final submission model must be retrained on
the organizer's list. See docs/data.md.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from moonseg.data.labels import decode_label
from moonseg.data.manifest import DEVELOPMENT_KIND, TEST_SCENES, TRAIN_SCENES, is_allowed, read_allowlist
from moonseg.utils import write_json

OFFICIAL_SPLITS = (("train", {1, 3, 6, 7, 8}), ("val", {5}), ("final_train", set(TRAIN_SCENES)))
DEVELOPMENT_SPLITS = (("dev_train", {1, 3, 6, 7, 8}), ("dev_val", {5}),
                      ("dev_val6_train", {1, 3, 5, 7, 8}), ("dev_val6", {6}),
                      ("dev_final_train", set(TRAIN_SCENES)))
DEVELOPMENT_NOTE = ("Public LuSNAR training-scene pool (scenes 1,3,5,6,7,8), for method development only. "
                    "Not an organizer sample list: membership is by public scene, not by the official allowlist. "
                    "Loaders accept it only with data.development_pool: true. Test scenes 2/4/9 are excluded and their "
                    "labels are never read. The final submission model must be retrained on the organizer's official list.")


def find_directory(root, template, scene):
    relative = template.format(scene=scene)
    candidates = [base / relative for base in (root, root / "train", root / "test")]
    present = [path for path in candidates if path.is_dir()]
    if len(present) > 1:
        raise ValueError(f"Ambiguous scene directory: {present}")
    return present[0] if present else None


def discover(root, scenes, image_template, label_template, with_labels):
    records = []
    seen = set()
    for scene in sorted(scenes):
        images = find_directory(root, image_template, scene)
        if images is None:
            continue
        labels = find_directory(root, label_template, scene) if with_labels else None
        for image in sorted(images.glob("*.png")):
            sample_id = image.stem
            if sample_id in seen:
                raise ValueError(f"sample_id must be globally unique; duplicate: {sample_id}")
            seen.add(sample_id)
            record = {"scene": scene, "sample_id": sample_id, "image": str(image.relative_to(root))}
            if with_labels:
                label = labels / image.name if labels is not None else None
                if label is None or not label.is_file():
                    raise FileNotFoundError(f"Matching left-camera training label missing: {scene}/{image.name}")
                record["label"] = str(label.relative_to(root))
            records.append(record)
    return records


def admitted(records, allowlist_path, scenes):
    allowlist = read_allowlist(allowlist_path)
    result = [record for record in records if is_allowed(record, allowlist)]
    found = {(row["scene"], row["sample_id"]) for row in result}
    found_ids = {row["sample_id"] for row in result}
    missing = [(scene, sample_id) for scene, sample_id in allowlist
               if (scene is None and sample_id not in found_ids) or
               (scene is not None and (scene not in scenes or (scene, sample_id) not in found))]
    if missing:
        raise ValueError(f"Official list has {len(missing)} missing/incompatible samples; first entries: {missing[:8]}")
    if not result:
        raise ValueError("No official samples found")
    return result


def write_manifest(path, records, kind, allowlist=None, source=None, note=None, extra=None):
    if note is None:
        note = ("Explicit official sample membership required; scene membership alone grants no training permission."
                if "exploration" in kind else "Filtered against the supplied explicit official sample list.")
    value = {"version": 1, "kind": kind, "official_allowlist": str(allowlist) if allowlist else None,
             "official_list_source": source, "note": note}
    value.update(extra or {})
    value["samples"] = records
    write_json(value, path)


def _validate_one(task):
    """Worker: decode one training label and compare it with its RGB size. Returns an error or None."""
    root, record = task
    try:
        with Image.open(root / record["label"]) as label:
            decoded = decode_label(label)
        with Image.open(root / record["image"]) as image:
            size = (image.height, image.width)
        if decoded.shape != size:
            return f"{record['scene']}/{record['sample_id']}: image/label size mismatch {size} vs {decoded.shape}"
    except Exception as error:  # report every failure, do not stop at the first
        return f"{record['scene']}/{record['sample_id']}: {type(error).__name__}: {error}"
    return None


def validate_labels(root, records, workers):
    """Validate label colors and dimensions with a process pool; only training-scene records are accepted."""
    if any(int(record["scene"]) not in TRAIN_SCENES or "label" not in record for record in records):
        raise ValueError("Label validation is restricted to labelled training scenes 1,3,5,6,7,8")
    tasks = [(root, record) for record in records]
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as pool:
        errors = [error for error in pool.map(_validate_one, tasks, chunksize=16) if error]
    if errors:
        raise ValueError(f"{len(errors)} label(s) failed validation; first entries: {errors[:8]}")
    return len(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="data/LuSNAR")
    parser.add_argument("--output-dir", default="data/manifests")
    parser.add_argument("--image-template", default="Moon_{scene}/color")
    parser.add_argument("--label-template", default="Moon_{scene}/label")
    parser.add_argument("--official-allowlist", help="Organizer's training sample list: JSON/JSONL/CSV or IDs.txt")
    parser.add_argument("--official-test-allowlist", help="Organizer's complete test sample list")
    parser.add_argument("--official-source", help="Organizer list URL or supplied file provenance, stored in manifests")
    parser.add_argument("--development-pool", action="store_true",
                        help="Also write public_development_manifest files (dev_*.json) from public training scenes "
                             "1,3,5,6,7,8 for method development only; see docs/data.md")
    parser.add_argument("--validate-labels", action="store_true",
                        help="Validate colors/dimensions of admitted official training samples, or with "
                             "--development-pool of all discovered training-scene labels (never test scenes)")
    parser.add_argument("--workers", type=int, default=64, help="Processes for --validate-labels")
    args = parser.parse_args()
    root, output = Path(args.root).resolve(), Path(args.output_dir)
    if args.validate_labels and not (args.official_allowlist or args.development_pool):
        parser.error("--validate-labels requires --official-allowlist or the explicit --development-pool opt-in; "
                     "unassigned public labels must not inform development otherwise")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if not root.is_dir():
        parser.error(f"Dataset root missing: {root}")
    train = discover(root, TRAIN_SCENES, args.image_template, args.label_template, True)
    test = discover(root, TEST_SCENES, args.image_template, args.label_template, False)
    if not train and not test:
        parser.error("No PNG images found; check --root and --image-template")
    all_ids = [row["sample_id"] for row in train + test]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Duplicate sample_id across training and test scenes")
    if any("label" in row for row in test):
        raise RuntimeError("Test-scene records must never carry labels")
    write_manifest(output / "exploration_train.json", train, "public_exploration_manifest")
    write_manifest(output / "exploration_test.json", test, "public_exploration_manifest")
    summary = {"public_train_pool_discovered": len(train), "public_test_images_discovered": len(test),
               "test_labels_opened": 0, "training_admitted": False}
    official = None
    if args.official_allowlist:
        official = admitted(train, args.official_allowlist, TRAIN_SCENES)
        for name, scenes in OFFICIAL_SPLITS:
            records = [row for row in official if row["scene"] in scenes]
            if not records:
                raise ValueError(f"Official list has no samples for {name}")
            write_manifest(output / f"{name}.json", records, "official_train_manifest", args.official_allowlist, args.official_source)
        summary.update(training_admitted=True, official_train_samples=len(official))
    if args.development_pool:
        counts = {}
        for name, scenes in DEVELOPMENT_SPLITS:
            if not scenes <= TRAIN_SCENES:
                raise RuntimeError("Development splits may only use training scenes 1,3,5,6,7,8")
            records = [row for row in train if row["scene"] in scenes]
            if not records:
                raise ValueError(f"Public training pool has no samples for {name}")
            if any(row["scene"] in TEST_SCENES or row["scene"] not in TRAIN_SCENES for row in records):
                raise RuntimeError("Development manifest would include a test scene")
            write_manifest(output / f"{name}.json", records, DEVELOPMENT_KIND, note=DEVELOPMENT_NOTE,
                           extra={"scenes": sorted(scenes),
                                  "source": "public LuSNAR training-scene pool (Moon_i/image0 color+label)"})
            counts[name] = {"samples": len(records), "scenes": sorted(scenes)}
        summary.update(development_pool_written=True, development_pool_admitted_with_opt_in=True,
                       development_manifests=counts,
                       development_note="Method development only; final submission model must use the organizer's list.")
    if args.validate_labels:
        # The development pool covers every discovered training-scene record (a superset of any official list).
        targets = train if args.development_pool else official
        summary["labels_validated"] = True
        summary["labels_validated_count"] = validate_labels(root, targets, args.workers)
        summary["label_validation_scope"] = ("all discovered training-scene labels (1,3,5,6,7,8)"
                                             if args.development_pool else "admitted official training samples")
        summary["label_validation_workers"] = args.workers
    elif official is not None or args.development_pool:
        summary["labels_validated"] = False
    if args.official_test_allowlist:
        official_test = admitted(test, args.official_test_allowlist, TEST_SCENES)
        write_manifest(output / "test.json", official_test, "official_test_manifest", args.official_test_allowlist, args.official_source)
        summary["official_test_samples"] = len(official_test)
    write_json(summary, output / "preparation_summary.json")
    print(summary)
    if not args.official_allowlist:
        if args.development_pool:
            print("Development pool written (public training scenes only, method development only). "
                  "Official training sample list is still required for the final submission model.")
        else:
            print("Discovery only. Official training sample list is required before training; no training manifests were authorized.")


if __name__ == "__main__":
    main()
