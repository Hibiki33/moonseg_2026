#!/usr/bin/env python3
"""Discover left RGB frames, then admit only samples on explicit official lists.

Without --official-allowlist this produces exploration manifests only. They cannot
be used by the training/evaluation loaders. Never derive an "official" allowlist
from the public archives: obtain the fixed sample list from the organizers.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from moonseg.data.labels import decode_label
from moonseg.data.manifest import TEST_SCENES, TRAIN_SCENES, is_allowed, read_allowlist
from moonseg.utils import write_json


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


def write_manifest(path, records, kind, allowlist=None, source=None):
    write_json({"version": 1, "kind": kind, "official_allowlist": str(allowlist) if allowlist else None,
                "official_list_source": source,
                "note": ("Explicit official sample membership required; scene membership alone grants no training permission."
                         if "exploration" in kind else "Filtered against the supplied explicit official sample list."),
                "samples": records}, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="data/LuSNAR")
    parser.add_argument("--output-dir", default="data/manifests")
    parser.add_argument("--image-template", default="Moon_{scene}/color")
    parser.add_argument("--label-template", default="Moon_{scene}/label")
    parser.add_argument("--official-allowlist", help="Organizer's training sample list: JSON/JSONL/CSV or IDs.txt")
    parser.add_argument("--official-test-allowlist", help="Organizer's complete test sample list")
    parser.add_argument("--official-source", help="Organizer list URL or supplied file provenance, stored in manifests")
    parser.add_argument("--validate-labels", action="store_true", help="Validate colors/dimensions ONLY for admitted official training samples")
    args = parser.parse_args()
    root, output = Path(args.root).resolve(), Path(args.output_dir)
    if args.validate_labels and not args.official_allowlist:
        parser.error("--validate-labels requires --official-allowlist; unassigned public labels must not inform development")
    if not root.is_dir():
        parser.error(f"Dataset root missing: {root}")
    train = discover(root, TRAIN_SCENES, args.image_template, args.label_template, True)
    test = discover(root, TEST_SCENES, args.image_template, args.label_template, False)
    if not train and not test:
        parser.error("No PNG images found; check --root and --image-template")
    all_ids = [row["sample_id"] for row in train + test]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Duplicate sample_id across training and test scenes")
    write_manifest(output / "exploration_train.json", train, "public_exploration_manifest")
    write_manifest(output / "exploration_test.json", test, "public_exploration_manifest")
    summary = {"public_train_pool_discovered": len(train), "public_test_images_discovered": len(test),
               "test_labels_opened": 0, "training_admitted": False}
    if args.official_allowlist:
        official = admitted(train, args.official_allowlist, TRAIN_SCENES)
        for name, scenes in (("train", {1, 3, 6, 7, 8}), ("val", {5}), ("final_train", TRAIN_SCENES)):
            records = [row for row in official if row["scene"] in scenes]
            if not records:
                raise ValueError(f"Official list has no samples for {name}")
            write_manifest(output / f"{name}.json", records, "official_train_manifest", args.official_allowlist, args.official_source)
        if args.validate_labels:
            for record in official:
                with Image.open(root / record["label"]) as label:
                    decoded = decode_label(label)
                with Image.open(root / record["image"]) as image:
                    if decoded.shape != (image.height, image.width):
                        raise ValueError(f"Image/label size mismatch: {record['sample_id']}")
        summary.update(training_admitted=True, official_train_samples=len(official), labels_validated=args.validate_labels)
    if args.official_test_allowlist:
        official_test = admitted(test, args.official_test_allowlist, TEST_SCENES)
        write_manifest(output / "test.json", official_test, "official_test_manifest", args.official_test_allowlist, args.official_source)
        summary["official_test_samples"] = len(official_test)
    write_json(summary, output / "preparation_summary.json")
    print(summary)
    if not args.official_allowlist:
        print("Discovery only. Official training sample list is required before training; no training manifests were authorized.")


if __name__ == "__main__":
    main()
