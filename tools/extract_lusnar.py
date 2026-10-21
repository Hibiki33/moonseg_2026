#!/usr/bin/env python3
"""Extract only left RGB and permitted training labels from the public ZIPs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import zipfile

TRAIN_SCENES = {1, 3, 5, 6, 7, 8}


def select_member(name, scene):
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe archive member: {name}")
    parts = [part.lower() for part in path.parts]
    if len(parts) < 3 or path.suffix.lower() != ".png":
        return None
    for part in parts:
        match = re.fullmatch(r"moon[_-]?(\d+)", part)
        if match and int(match[1]) != scene:
            raise ValueError(f"Archive scene/path disagreement: {name}")
    # Verified public ZIP layout: image0 is the annotated left camera;
    # image1/color is the right camera and must never pass the generic branch.
    if parts[-3:-1] == ["image0", "color"]:
        return "color", path.name
    if scene in TRAIN_SCENES and parts[-3:-1] == ["image0", "label"]:
        return "label", path.name
    # Also accept a pre-organized challenge package with explicit color/label.
    has_camera_directory = any(re.fullmatch(r"image\d+", part) for part in parts)
    if not has_camera_directory and parts[-2] == "color":
        return "color", path.name
    if scene in TRAIN_SCENES and not has_camera_directory and parts[-2] == "label":
        return "label", path.name
    return None


def extract(archive, output, scene):
    if Path(str(archive) + ".aria2").exists():
        raise ValueError(f"Download is incomplete: {archive}")
    with zipfile.ZipFile(archive) as bundle:
        selected, seen = [], set()
        for item in bundle.infolist():
            if item.is_dir():
                continue
            relative = select_member(item.filename, scene)
            if relative is None:
                continue
            if relative in seen:
                raise ValueError(f"Duplicate output path in archive: {relative}")
            seen.add(relative)
            selected.append((item, relative))
        images = {name for kind, name in seen if kind == "color"}
        labels = {name for kind, name in seen if kind == "label"}
        if not images or (scene in TRAIN_SCENES and images != labels):
            raise ValueError(
                f"Moon_{scene}: found {len(images)} left images and {len(labels)} permitted labels; "
                "archive layout or timestamp pairing differs from documented layout. "
                "Inspect the ZIP filenames before changing extraction rules."
            )
        for item, relative in selected:
            destination = output / f"Moon_{scene}" / Path(*relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_file() and destination.stat().st_size == item.file_size:
                continue
            temporary = destination.with_suffix(destination.suffix + ".part")
            # Test labels are never opened or extracted. ZIP reads perform the
            # normal format consistency check, without a separate checksum pass.
            with bundle.open(item) as reader, temporary.open("wb") as writer:
                shutil.copyfileobj(reader, writer, length=1024 * 1024)
            temporary.replace(destination)
        record = {"archive": archive.name, "scene": scene, "rgb_count": len(images),
                  "label_count": len(labels), "camera": "left", "raw_camera_directory": "image0",
                  "pairing": "identical timestamp PNG filenames", "sample_ids": sorted(Path(n).stem for n in images),
                  "status": "public archive discovery, not an official challenge allowlist"}
        (output / f"Moon_{scene}/extraction.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"Moon_{scene}: {len(images)} left RGB; {len(labels)} training labels", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, default=Path("data/archives"))
    parser.add_argument("--output", type=Path, default=Path("data/LuSNAR"))
    parser.add_argument("--scenes", type=int, nargs="+", default=list(range(1, 10)))
    args = parser.parse_args()
    if len(args.scenes) != len(set(args.scenes)) or not set(args.scenes) <= set(range(1, 10)):
        parser.error("Scenes must be unique integers from 1 through 9")
    for scene in args.scenes:
        extract(args.archives / f"Moon_{scene}.zip", args.output, scene)


if __name__ == "__main__":
    main()
