#!/usr/bin/env python3
"""Resumable parallel asset downloads; stdlib + aria2, no checksum pass."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from urllib.parse import quote
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parents[1]
DINOV3_REVISION = "6876159a11b4df116f30f667f8c9888617df0751"
DATASET_REVISION = "df281ef5139f78a21a162d684186efd3c2ae3437"
DATASET_ID = "xuboluo2001/LuSNAR"
NATIVE_NAMES = {
    "vitb16": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "vitl16": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
}


def get_json(url):
    with urlopen(Request(url, headers={"User-Agent": "MoonSeg/0.1"}), timeout=60) as response:
        return json.load(response)


def download(entries, *, jobs=3, connections=8):
    """entries: (url, destination, expected_size_or_None); aria2 resumes its sidecar."""
    if shutil.which("aria2c") is None:
        raise RuntimeError("aria2c is required for parallel, resumable downloads")
    pending = []
    for url, destination, size in entries:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        complete = destination.is_file() and not Path(str(destination) + ".aria2").exists()
        if complete and (size is None or destination.stat().st_size == size):
            print(f"Present: {destination.name}", flush=True)
            continue
        if complete and size is not None:
            raise RuntimeError(f"Unexpected file size: {destination}; move this file aside before retrying")
        if any(char in url + str(destination) for char in "\r\n"):
            raise ValueError("URLs and paths must not contain newlines")
        pending.append((url, destination, size))
    if not pending:
        return
    # Signed URLs are stored only in a private transient input file, never in logs.
    with tempfile.NamedTemporaryFile(mode="w", prefix="moonseg-aria2-", delete=False) as stream:
        input_path = Path(stream.name)
        for url, destination, _ in pending:
            stream.write(f"{url}\n  dir={destination.parent}\n  out={destination.name}\n")
    try:
        subprocess.run([
            "aria2c", "--input-file=" + str(input_path), "--continue=true",
            "--auto-file-renaming=false", "--allow-overwrite=false",
            f"--max-concurrent-downloads={jobs}", f"--max-connection-per-server={connections}",
            f"--split={connections}", "--min-split-size=8M", "--max-tries=8",
            "--retry-wait=5", "--connect-timeout=20", "--timeout=60",
            "--file-allocation=none", "--summary-interval=15", "--console-log-level=warn",
            "--download-result=hide", "--enable-color=false",
        ], check=True)
    finally:
        input_path.unlink(missing_ok=True)
    for _, destination, size in pending:
        if not destination.is_file() or Path(str(destination) + ".aria2").exists():
            raise RuntimeError(f"Incomplete download: {destination}")
        if size is not None and destination.stat().st_size != size:
            raise RuntimeError(f"Download size mismatch: {destination}")


def source(args):
    target = REPO / "third_party/dinov3"
    marker = target / "SOURCE_COMPLETE"
    if marker.exists() and marker.read_text().strip() == DINOV3_REVISION:
        print("Complete official DINOv3 source already present")
        return
    with tempfile.TemporaryDirectory(prefix="moonseg-source-") as directory:
        archive = Path(directory) / "dinov3.tar.gz"
        url = f"https://codeload.github.com/facebookresearch/dinov3/tar.gz/{DINOV3_REVISION}"
        download([(url, archive, None)], jobs=1, connections=args.connections)
        with tarfile.open(archive) as bundle:
            members = bundle.getmembers()
            root = f"dinov3-{DINOV3_REVISION}"
            for item in members:
                parts = Path(item.name).parts
                if not parts or parts[0] != root or ".." in parts or item.issym() or item.islnk():
                    raise ValueError(f"Unexpected source archive member: {item.name}")
                if item.isfile() and len(parts) > 1:
                    destination = target.joinpath(*parts[1:])
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.extractfile(item) as reader, destination.open("wb") as writer:
                        shutil.copyfileobj(reader, writer)
    (target / "SOURCE_REVISION").write_text(DINOV3_REVISION + "\n")
    marker.write_text(DINOV3_REVISION + "\n")
    (target / "SOURCE_MANIFEST.json").write_text(json.dumps({
        "revision": DINOV3_REVISION, "source": url, "kind": "complete_official_archive"
    }, indent=2) + "\n")


def link_assets(root):
    for name, target in (("data", root / "data"), ("ckpts", root / "checkpoints")):
        target.mkdir(parents=True, exist_ok=True)
        link = REPO / name
        if link.is_symlink() and link.resolve() == target.resolve():
            continue
        if link.exists() or link.is_symlink():
            raise FileExistsError(f"Refusing to replace existing {link}; expected link to {target}")
        link.symlink_to(target, target_is_directory=True)


def dataset(args):
    endpoint = args.hf_endpoint.rstrip("/")
    api = f"{endpoint}/api/datasets/{DATASET_ID}/tree/{DATASET_REVISION}?recursive=true&expand=false"
    files = get_json(api)
    if not isinstance(files, list):
        raise ValueError("Unexpected Hugging Face tree response")
    files = {entry["path"]: entry for entry in files if entry.get("type") == "file"}
    archive_dir = args.storage / "data/archives"
    entries, record = [], []
    for scene in args.scenes:
        name = f"Moon_{scene}.zip"
        if name not in files:
            raise FileNotFoundError(f"{name} is absent at the recorded dataset revision")
        size = int(files[name]["size"])
        url = f"{endpoint}/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/{name}"
        entries.append((url, archive_dir / name, size))
        record.append({"scene": scene, "file": name, "bytes": size, "url": url})
    if args.dataset_engine == "curl":
        from download_ranges import download_ranges
        download_ranges(entries, jobs=args.jobs, connections=args.connections,
                        dns_resolver=args.dns_resolver, resolve_overrides=args.resolve_host)
    else:
        download(entries, jobs=args.jobs, connections=args.connections)
    (archive_dir / "download_manifest.json").write_text(json.dumps({
        "repository": DATASET_ID, "revision": DATASET_REVISION, "files": record,
        "validation": "remote byte length and downloader completion; no checksum pass",
        "engine": args.dataset_engine,
    }, indent=2) + "\n")
    if args.extract:
        subprocess.run([sys.executable, str(REPO / "tools/extract_lusnar.py"),
                        "--archives", str(archive_dir), "--output", str(args.storage / "data/LuSNAR"),
                        "--scenes", *map(str, args.scenes)], check=True)


def weights(args):
    output = args.storage / "checkpoints"
    name = NATIVE_NAMES[args.backbone]
    provenance_path = output / (name + ".source.json")
    if (output / name).is_file() and provenance_path.is_file():
        print(f"Prepared checkpoint already present: {output / name}")
        return
    if (output / name).exists() and args.weights_provider != "meta":
        raise FileExistsError(f"Existing checkpoint without download provenance: {output / name}; preserve or move it before conversion")
    if args.weights_provider == "meta":
        url = os.environ.get("DINOV3_WEIGHTS_URL")
        if not url:
            raise RuntimeError("Set DINOV3_WEIGHTS_URL to the authorized Meta native .pth download URL")
        download([(url, output / name, None)], jobs=1, connections=args.connections)
        metadata = {"provider": "Meta", "format": "official native PyTorch", "backbone": args.backbone,
                    "pretraining": "LVD-1689M", "file": name, "url": "authorized URL omitted"}
    else:
        model_id = f"facebook/dinov3-{args.backbone}-pretrain-lvd1689m"
        if args.weights_provider == "modelscope":
            base = f"https://modelscope.cn/models/{model_id}/resolve/{args.modelscope_revision}"
        else:
            base = f"{args.hf_endpoint.rstrip('/')}/{model_id}/resolve/main"
        raw = output / ("modelscope" if args.weights_provider == "modelscope" else "huggingface") / args.backbone
        # The format is checked by the converter before the resulting .pth can be used.
        download([(base + "/" + quote(file), raw / file, None)
                  for file in ("config.json", "model.safetensors", "README.md", "LICENSE.md")],
                 jobs=args.jobs, connections=args.connections)
        subprocess.run([sys.executable, str(REPO / "tools/convert_dinov3_weights.py"),
                        "--input", str(raw), "--output", str(output / name),
                        "--backbone", args.backbone], check=True)
        metadata = {"provider": args.weights_provider, "repository": model_id, "source": base,
                    "format": "Transformers safetensors converted to official native state dict",
                    "backbone": args.backbone, "pretraining": "LVD-1689M", "file": name,
                    "revision": args.modelscope_revision if args.weights_provider == "modelscope" else "main",
                    "note": "Original model card and license retained next to downloaded files"}
    provenance_path.write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", type=Path, default=Path("/home/disk02/wjj/moonseg"))
    parser.add_argument("--only", choices=("all", "data", "weights", "source", "links"), default="all")
    parser.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    parser.add_argument("--scenes", nargs="+", type=int, default=list(range(1, 10)))
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--connections", type=int, default=8)
    parser.add_argument("--dataset-engine", choices=("aria2", "curl"), default="aria2")
    parser.add_argument("--dns-resolver", help="Optional HTTPS JSON DNS endpoint for curl CAS downloads")
    parser.add_argument("--resolve-host", action="append", default=[], help="curl HOST:PORT:ADDRESS override, TLS remains verified")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--backbone", choices=("vitb16", "vitl16"), default="vitb16")
    parser.add_argument("--weights-provider", choices=("modelscope", "hf", "meta"), default="modelscope")
    parser.add_argument("--modelscope-revision", default="23d0280ae6ee4ced592a3459674ad027d3c18906")
    parser.add_argument("--plan", action="store_true", help="Print configuration without network or filesystem changes")
    args = parser.parse_args()
    args.storage = args.storage.expanduser().absolute()
    if len(args.scenes) != len(set(args.scenes)) or not set(args.scenes) <= set(range(1, 10)):
        parser.error("Scenes must be unique integers from 1 through 9")
    if args.jobs < 1 or not 1 <= args.connections <= 16:
        parser.error("jobs must be positive and connections must be between 1 and 16")
    if args.plan:
        print(json.dumps({**vars(args), "storage": str(args.storage), "source_revision": DINOV3_REVISION,
                          "dataset_revision": DATASET_REVISION}, indent=2))
        return
    if args.only != "source":
        link_assets(args.storage)
    actions = {"source": source, "data": dataset, "weights": weights}
    for action in (("source", "data", "weights") if args.only == "all" else (args.only,)):
        if action in actions:
            actions[action](args)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Asset preparation failed: {error}", file=sys.stderr)
        sys.exit(1)
