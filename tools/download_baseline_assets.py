#!/usr/bin/env python3
"""Download the three pinned baseline checkpoints; official source is vendored.

Only byte lengths and checkpoint key/shape inventories are checked. Existing
source records are preserved. No training, inference, or environment installation
is performed by this command.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.parse import urlparse

REPO = Path(__file__).resolve().parents[1]
# Values transcribed from the downloaded assets' adjacent .source.json records.
SAM3_REVISION = "f8ee5d16d5c30d0e87b1d22cbd358ce97761df56"
MIT_REVISION = "40357155205b036cf11b61f132d53d2f8861f170"
MOBILE_REVISION = "96f46a1c52932f27492dff66c72378eb99b443a7"


@dataclass(frozen=True)
class Asset:
    family: str
    relative_path: str
    repository: str
    revision: str
    remote_file: str
    byte_size: int
    kind: str

    def url(self, hf_endpoint):
        if self.family == "sam3":
            return f"https://modelscope.cn/models/{self.repository}/resolve/{self.revision}/{self.remote_file}"
        return f"{hf_endpoint}/{self.repository}/resolve/{self.revision}/{self.remote_file}"


ASSETS = (
    Asset("sam3", "sam3/sam3.pt", "facebook/sam3", SAM3_REVISION, "sam3.pt", 3450062241, "sam3"),
    Asset("segformer", "segformer/nvidia_mit_b5_imagenet.pytorch_model.bin", "nvidia/mit-b5",
          MIT_REVISION, "pytorch_model.bin", 328277459, "mit_hf"),
    Asset("deeplabv3plus", "deeplabv3plus/mobilenetv3_large_100.ra_in1k.safetensors",
          "timm/mobilenetv3_large_100.ra_in1k", MOBILE_REVISION, "model.safetensors", 22058321, "mobile"),
)
MIT_NATIVE = "segformer/mit_b5_imagenet.pth"


def inspect_checkpoint(path, kind, expected_bytes=None):
    """Read tensor metadata on CPU/meta, without constructing a model or using a GPU."""
    path = Path(path)
    size = path.stat().st_size
    if expected_bytes is not None and size != expected_bytes:
        raise ValueError(f"Checkpoint size mismatch: {path}: {size}, expected {expected_bytes}")
    if kind == "mit_native" and not 320_000_000 <= size <= 330_000_000:
        raise ValueError(f"Converted MiT-B5 checkpoint size is implausible: {path}: {size}")
    if Path(str(path) + ".aria2").exists():
        raise ValueError(f"Incomplete aria2 checkpoint: {path}")
    if kind == "mobile":
        from safetensors import safe_open
        with safe_open(path, framework="pt", device="cpu") as handle:
            shapes = {key: tuple(handle.get_slice(key).get_shape()) for key in handle.keys()}
        count, probes = 312, {"conv_stem.weight": (16, 3, 3, 3), "classifier.weight": (1000, 1280)}
    else:
        import torch
        state = torch.load(path, map_location="meta", weights_only=True, mmap=True)
        if kind == "mit_native":
            state = state.get("state_dict", state)
        elif kind == "sam3":
            state = state.get("model", state)
        if not isinstance(state, dict):
            raise ValueError(f"Checkpoint is not a tensor dictionary: {path}")
        shapes = {key: tuple(value.shape) for key, value in state.items() if isinstance(value, torch.Tensor)}
        if len(shapes) != len(state):
            raise ValueError(f"Checkpoint contains unexpected non-tensor entries: {path}")
        if kind == "sam3":
            count, probes = 1465, {
                "detector.backbone.vision_backbone.trunk.patch_embed.proj.weight": (1024, 3, 14, 14),
                "detector.segmentation_head.semantic_seg_head.weight": (1, 256, 1, 1),
            }
        elif kind == "mit_hf":
            count, probes = 1158, {"segformer.encoder.patch_embeddings.0.proj.weight": (64, 3, 7, 7),
                                   "classifier.weight": (1000, 512)}
        elif kind == "mit_native":
            count, probes = 1052, {"patch_embed1.proj.weight": (64, 3, 7, 7),
                                   "block1.0.attn.kv.weight": (128, 64), "norm4.weight": (512,)}
        else:
            raise ValueError(f"Unknown checkpoint kind: {kind}")
    if len(shapes) != count or any(shapes.get(key) != shape for key, shape in probes.items()):
        raise ValueError(f"Unexpected {kind} checkpoint keys/shapes: {path}")
    return {"bytes": size, "tensor_keys": len(shapes), "representative_shapes": probes}


def source_path(asset, root):
    destination = root / (MIT_NATIVE if asset.family == "segformer" else asset.relative_path)
    return Path(str(destination) + ".source.json")


def recorded_source(asset, root):
    """Read existing provenance, refusing to reinterpret a different model/version."""
    path = source_path(asset, root)
    if not path.exists():
        return {}
    record = json.loads(path.read_text())
    revision = record.get("mirror_revision" if asset.family == "sam3" else "weight_revision")
    size = record.get("expected_bytes" if asset.family == "sam3" else "weight_bytes")
    if revision != asset.revision or size != asset.byte_size:
        raise ValueError(f"Existing provenance disagrees with the pinned baseline: {path}")
    return record


def save_source_once(path, record):
    """Never replace an existing source record, including one written by another process."""
    try:
        with Path(path).open("x") as stream:
            stream.write(json.dumps(record, indent=2) + "\n")
    except FileExistsError:
        pass


def ensure_ckpts_link(storage, dry_run=False):
    destination = storage / "checkpoints"
    link = REPO / "ckpts"
    if link.is_symlink() and link.resolve() == destination.resolve():
        return
    if link.exists() or link.is_symlink():
        raise FileExistsError(f"Refusing to replace {link}; expected a symlink to {destination}")
    if not dry_run:
        destination.mkdir(parents=True, exist_ok=True)
        link.symlink_to(destination, target_is_directory=True)


def download_file(asset, root, args):
    destination = root / asset.relative_path
    if destination.exists():
        info = inspect_checkpoint(destination, asset.kind, asset.byte_size)
        print(f"Present and validated: {destination} ({info['tensor_keys']} tensor keys)", flush=True)
        return destination
    partial = destination.with_name(destination.name + ".part")
    # A completed staging file can remain after interruption before final rename.
    if partial.exists() and partial.stat().st_size == asset.byte_size and not Path(str(partial) + ".aria2").exists():
        inspect_checkpoint(partial, asset.kind, asset.byte_size)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        engine = args.engine
        if engine == "auto":
            curl_parts = partial.parent / ("." + partial.name + ".parts")
            if curl_parts.exists() or args.dns_resolver or args.resolve_host:
                engine = "curl"
            else:
                engine = "aria2" if shutil.which("aria2c") else "curl"
        if engine == "aria2":
            if args.dns_resolver or args.resolve_host:
                raise ValueError("DNS overrides require --engine curl")
            if shutil.which("aria2c") is None:
                raise RuntimeError("aria2c is unavailable; use --engine curl")
            subprocess.run([
                "aria2c", "--continue=true", "--check-certificate=true", "--auto-file-renaming=false",
                "--allow-overwrite=false", f"--max-connection-per-server={args.connections}",
                f"--split={args.connections}", "--min-split-size=8M", "--max-tries=8", "--retry-wait=5",
                "--connect-timeout=20", "--timeout=60", "--file-allocation=none", "--summary-interval=15",
                "--console-log-level=warn", "--download-result=hide", "--enable-color=false",
                "--dir=" + str(partial.parent), "--out=" + partial.name, asset.url(args.hf_endpoint),
            ], check=True)
        else:
            from download_ranges import download_ranges
            download_ranges([(asset.url(args.hf_endpoint), partial, asset.byte_size)], jobs=1,
                            connections=args.connections, dns_resolver=args.dns_resolver,
                            resolve_overrides=args.resolve_host)
        inspect_checkpoint(partial, asset.kind, asset.byte_size)
    partial.replace(destination)
    return destination


def convert_mit(raw, destination):
    import torch
    sys.path.insert(0, str(REPO))
    from moonseg.models.segformer import convert_nvidia_mit_b5_state_dict
    state = torch.load(raw, map_location="cpu", weights_only=True, mmap=True)
    converted = convert_nvidia_mit_b5_state_dict(state)
    temporary = destination.with_name(destination.name + ".converting")
    with temporary.open("wb") as stream:
        torch.save({"state_dict": converted}, stream)
    inspect_checkpoint(temporary, "mit_native")
    temporary.replace(destination)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("sam3", "segformer", "deeplabv3plus", "all"), default="all")
    parser.add_argument("--storage", type=Path, default=REPO.parent,
                        help="Heavy-asset storage root; checkpoints go under STORAGE/checkpoints")
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--engine", choices=("auto", "aria2", "curl"), default="auto")
    parser.add_argument("--connections", type=int, default=8)
    parser.add_argument("--dns-resolver", help="Optional HTTPS DNS endpoint used by curl; TLS stays verified")
    parser.add_argument("--resolve-host", action="append", default=[], help="curl HOST:PORT:ADDRESS override")
    parser.add_argument("--dry-run", "--plan", action="store_true", help="Validate existing assets and show actions; write nothing")
    args = parser.parse_args(argv)
    args.storage = args.storage.expanduser().resolve()
    args.hf_endpoint = args.hf_endpoint.rstrip("/")
    if urlparse(args.hf_endpoint).scheme != "https" or not urlparse(args.hf_endpoint).netloc:
        parser.error("--hf-endpoint must be an HTTPS URL")
    if args.dns_resolver and urlparse(args.dns_resolver).scheme != "https":
        parser.error("--dns-resolver must be an HTTPS URL")
    if not 1 <= args.connections <= 16:
        parser.error("--connections must be between 1 and 16")
    if args.storage.is_relative_to(REPO):
        parser.error("--storage must be outside the code repository")
    ensure_ckpts_link(args.storage, dry_run=args.dry_run)
    root = args.storage / "checkpoints"
    for asset in ASSETS:
        if args.only not in ("all", asset.family):
            continue
        existing_record = recorded_source(asset, root)
        target = root / (MIT_NATIVE if asset.family == "segformer" else asset.relative_path)
        kind = "mit_native" if asset.family == "segformer" else asset.kind
        expected = None if asset.family == "segformer" else asset.byte_size
        if target.exists():
            info = inspect_checkpoint(target, kind, expected)
            print(json.dumps({"family": asset.family, "action": "skip_validated", "path": str(target),
                              "validation": info, "existing_source_record": bool(existing_record)}), flush=True)
            continue
        print(json.dumps({"family": asset.family, "action": "download_and_convert" if asset.family == "segformer" else "download",
                          "url": asset.url(args.hf_endpoint), "download_bytes": asset.byte_size,
                          "path": str(target), "resumable": True, "tls_verified": True}), flush=True)
        if args.dry_run:
            continue
        downloaded = download_file(asset, root, args)
        if asset.family == "segformer":
            convert_mit(downloaded, target)
        record = {"weight_repository": ("https://modelscope.cn/models/" if asset.family == "sam3" else
                                         "https://huggingface.co/") + asset.repository,
                  "weight_revision": asset.revision, "weight_file": asset.remote_file,
                  "weight_download_url": asset.url(args.hf_endpoint), "weight_bytes": asset.byte_size,
                  "validation": inspect_checkpoint(target, kind, expected), "tls_verification": True,
                  "pretraining": "Original SAM3" if asset.family == "sam3" else "ImageNet-1K encoder only"}
        if asset.family == "sam3":
            record.update(mirror_revision=asset.revision, expected_bytes=asset.byte_size,
                          license="SAM License; see third_party/sam3/LICENSE")
        elif asset.family == "segformer":
            record["mapping"] = "moonseg.models.segformer.convert_nvidia_mit_b5_state_dict"
        save_source_once(source_path(asset, root), record)


if __name__ == "__main__":
    main()
