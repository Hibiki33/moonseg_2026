"""Convert a local Transformers DINOv3 ViT-B/L/H+ snapshot to Meta's state_dict.

Mapping reference (accessed 2026-10-21):
https://github.com/huggingface/transformers/blob/main/src/transformers/models/dinov3_vit/convert_dinov3_vit_to_hf.py
The original HF conversion discards masked K biases; this restores zeros there,
not their original serialized values. Effective attention remains unchanged.
ViT-H+ uses a SwiGLU MLP: HF down_proj(silu(gate_proj(x)) * up_proj(x)) equals
native SwiGLUFFN w3(silu(w1(x)) * w2(x)), so gate_proj->w1, up_proj->w2, down_proj->w3.
"""
from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
# width, depth, heads, HF intermediate_size, gated (SwiGLU) MLP. The H+ hidden size is
# native SwiGLUFFN's int(1280 * ffn_ratio 6 * 2/3) = 5120 (already a multiple of align_to=8).
SIZES = {
    "vitb16": (768, 12, 12, 3072, False),
    "vitl16": (1024, 24, 16, 4096, False),
    "vith16plus": (1280, 32, 20, 5120, True),
}


def check_config(config: dict, arch: str) -> None:
    width, depth, heads, intermediate, gated = SIZES[arch]
    required = {
        "model_type": "dinov3_vit", "hidden_size": width,
        "intermediate_size": intermediate, "num_hidden_layers": depth,
        "num_attention_heads": heads, "num_register_tokens": 4,
    }
    defaults = {
        "num_channels": 3, "use_gated_mlp": False, "hidden_act": "gelu",
        "query_bias": True, "key_bias": False, "value_bias": True,
        "proj_bias": True, "mlp_bias": True, "rope_theta": 100.0,
        "layer_norm_eps": 1e-5, "pos_embed_shift": None,
        "pos_embed_jitter": None, "pos_embed_rescale": 2.0,
    }
    if gated:
        # Transformers defaults to a non-gated GELU MLP; the SwiGLU variant must state both.
        for name, expected in (("use_gated_mlp", True), ("hidden_act", "silu")):
            del defaults[name]
            required[name] = expected
    for name, expected in required.items():
        if config.get(name) != expected:
            raise ValueError(f"config {name}: expected {expected!r}, got {config.get(name)!r}")
    for name, expected in defaults.items():
        if config.get(name, expected) != expected:
            raise ValueError(f"Unsupported config {name}={config[name]!r}; expected {expected!r}")
    if config.get("patch_size", 16) not in (16, [16, 16]):
        raise ValueError("Only 16x16 patch embeddings are supported")


def native_model(arch: str, repo: Path):
    sys.path.insert(0, str(repo.resolve()))
    from dinov3.hub import backbones

    # Upstream imports torch._dynamo lazily when its model module is first used.
    # Initialize that module before the meta-device context: PyTorch 2.7's
    # default-device interception otherwise recurses during Dynamo initialization.
    importlib.import_module("dinov3.models.vision_transformer")
    # Build schema without allocating hundreds of MB of unused random weights.
    with torch.device("meta"):
        return getattr(backbones, f"dinov3_{arch}")(pretrained=False)


def read_snapshot(directory: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    index = directory / "model.safetensors.index.json"
    if index.is_file():
        mapping = json.loads(index.read_text())["weight_map"]
        files = sorted(set(mapping.values()))
    else:
        mapping = None
        files = ["model.safetensors"]
    state = {}
    for name in files:
        path = (directory / name).resolve()
        if directory.resolve() not in path.parents:
            raise ValueError(f"Shard path escapes snapshot: {name}")
        tensors = load_file(str(path), device="cpu")
        if state.keys() & tensors.keys():
            raise ValueError(f"Duplicate tensor keys in shard {name}")
        if mapping is not None and any(mapping.get(k) != name for k in tensors):
            raise ValueError(f"Shard {name} does not match weight_map")
        state.update(tensors)
    if mapping is not None and state.keys() != mapping.keys():
        raise ValueError("Shard keys do not match safetensors index")
    return state


def convert_state(state: dict[str, torch.Tensor], model, arch: str) -> dict[str, torch.Tensor]:
    """Fail on unknown, missing, duplicate or incorrectly shaped tensors."""
    width, depth, heads, _, gated = SIZES[arch]
    mlp_names = ({"gate_proj": "w1", "up_proj": "w2", "down_proj": "w3"} if gated
                 else {"up_proj": "fc1", "down_proj": "fc2"})
    expected = model.state_dict()
    output: dict[str, torch.Tensor] = {}
    parts: dict[tuple[int, str], dict[str, torch.Tensor]] = {}

    def put(name, value):
        if name not in expected:
            raise ValueError(f"Unexpected target key: {name}")
        if name in output:
            raise ValueError(f"Duplicate target key: {name}")
        if value.shape != expected[name].shape:
            raise ValueError(f"Shape mismatch for {name}: {tuple(value.shape)} != {tuple(expected[name].shape)}")
        if not value.is_floating_point():
            raise ValueError(f"Non-floating tensor: {name}")
        output[name] = value

    direct = {
        "embeddings.cls_token": "cls_token",
        "embeddings.mask_token": "mask_token",
        "embeddings.register_tokens": "storage_tokens",
        "embeddings.patch_embeddings.weight": "patch_embed.proj.weight",
        "embeddings.patch_embeddings.bias": "patch_embed.proj.bias",
        "norm.weight": "norm.weight", "norm.bias": "norm.bias",
    }
    for key, value in state.items():
        if key in direct:
            if key == "embeddings.mask_token":
                if value.shape != (1, 1, width):
                    raise ValueError(f"Unexpected mask token shape: {tuple(value.shape)}")
                value = value.squeeze(1)
            put(direct[key], value)
            continue
        match = re.fullmatch(r"(?:model\.)?layer\.(\d+)\.(.+)", key)
        if match is None:
            raise ValueError(f"Unknown Transformers tensor key: {key}")
        index, tail = int(match[1]), match[2]
        if not 0 <= index < depth:
            raise ValueError(f"Layer index out of range: {key}")
        qkv = re.fullmatch(r"attention\.([qkv])_proj\.(weight|bias)", tail)
        if qkv:
            group = parts.setdefault((index, qkv[2]), {})
            if qkv[1] in group:
                raise ValueError(f"Duplicate QKV component: {key}")
            group[qkv[1]] = value
            continue
        destination = None
        if re.fullmatch(r"norm[12]\.(weight|bias)", tail):
            destination = tail
        elif re.fullmatch(r"attention\.o_proj\.(weight|bias)", tail):
            destination = tail.replace("attention.o_proj", "attn.proj")
        elif re.fullmatch(r"layer_scale[12]\.lambda1", tail):
            destination = f"ls{tail[len('layer_scale')]}.gamma"
        elif (mlp := re.fullmatch(r"mlp\.([a-z_]+)\.(weight|bias)", tail)) and mlp[1] in mlp_names:
            destination = f"mlp.{mlp_names[mlp[1]]}.{mlp[2]}"
        if destination is None:
            raise ValueError(f"Unknown layer tensor: {key}")
        put(f"blocks.{index}.{destination}", value)

    for index in range(depth):
        for kind in ("weight", "bias"):
            group = parts.get((index, kind), {})
            wanted = {"q", "k", "v"} if kind == "weight" else {"q", "v"}
            if group.keys() != wanted:
                raise ValueError(f"Layer {index} {kind}: expected {wanted}, got {set(group)}")
            shape = (width, width) if kind == "weight" else (width,)
            if any(tuple(t.shape) != shape for t in group.values()):
                raise ValueError(f"Layer {index}: incorrect QKV {kind} shape")
            if len({t.dtype for t in group.values()}) != 1:
                raise ValueError(f"Layer {index}: mixed QKV {kind} dtypes")
            key_tensor = group["k"] if kind == "weight" else torch.zeros_like(group["q"])
            put(f"blocks.{index}.attn.qkv.{kind}", torch.cat((group["q"], key_tensor, group["v"])))
        mask = torch.ones(3 * width, dtype=output[f"blocks.{index}.attn.qkv.bias"].dtype)
        mask[width:2 * width] = 0
        put(f"blocks.{index}.attn.qkv.bias_mask", mask)

    # HF drops this deterministic persistent buffer. Use the official initializer.
    from dinov3.layers.rope_position_encoding import RopePositionEmbedding

    rope = RopePositionEmbedding(width, num_heads=heads, base=100.0, dtype=torch.float32)
    put("rope_embed.periods", rope.periods)
    missing = expected.keys() - output.keys()
    if missing:
        raise ValueError(f"Missing native tensors: {sorted(missing)}")
    model.load_state_dict(output, strict=True, assign=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Local snapshot containing config.json/safetensors")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--arch", "--backbone", dest="arch", choices=SIZES, required=True)
    parser.add_argument("--dinov3-repo", type=Path, default=ROOT / "third_party/dinov3")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    config = json.loads((args.input / "config.json").read_text())
    check_config(config, args.arch)
    model = native_model(args.arch, args.dinov3_repo)
    output = convert_state(read_snapshot(args.input), model, args.arch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    if temporary.exists():
        parser.error(f"Temporary output already exists: {temporary}")
    try:
        torch.save(output, temporary)
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Converted {len(output)} tensors; official strict=True key/shape validation passed: {args.output}")
    print("K bias was reconstructed as zero; this is not a byte-identical copy of Meta's original checkpoint.")


if __name__ == "__main__":
    main()
