"""Official DA3MONO encoder with a pixel-coordinate bridge to ViT-Adapter.

Only the depth-trained ViT is loaded. The upstream depth/camera heads are never built.
Native patch size remains 14; Adapter receives features on its existing stride-16 grid.
"""
from __future__ import annotations

import importlib
from pathlib import Path
import sys
import types

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def official_encoder_module(source=None):
    source = Path(source or Path(__file__).resolve().parents[2] / "third_party/depth_anything_3").resolve()
    root = source / "src"
    if not (root / "depth_anything_3/model/dinov2/vision_transformer.py").is_file():
        raise FileNotFoundError(f"Official DA3 encoder source missing at {source}")
    # Skip the upstream application API and its optional rendering/UI dependencies.
    for name in ("depth_anything_3", "depth_anything_3.model", "depth_anything_3.utils"):
        directory = root.joinpath(*name.split("."))
        if name in sys.modules:
            if list(sys.modules[name].__path__) != [str(directory)]:
                raise RuntimeError("Another DA3 source is imported; use a fresh Python process")
        else:
            package = types.ModuleType(name)
            package.__path__ = [str(directory)]
            package.__package__ = name
            package.__spec__ = importlib.util.spec_from_loader(name, loader=None, is_package=True)
            sys.modules[name] = package
    return importlib.import_module("depth_anything_3.model.dinov2.vision_transformer")


def bridge_native_tokens(tokens, image_size, native_patch_size=14, adapter_stride=16):
    """Sample native patch centers at Adapter pixel centers, excluding padded FOV.

    align_corners=False maps g to source index ((g+1)*N-1)/2. Thus g=2*p/(N*14)-1
    samples physical pixel p; the first native patch center is p=7. Padding exists
    only at right/bottom and never changes the image scale or target coordinates.
    """
    height, width = image_size
    padded_h = (height + native_patch_size - 1) // native_patch_size * native_patch_size
    padded_w = (width + native_patch_size - 1) // native_patch_size * native_patch_size
    batch, views, count, dim = tokens.shape
    if views != 1 or count != (padded_h // native_patch_size) * (padded_w // native_patch_size):
        raise ValueError("Expected independent single-view native DA3 patch tokens")
    target_h, target_w = height // adapter_stride, width // adapter_stride
    y = (torch.arange(target_h, device=tokens.device, dtype=torch.float32) + .5) * adapter_stride
    x = (torch.arange(target_w, device=tokens.device, dtype=torch.float32) + .5) * adapter_stride
    yy, xx = torch.meshgrid(2 * y / padded_h - 1, 2 * x / padded_w - 1, indexing="ij")
    grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
    feature = tokens[:, 0].transpose(1, 2).reshape(batch, dim, padded_h // native_patch_size,
                                                padded_w // native_patch_size)
    with torch.autocast(device_type=tokens.device.type, enabled=False):
        result = F.grid_sample(feature.float(), grid, mode="bilinear", padding_mode="border",
                               align_corners=False)
    return result.to(tokens.dtype).flatten(2).transpose(1, 2)


def build_da3mono(settings, load_weights=True):
    official = official_encoder_module(settings.get("da3_repo"))

    class DA3MonoBackbone(official.DinoVisionTransformer):
        adapter_patch_size = 16
        model_family = "da3mono-large"

        def process_attention(self, x, block, attn_type="global", pos=None, attn_mask=None):
            # Adapter deliberately keeps the encoder in eval mode even during fine tuning.
            # Non-reentrant checkpointing works for trainable parameters even if x needs no grad.
            def run(value):
                return super(DA3MonoBackbone, self).process_attention(
                    value, block, attn_type=attn_type, pos=pos, attn_mask=attn_mask)
            if (self.gradient_checkpointing and torch.is_grad_enabled()
                    and any(parameter.requires_grad for parameter in block.parameters())):
                return checkpoint(run, x, use_reentrant=False, preserve_rng_state=True)
            return run(x)

        def get_intermediate_layers(self, images, n=1, return_class_token=False):
            height, width = images.shape[-2:]
            padded = F.pad(images, (0, (-width) % self.patch_size, 0, (-height) % self.patch_size))
            # B independent images, one view each. Call the differentiable encoder directly.
            layers, _ = super().get_intermediate_layers(padded.unsqueeze(1), n=n)
            outputs = []
            for tokens, camera_token in layers:
                tokens = bridge_native_tokens(tokens, (height, width), self.patch_size, self.adapter_patch_size)
                outputs.append((tokens, camera_token[:, 0]) if return_class_token else tokens)
            return tuple(outputs)

    encoder = DA3MonoBackbone(img_size=518, patch_size=14, embed_dim=1024, depth=24,
                              num_heads=16, mlp_ratio=4, ffn_layer="mlp", num_register_tokens=0,
                              alt_start=-1, qknorm_start=-1, rope_start=-1, cat_token=False)
    encoder.gradient_checkpointing = bool(settings.get("backbone_gradient_checkpointing", True))
    if load_weights:
        from safetensors import safe_open
        path = Path(settings["backbone_weights"]).expanduser()
        prefix = "model.backbone.pretrained."
        with safe_open(path, framework="pt", device="cpu") as handle:
            state = {key[len(prefix):]: handle.get_tensor(key) for key in handle.keys() if key.startswith(prefix)}
        encoder.load_state_dict(state, strict=True)
    return encoder
