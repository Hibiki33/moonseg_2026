"""Independent DA3-LARGE-1.1 encoder, DPT fusion and Mask2Former queries.

DA3 source: ByteDance-Seed/Depth-Anything-3, revision 3d835ec1a5802d64a8b8b15f817a1ab54809bfe4.
The DPT pixel decoder follows DualDPT's shared reassembly and main fusion branch.
Only the native encoder is pretrained; no Adapter or depth/ray/camera prediction heads are built.
"""
from __future__ import annotations

import importlib
import math
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .da3mono import official_encoder_module
from .dinov3_mask2former import _official_modules


OUT_LAYERS = (11, 15, 19, 23)


def build_native_encoder(settings, load_weights=True):
    official = official_encoder_module(settings.get("da3_repo"))

    class NativeDA3Encoder(official.DinoVisionTransformer):
        def process_attention(self, x, block, attn_type="global", pos=None, attn_mask=None):
            def run(value):
                return super(NativeDA3Encoder, self).process_attention(
                    value, block, attn_type=attn_type, pos=pos, attn_mask=attn_mask)
            if (self.gradient_checkpointing and torch.is_grad_enabled()
                    and any(parameter.requires_grad for parameter in block.parameters())):
                return checkpoint(run, x, use_reentrant=False, preserve_rng_state=True)
            return run(x)

    encoder = NativeDA3Encoder(
        img_size=518, patch_size=14, embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4, ffn_layer="mlp", num_register_tokens=0,
        alt_start=8, qknorm_start=8, rope_start=8, cat_token=True,
    )
    encoder.gradient_checkpointing = bool(settings.get("backbone_gradient_checkpointing", True))
    if load_weights:
        from safetensors import safe_open
        path = Path(settings["backbone_weights"]).expanduser()
        prefix = "model.backbone.pretrained."
        with safe_open(path, framework="pt", device="cpu") as handle:
            state = {key[len(prefix):]: handle.get_tensor(key)
                     for key in handle.keys() if key.startswith(prefix)}
        result = encoder.load_state_dict(state, strict=True)
        encoder.pretrained_load_info = {
            "path": str(path), "tensors": len(state),
            "parameters": sum(value.numel() for value in state.values()),
            "missing_keys": list(result.missing_keys), "unexpected_keys": list(result.unexpected_keys),
        }
    return encoder


class DA3DPTPixelDecoder(nn.Module):
    """DualDPT main feature path, exposing coarse query scales and a mask field.

    Native patch grid -> x4/x2/x1/ceil(x0.5) reassembly -> top-down fusion.
    The final refinenet1 keeps the finest reassembled size instead of doubling it.
    This segmentation-specific choice keeps mask features at patch/4 spacing.
    """
    def __init__(self, dimension=256, gradient_checkpointing=False):
        super().__init__()
        dpt = importlib.import_module("depth_anything_3.model.dpt")
        self.add_position = dpt.DPT._add_pos_embed
        channels = (256, 512, 1024, 1024)
        self.norm = nn.LayerNorm(2048)
        self.projects = nn.ModuleList(nn.Conv2d(2048, width, 1) for width in channels)
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(256, 256, 4, stride=4),
            nn.ConvTranspose2d(512, 512, 2, stride=2),
            nn.Identity(), nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
        ])
        self.scratch = dpt._make_scratch(list(channels), dimension, expand=False)
        for index in range(1, 5):
            setattr(self.scratch, f"refinenet{index}", dpt._make_fusion_block(
                dimension, has_residual=index != 4, inplace=False))
        self.gradient_checkpointing = gradient_checkpointing

    def _forward_features(self, *features):
        resized = []
        ph, pw = features[0].shape[-2:]
        for feature, project, resize in zip(features, self.projects, self.resize_layers):
            feature = self.norm(feature.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
            feature = project(feature)
            feature = self.add_position(self, feature, pw * 14, ph * 14)
            resized.append(resize(feature))
        l1, l2, l3, l4 = [getattr(self.scratch, f"layer{index}_rn")(feature)
                          for index, feature in enumerate(resized, 1)]
        coarse = self.scratch.refinenet4(l4, size=l3.shape[-2:])
        middle = self.scratch.refinenet3(coarse, l3, size=l2.shape[-2:])
        fine = self.scratch.refinenet2(middle, l2, size=l1.shape[-2:])
        mask = self.scratch.refinenet1(fine, l1, size=l1.shape[-2:])
        # Mask2Former attends to the three coarsest levels; the finest DPT
        # output remains dedicated to mask prediction. Native spacings: 28/14/7.
        return mask, (l4, coarse, middle)

    def forward(self, features):
        if self.gradient_checkpointing and torch.is_grad_enabled():
            return checkpoint(self._forward_features, *features, use_reentrant=False,
                              preserve_rng_state=True)
        return self._forward_features(*features)


def crop_mask_field(masks, image_size, padded_size):
    """Crop continuous mask logits using physical image pixel centers.

    A target center p maps to normalized grid coordinate 2*p/padded_extent-1.
    Keeping a low-resolution mask field avoids materializing every auxiliary
    query at full image resolution while excluding the right/bottom padded FOV.
    """
    if tuple(image_size) == tuple(padded_size):
        return masks
    height, width = image_size
    padded_h, padded_w = padded_size
    target_h = max(1, math.ceil(height * masks.shape[-2] / padded_h))
    target_w = max(1, math.ceil(width * masks.shape[-1] / padded_w))
    y = (torch.arange(target_h, device=masks.device, dtype=torch.float32) + .5) * height / target_h
    x = (torch.arange(target_w, device=masks.device, dtype=torch.float32) + .5) * width / target_w
    yy, xx = torch.meshgrid(2 * y / padded_h - 1, 2 * x / padded_w - 1, indexing="ij")
    grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(masks.shape[0], -1, -1, -1)
    with torch.autocast(device_type=masks.device.type, enabled=False):
        result = F.grid_sample(masks.float(), grid, mode="bilinear", padding_mode="border",
                               align_corners=False)
    return result.to(masks.dtype)


class DA3DPTMask2Former(nn.Module):
    checkpoint_full_state = True
    native_backbone_prefix = "encoder."
    inference_input_multiple = 1
    inference_internal_padding = {"multiple": 14, "sides": ["bottom", "right"],
                                  "output_fov": "original_image"}
    deformable_attention = "not_used"

    def __init__(self, encoder, pixel_decoder, transformer_decoder, settings):
        super().__init__()
        self.encoder = encoder
        self.pixel_decoder = pixel_decoder
        self.transformer_decoder = transformer_decoder
        self.freeze_backbone = bool(settings.get("freeze_backbone", False))
        self.encoder.requires_grad_(not self.freeze_backbone)
        self.decoder_gradient_checkpointing = bool(settings.get("gradient_checkpointing", False))

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.encoder.eval()
        return self

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected normalized RGB images with shape [B,3,H,W]")
        height, width = images.shape[-2:]
        if min(height, width) < 1:
            raise ValueError("DA3 needs nonempty images")
        images = F.pad(images, (0, (-width) % 14, 0, (-height) % 14))
        padded_size = images.shape[-2:]
        # Each batch item is an independent single-view scene. The native method
        # preserves local/global concatenation, global-half norm and camera token.
        with torch.no_grad() if self.freeze_backbone else nullcontext():
            layers, _ = self.encoder.get_intermediate_layers(images.unsqueeze(1), n=OUT_LAYERS)
        ph, pw = padded_size[0] // 14, padded_size[1] // 14
        features = [tokens[:, 0].transpose(1, 2).reshape(images.shape[0], 2048, ph, pw)
                    for tokens, _ in layers]
        mask_features, pyramid = self.pixel_decoder(features)
        if self.decoder_gradient_checkpointing and torch.is_grad_enabled():
            outputs = checkpoint(self.transformer_decoder, pyramid, mask_features, None,
                                 use_reentrant=False, preserve_rng_state=True)
        else:
            outputs = self.transformer_decoder(pyramid, mask_features, None)
        for result in [outputs, *outputs.get("aux_outputs", [])]:
            result["pred_masks"] = crop_mask_field(result["pred_masks"], (height, width), padded_size)
        return outputs


def build_da3_dpt_mask2former(config, load_backbone=True):
    settings = config.get("model", {})
    if settings.get("model_family", "da3-large-1.1") != "da3-large-1.1":
        raise ValueError("da3_dpt_mask2former requires the native DA3-LARGE-1.1 encoder")
    if int(settings.get("unfreeze_last_blocks", 0)):
        raise ValueError("Independent DA3 supports freeze_backbone or full encoder training")
    dimension = int(settings.get("hidden_dim", 256))
    heads = int(settings.get("decoder_heads", 8))
    layers = int(settings.get("decoder_layers", 6))
    if dimension % 4 or dimension % heads or layers < 1:
        raise ValueError("hidden_dim must be divisible by 4 and decoder_heads; decoder_layers must be positive")
    encoder = build_native_encoder(settings, load_weights=load_backbone)
    source = settings.get("dinov3_repo", Path(__file__).resolve().parents[2] / "third_party/dinov3")
    _, _, _, decoder_module, _ = _official_modules(source)
    pixel_decoder = DA3DPTPixelDecoder(dimension, bool(settings.get("gradient_checkpointing", False)))
    decoder = decoder_module.MultiScaleMaskedTransformerDecoder(
        in_channels=dimension, mask_classification=True,
        num_classes=int(settings.get("num_classes", 5)), hidden_dim=dimension,
        num_queries=int(settings.get("num_queries", 100)), nheads=heads,
        dim_feedforward=int(settings.get("dim_feedforward", 2048)), dec_layers=layers,
        pre_norm=False, mask_dim=dimension, enforce_input_project=False,
    )
    return DA3DPTMask2Former(encoder, pixel_decoder, decoder, settings)
