"""Official DINOv3 ViT-Adapter and Mask2Former, without custom CUDA ops."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def _official_modules(source: str | Path):
    source = Path(source).expanduser().resolve()
    if not (source / "dinov3/eval/segmentation/models/backbone/dinov3_adapter.py").is_file():
        raise FileNotFoundError(
            f"Official DINOv3 source is missing at {source}. Run the source setup command "
            "in README, or set model.dinov3_repo to a local facebookresearch/dinov3 checkout."
        )
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    dinov3 = importlib.import_module("dinov3")
    if Path(dinov3.__file__).resolve().parent != source / "dinov3":
        raise RuntimeError("Another dinov3 installation is already imported; use a fresh Python process")
    # The upstream models/__init__.py imports the full evaluation/data stack.
    # Register its directories as namespace packages, then load the exact official
    # implementation files with their normal absolute imports. No vendor edits.
    for name in ("dinov3.eval", "dinov3.eval.segmentation", "dinov3.eval.segmentation.models",
                 "dinov3.eval.segmentation.models.backbone", "dinov3.eval.segmentation.models.heads",
                 "dinov3.eval.segmentation.models.utils"):
        if name not in sys.modules:
            directory = source.joinpath(*name.split("."))
            package = types.ModuleType(name)
            package.__path__ = [str(directory)]
            package.__package__ = name
            package.__spec__ = importlib.util.spec_from_loader(name, loader=None, is_package=True)
            sys.modules[name] = package
    prefix = "dinov3.eval.segmentation.models"
    ops = importlib.import_module(prefix + ".utils.ms_deform_attn")
    adapter = importlib.import_module(prefix + ".backbone.dinov3_adapter")
    pixel = importlib.import_module(prefix + ".heads.pixel_decoder")
    decoder = importlib.import_module(prefix + ".heads.mask2former_transformer_decoder")
    backbones = importlib.import_module("dinov3.hub.backbones")
    return backbones, adapter, pixel, decoder, ops


def _enable_torch_deformable_attention(ops):
    """Let autograd differentiate the official grid_sample implementation.

    Upstream's custom Function uses PyTorch forward but a compiled CUDA backward.
    This explicit replacement removes that Function boundary for *both* passes.
    The replacement lives in memory, and never modifies the official checkout.
    """
    class TorchDeformableAttention:
        @staticmethod
        def apply(value, spatial_shapes, level_start_index, locations, weights, im2col_step):
            del level_start_index, im2col_step
            with torch.autocast(device_type=value.device.type, enabled=False):
                # grid_sample supports FP32 on both CPU and CUDA; BF16 is not
                # supported by all installations. Casts retain their gradients.
                result = ops.ms_deform_attn_core_pytorch(
                    value.float(), spatial_shapes, locations.float(), weights.float()
                )
            return result.to(value.dtype)

    ops.MSDeformAttnFunction = TorchDeformableAttention


def _batch_norm(module):
    """Keep upstream BatchNorm semantics while supporting CPU and one GPU."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.SyncBatchNorm):
            replacement = nn.BatchNorm2d(child.num_features, eps=child.eps, momentum=child.momentum,
                                         affine=child.affine, track_running_stats=child.track_running_stats)
            replacement.load_state_dict(child.state_dict())
            replacement.train(child.training)
            setattr(module, name, replacement)
        else:
            _batch_norm(child)


def _build_adapter(adapter_module, backbone, settings):
    class TrainableDINOv3Adapter(adapter_module.DINOv3_Adapter):
        # Same official components and four extraction interactions. This forward
        # removes CUDA-only autocast and enables the optional backbone fine tune.
        def forward(self, images):
            deform1, deform2 = adapter_module.deform_inputs(images, self.patch_size)
            c1, c2, c3, c4 = self.spm(images)
            c2, c3, c4 = self._add_level_embed(c2, c3, c4)
            length2, length3 = c2.shape[1], c3.shape[1]
            spatial = torch.cat((c2, c3, c4), dim=1)
            context = torch.no_grad() if self.freeze_backbone else nullcontext()
            with context:
                layers = self.backbone.get_intermediate_layers(
                    images, n=self.interaction_indexes, return_class_token=True
                )
            batch = images.shape[0]
            height, width = images.shape[-2] // 16, images.shape[-1] // 16
            token_height = images.shape[-2] // self.patch_size
            token_width = images.shape[-1] // self.patch_size
            outputs = []
            for interaction, (tokens, cls) in zip(self.interactions, layers):
                _, spatial, _ = interaction(tokens, spatial, cls, deform1, deform2,
                                              height, width, token_height, token_width)
                outputs.append(tokens.transpose(1, 2).reshape(batch, -1, token_height, token_width))
            c2 = spatial[:, :length2].transpose(1, 2).reshape(batch, -1, height * 2, width * 2)
            c3 = spatial[:, length2:length2 + length3].transpose(1, 2).reshape(batch, -1, height, width)
            c4 = spatial[:, length2 + length3:].transpose(1, 2).reshape(batch, -1, height // 2, width // 2)
            c1 = self.up(c2) + c1
            pyramid = [c1, c2, c3, c4]
            norms = [self.norm1, self.norm2, self.norm3, self.norm4]
            if self.add_vit_feature:
                pyramid = [feature + F.interpolate(tokens, size=feature.shape[-2:], mode="bilinear", align_corners=False)
                           for feature, tokens in zip(pyramid, outputs)]
            return {str(index + 1): norm(feature) for index, (norm, feature) in enumerate(zip(norms, pyramid))}

        def train(self, mode=True):
            super().train(mode)
            if self.freeze_backbone:
                self.backbone.eval()
            return self

    depth = len(backbone.blocks)
    indices = settings.get("interaction_indexes", [4, 11, 17, 23] if depth == 24 else [2, 5, 8, 11])
    if len(indices) != 4 or sorted(set(indices)) != list(indices) or indices[-1] >= depth or indices[0] < 0:
        raise ValueError("ViT-Adapter requires four increasing valid backbone interaction indexes")
    adapter = TrainableDINOv3Adapter(
        backbone, interaction_indexes=indices,
        conv_inplane=int(settings.get("adapter_conv_dim", 64)),
        deform_num_heads=int(settings.get("adapter_heads", 12 if backbone.embed_dim == 768 else 16)),
        drop_path_rate=float(settings.get("adapter_drop_path", 0.1)),
        with_cp=bool(settings.get("gradient_checkpointing", True)),
    )
    adapter.freeze_backbone = bool(settings.get("freeze_backbone", True))
    adapter.backbone.requires_grad_(not adapter.freeze_backbone)
    if not settings.get("adapter_sync_batchnorm", False):
        _batch_norm(adapter.spm)
        for index in range(1, 5):
            norm = getattr(adapter, f"norm{index}")
            replacement = nn.BatchNorm2d(norm.num_features, eps=norm.eps, momentum=norm.momentum)
            replacement.load_state_dict(norm.state_dict())
            setattr(adapter, f"norm{index}", replacement)
    adapter.train()
    return adapter


def _load_backbone(backbone, path):
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"DINOv3 backbone weights not found: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("state_dict", "model", "teacher", "backbone"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
    if not isinstance(state, dict):
        raise ValueError("Expected a DINOv3 PyTorch state dictionary")
    for prefix in ("module.", "backbone."):
        if state and all(key.startswith(prefix) for key in state):
            state = {key[len(prefix):]: value for key, value in state.items()}
    backbone.load_state_dict(state, strict=True)


class DINOv3Mask2Former(nn.Module):
    def __init__(self, backbone, pixel_decoder, transformer_decoder):
        super().__init__()
        self.backbone = backbone
        self.pixel_decoder = pixel_decoder
        self.transformer_decoder = transformer_decoder

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected normalized RGB images with shape [B,3,H,W]")
        height, width = images.shape[-2:]
        # Adapter's convolutional pyramid requires spatial dimensions /32.
        padded_height, padded_width = (height + 31) // 32 * 32, (width + 31) // 32 * 32
        if min(padded_height, padded_width) < 64:
            raise ValueError("ViT-Adapter needs image dimensions of at least 64 pixels")
        images = F.pad(images, (0, padded_width - width, 0, padded_height - height))
        features = self.backbone(images)
        mask_features, _, pyramid = self.pixel_decoder.forward_features(features)
        outputs = self.transformer_decoder(pyramid, mask_features, None)
        if (height, width) != (padded_height, padded_width):
            # Return logits in the original field of view. Full resolution avoids
            # fractional /4 crop boundaries for arbitrary inference dimensions.
            for result in [outputs, *outputs.get("aux_outputs", [])]:
                result["pred_masks"] = F.interpolate(result["pred_masks"], size=(padded_height, padded_width),
                                                     mode="bilinear", align_corners=False)[..., :height, :width]
        return outputs


def build_model(config: dict, load_backbone: bool = True) -> DINOv3Mask2Former:
    settings = config.get("model", {})
    architecture = str(settings.get("backbone", "vitb16")).removeprefix("dinov3_")
    if architecture not in ("vitb16", "vitl16"):
        raise ValueError("model.backbone must be vitb16 or vitl16")
    source = settings.get("dinov3_repo", Path(__file__).resolve().parents[2] / "third_party/dinov3")
    backbones, adapter_module, pixel_module, decoder_module, ops = _official_modules(source)
    _enable_torch_deformable_attention(ops)
    backbone = getattr(backbones, "dinov3_" + architecture)(pretrained=False)
    if load_backbone:
        weights = settings.get("backbone_weights")
        if not weights:
            raise ValueError("Set model.backbone_weights to official local DINOv3 weights; no automatic download is performed")
        _load_backbone(backbone, weights)
    adapter = _build_adapter(adapter_module, backbone, settings)
    dimension = int(settings.get("hidden_dim", 256))
    heads = int(settings.get("decoder_heads", 8))
    if dimension % 32 or dimension % heads:
        raise ValueError("model.hidden_dim must be divisible by 32 and decoder_heads")
    encoder_layers = int(settings.get("encoder_layers", 6))
    decoder_layers = int(settings.get("decoder_layers", 6))
    if encoder_layers < 1 or decoder_layers < 1:
        raise ValueError("Mask2Former requires at least one encoder and decoder layer")
    pixel_decoder = pixel_module.MSDeformAttnPixelDecoder(
        input_shape={str(index + 1): [backbone.embed_dim, 0, 0, stride] for index, stride in enumerate((4, 8, 16, 32))},
        transformer_dropout=0.0, transformer_nheads=heads,
        transformer_dim_feedforward=int(settings.get("dim_feedforward", 2048)),
        transformer_enc_layers=encoder_layers, conv_dim=dimension, mask_dim=dimension,
        norm="GN", transformer_in_features=["1", "2", "3", "4"], common_stride=4,
    )
    decoder = decoder_module.MultiScaleMaskedTransformerDecoder(
        in_channels=dimension, mask_classification=True, num_classes=int(settings.get("num_classes", 5)),
        hidden_dim=dimension, num_queries=int(settings.get("num_queries", 100)), nheads=heads,
        dim_feedforward=int(settings.get("dim_feedforward", 2048)), dec_layers=decoder_layers,
        pre_norm=False, mask_dim=dimension, enforce_input_project=False,
    )
    return DINOv3Mask2Former(adapter, pixel_decoder, decoder)
