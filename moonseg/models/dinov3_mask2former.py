"""Official DINOv3 ViT-Adapter and Mask2Former.

Deformable attention uses the official compiled MultiScaleDeformableAttention CUDA extension when it
is available (``model.deformable_attention: auto|cuda``), otherwise the pure-PyTorch grid_sample path.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import math
import os
import sys
import types
import warnings
from pathlib import Path

import torch
from torch import nn
from torch.autograd.function import once_differentiable
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .vit_adapter import build_adapter

DEFORMABLE_ATTENTION_MODES = ("auto", "cuda", "torch")
# Official hub ViTs (dinov3/hub/backbones.py): B 768x12, L 1024x24, H+ 1280x32 with a SwiGLU MLP.
BACKBONES = ("vitb16", "vitl16", "vith16plus")
# Built by scripts/build_ops.sh; override with model.deformable_attention_extension_dir or MOONSEG_MSDA_DIR.
DEFAULT_EXTENSION_DIR = Path(__file__).resolve().parents[2] / "build" / "ms_deform_attn"


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
    return TorchDeformableAttention


def _load_deformable_attention_extension(directory=None):
    """Import the compiled official extension; return (module, None) or (None, reason)."""
    directory = Path(directory or os.environ.get("MOONSEG_MSDA_DIR") or DEFAULT_EXTENSION_DIR).expanduser().resolve()
    manifest = directory / "build_manifest.json"
    if manifest.is_file():
        built = json.loads(manifest.read_text(encoding="utf-8"))
        if built.get("torch") != torch.__version__:
            return None, (f"{directory} was built for torch {built.get('torch')}, running {torch.__version__}; "
                          "rerun scripts/build_ops.sh")
    if directory.is_dir() and str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    try:
        import MultiScaleDeformableAttention as extension  # noqa: N813
    except (ImportError, OSError) as error:  # missing, or built for another torch/CUDA/Python ABI
        return None, f"{type(error).__name__}: {error}"
    return extension, None


def _enable_cuda_deformable_attention(ops, extension):
    """Official compiled forward and backward (ops/functions/ms_deform_attn_func.py semantics).

    The kernels dispatch float/double only, so inputs are cast to FP32 exactly like the PyTorch
    path; BF16 autocast outside this call is unaffected. CPU tensors use the PyTorch path.
    """
    torch_path = _enable_torch_deformable_attention(ops)

    class CompiledFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value, spatial_shapes, level_start_index, locations, weights, im2col_step):
            ctx.im2col_step = im2col_step
            output = extension.ms_deform_attn_forward(value, spatial_shapes, level_start_index,
                                                      locations, weights, im2col_step)
            ctx.save_for_backward(value, spatial_shapes, level_start_index, locations, weights)
            return output

        @staticmethod
        @once_differentiable
        def backward(ctx, grad_output):
            value, spatial_shapes, level_start_index, locations, weights = ctx.saved_tensors
            grad_value, grad_locations, grad_weights = extension.ms_deform_attn_backward(
                value, spatial_shapes, level_start_index, locations, weights,
                grad_output.contiguous(), ctx.im2col_step)
            return grad_value, None, None, grad_locations, grad_weights, None

    class CudaDeformableAttention:
        @staticmethod
        def apply(value, spatial_shapes, level_start_index, locations, weights, im2col_step):
            if not value.is_cuda:
                return torch_path.apply(value, spatial_shapes, level_start_index, locations, weights, im2col_step)
            # The kernel requires batch % min(batch, im2col_step) == 0.
            step = math.gcd(value.shape[0], int(im2col_step))
            with torch.autocast(device_type="cuda", enabled=False):
                result = CompiledFunction.apply(
                    value.float().contiguous(), spatial_shapes.contiguous(), level_start_index.contiguous(),
                    locations.float().contiguous(), weights.float().contiguous(), step)
            return result.to(value.dtype)

    ops.MSDeformAttnFunction = CudaDeformableAttention
    return CudaDeformableAttention


def _select_deformable_attention(ops, settings):
    # Process-wide: this patches the official module, so the last built model's mode applies to all.
    mode = str(settings.get("deformable_attention", "auto")).lower()
    if mode not in DEFORMABLE_ATTENTION_MODES:
        raise ValueError("model.deformable_attention must be auto, cuda or torch")
    if mode == "torch":
        _enable_torch_deformable_attention(ops)
        return "torch", "configured PyTorch grid_sample path"
    extension, reason = _load_deformable_attention_extension(settings.get("deformable_attention_extension_dir"))
    if extension is None:
        if mode == "cuda":
            raise RuntimeError("model.deformable_attention=cuda but the compiled MultiScaleDeformableAttention "
                               f"extension is unavailable ({reason}). Run scripts/build_ops.sh.")
        warnings.warn(f"Compiled MultiScaleDeformableAttention unavailable ({reason}); "
                      "using the slower, more memory-hungry PyTorch deformable attention.", stacklevel=2)
        _enable_torch_deformable_attention(ops)
        return "torch", "PyTorch grid_sample fallback"
    _enable_cuda_deformable_attention(ops, extension)
    return "cuda", f"compiled {extension.__file__}; CPU tensors use PyTorch grid_sample"


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


def _checkpoint_dino_blocks(backbone):
    """Recompute native ViT blocks without changing parameters, keys or eval behavior."""
    for block in backbone.blocks:
        if getattr(block, "_moonseg_checkpointed", False):
            continue
        original_forward = block.forward
        def forward(self, *args, _original=original_forward, **kwargs):
            # The encoder stays in eval mode during fine tuning; gradient mode,
            # rather than self.training, determines whether recomputation helps.
            if torch.is_grad_enabled() and any(p.requires_grad for p in self.parameters()):
                return activation_checkpoint(_original, *args, use_reentrant=False,
                                             preserve_rng_state=True, **kwargs)
            return _original(*args, **kwargs)
        block.forward = types.MethodType(forward, block)
        block._moonseg_checkpointed = True


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
    if architecture not in (*BACKBONES, "da3mono_vitl14"):
        raise ValueError(f"model.backbone must be one of {', '.join(BACKBONES)}")
    source = settings.get("dinov3_repo", Path(__file__).resolve().parents[2] / "third_party/dinov3")
    backbones, adapter_module, pixel_module, decoder_module, ops = _official_modules(source)
    deformable_attention, attention_detail = _select_deformable_attention(ops, settings)
    if architecture == "da3mono_vitl14":
        from .da3mono import build_da3mono
        backbone = build_da3mono(settings, load_weights=load_backbone)
    else:
        backbone = getattr(backbones, "dinov3_" + architecture)(pretrained=False)
    if load_backbone and architecture != "da3mono_vitl14":
        weights = settings.get("backbone_weights")
        if not weights:
            raise ValueError("Set model.backbone_weights to official local DINOv3 weights; no automatic download is performed")
        _load_backbone(backbone, weights)
    if architecture != "da3mono_vitl14" and settings.get("backbone_gradient_checkpointing", False):
        _checkpoint_dino_blocks(backbone)
    adapter = build_adapter(adapter_module, backbone, settings)
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
    model = DINOv3Mask2Former(adapter, pixel_decoder, decoder)
    model.deformable_attention = deformable_attention  # "cuda" or "torch"; plain attribute, not in state_dict
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"MoonSeg deformable attention: {deformable_attention} ({attention_detail})", flush=True)
    return model
