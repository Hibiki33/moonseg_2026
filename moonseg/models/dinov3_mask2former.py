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
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from torch.autograd.function import once_differentiable
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

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
            # Frozen leading blocks record no graph even without no_grad: their inputs need no grad.
            context = nullcontext() if self.backbone_trainable else torch.no_grad()
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
            # The original ViT always stays in eval mode, also when (partly) fine tuned. The hub ViT has
            # pos_embed_rope_rescale_coords=2: a pretraining-time random RoPE coordinate rescale that runs
            # only in training mode and must not run during segmentation fine tuning. The ViT has no
            # BatchNorm, so eval mode affects only that jitter and drop path (rate 0 in the hub configs).
            self.backbone.eval()
            return self

    depth = len(backbone.blocks)
    # Upstream FOUR_EVEN_INTERVALS (dinov3/eval/segmentation/models/__init__.py): the paper's ViT-L set,
    # otherwise the last block of each quarter (depth 12 -> [2, 5, 8, 11], depth 32 -> [7, 15, 23, 31]).
    default_indices = [4, 11, 17, 23] if depth == 24 else [i * (depth // 4) - 1 for i in range(1, 5)]
    indices = settings.get("interaction_indexes", default_indices)
    if len(indices) != 4 or sorted(set(indices)) != list(indices) or indices[-1] >= depth or indices[0] < 0:
        raise ValueError("ViT-Adapter requires four increasing valid backbone interaction indexes")
    adapter = TrainableDINOv3Adapter(
        backbone, interaction_indexes=indices,
        conv_inplane=int(settings.get("adapter_conv_dim", 64)),
        # 64 channels per deformable head as for B (12) and L (16); H+ 1280 -> 20, a power-of-2 head
        # size that upstream MSDeformAttn prefers (16 heads would give 80 and warn).
        deform_num_heads=int(settings.get("adapter_heads", backbone.embed_dim // 64)),
        drop_path_rate=float(settings.get("adapter_drop_path", 0.1)),
        with_cp=bool(settings.get("gradient_checkpointing", True)),
    )
    # DA3 keeps its native patch14 encoder while bridging only extracted features to /16.
    adapter.patch_size = getattr(backbone, "adapter_patch_size", backbone.patch_size)
    adapter.freeze_backbone = bool(settings.get("freeze_backbone", True))
    unfreeze = settings.get("unfreeze_last_blocks", 0)
    if isinstance(unfreeze, bool) or not isinstance(unfreeze, int) or not 0 <= unfreeze <= depth:
        raise ValueError(f"model.unfreeze_last_blocks must be an integer in [0, {depth}]")
    if unfreeze and not adapter.freeze_backbone:
        raise ValueError("model.unfreeze_last_blocks needs freeze_backbone=true; freeze_backbone=false trains the whole ViT")
    if unfreeze and indices[-1] != depth - 1:
        raise ValueError("model.unfreeze_last_blocks needs the final ViT block as the last interaction index; "
                         "later blocks would receive no gradient")
    adapter.backbone.requires_grad_(not adapter.freeze_backbone)
    if unfreeze:
        # Only the last blocks and the final norm applied to every extracted layer. Patch embedding,
        # class/storage/mask tokens stay frozen; RoPE periods are a buffer, never trained.
        for module in (*adapter.backbone.blocks[-unfreeze:], adapter.backbone.norm):
            module.requires_grad_(True)
    adapter.backbone_trainable = any(parameter.requires_grad for parameter in adapter.backbone.parameters())
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
    model = DINOv3Mask2Former(adapter, pixel_decoder, decoder)
    model.deformable_attention = deformable_attention  # "cuda" or "torch"; plain attribute, not in state_dict
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"MoonSeg deformable attention: {deformable_attention} ({attention_detail})", flush=True)
    return model
