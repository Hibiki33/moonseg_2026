"""Frozen text-only SAM 3 baseline using Meta's original image model.

The five independently prompted dense ``semantic_seg`` logits are resized to the
input coordinates, sigmoid-transformed, and returned as class scores. This is a
project adaptation, not a reproduction of an undisclosed LuSNAR SAM3 protocol.
"""
from __future__ import annotations

import importlib
from pathlib import Path
import sys
from threading import Lock

import torch
from torch import nn
from torch.nn import functional as F


PROMPTS = ("lunar regolith", "rock", "impact crater", "mountain", "sky")
_BUILD_LOCK = Lock()


def _official_modules(source):
    source = Path(source).resolve()
    if not (source / "sam3/model_builder.py").is_file():
        raise FileNotFoundError(f"Official facebookresearch/sam3 source missing at {source}")
    existing = sys.modules.get("sam3")
    if existing is not None and Path(existing.__file__).resolve().parent != source / "sam3":
        raise RuntimeError("Another SAM3 source is imported; use a fresh Python process")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    return (importlib.import_module("sam3.model_builder"),
            importlib.import_module("sam3.model.sam3_image_processor").Sam3Processor)


class SAM3Semantic(nn.Module):
    """Inference-only adapter accepting the project's normalized RGB BCHW input."""

    model_family = "sam3"
    inference_only = True
    prompts = PROMPTS
    native_resolution = 1008

    def __init__(self, model, processor_class, mean, std):
        super().__init__()
        self.model = model
        self.processor_class = processor_class
        self.register_buffer("input_mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1),
                             persistent=False)
        self.register_buffer("input_std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1),
                             persistent=False)
        if not torch.isfinite(self.input_mean).all() or not torch.isfinite(self.input_std).all():
            raise ValueError("SAM3 input normalization must be finite")
        if (self.input_std <= 0).any():
            raise ValueError("SAM3 input normalization standard deviations must be positive")
        self._text_cache = {}
        self.requires_grad_(False)
        super().train(False)

    def train(self, mode=True):
        if mode:
            raise RuntimeError("SAM3 is an inference-only baseline; training is disabled")
        return super().train(False)

    def _apply(self, fn, recurse=True):
        self._text_cache.clear()
        # Official position encodings use a plain dict, so nn.Module.to does not
        # move their cache. Recompute on the actual forward device after a move.
        for module in self.model.modules():
            if module.__class__.__name__ == "PositionEmbeddingSine":
                module.cache.clear()
            if hasattr(module, "compilable_cord_cache"):
                module.compilable_cord_cache = None
                module.compilable_stored_size = None
                module.coord_cache.clear()
            if hasattr(module, "_device"):
                module._device = None
        return super()._apply(fn, recurse=recurse)

    @torch.inference_mode()
    def forward(self, images):
        if self.training or any(module.training for module in self.model.modules()):
            raise RuntimeError("SAM3 and all of its modules must remain in eval mode")
        if images.ndim != 4 or images.shape[0] < 1 or images.shape[1] != 3:
            raise ValueError("SAM3 expects a nonempty batch of normalized RGB BCHW images")
        if not images.is_floating_point():
            raise ValueError("SAM3 expects floating point normalized images")
        if images.device != self.input_mean.device:
            raise ValueError("SAM3 model and input images must be on the same device")
        height, width = images.shape[-2:]
        # Round restores the original uint8 pixels after ImageNet normalization
        # without the one-byte truncation error of a float-to-uint8 cast.
        pixels = ((images.float() * self.input_std + self.input_mean).clamp(0, 1) * 255).round()
        pixels = pixels.to(torch.uint8)
        cache_key = (str(images.device), torch.is_autocast_enabled(images.device.type),
                     torch.get_autocast_dtype(images.device.type))
        if cache_key not in self._text_cache:
            self._text_cache[cache_key] = tuple(
                self.model.backbone.forward_text([prompt], device=images.device) for prompt in self.prompts
            )
        text_outputs = self._text_cache[cache_key]
        processor = self.processor_class(self.model, resolution=self.native_resolution, device=images.device)
        batch_scores = []
        for image in pixels:
            state = processor.set_image(image)
            class_scores = []
            for text_out in text_outputs:
                # Each prompt starts from the same image features and an empty
                # geometric prompt. No earlier class prediction is reused.
                backbone_out = dict(state["backbone_out"])
                backbone_out.update(text_out)
                outputs = self.model.forward_grounding(
                    backbone_out=backbone_out, find_input=processor.find_stage,
                    geometric_prompt=self.model._get_dummy_prompt(), find_target=None,
                )
                dense = outputs.get("semantic_seg")
                if dense is None or dense.ndim != 4 or tuple(dense.shape[:2]) != (1, 1):
                    raise RuntimeError("Official SAM3 did not return its Bx1xHxW dense semantic_seg logits")
                dense = F.interpolate(dense.float(), size=(height, width), mode="bilinear", align_corners=False)
                class_scores.append(dense.sigmoid())
            batch_scores.append(torch.cat(class_scores, dim=1))
        return {"semantic_scores": torch.cat(batch_scores, dim=0)}


def build_sam3(config, load_weights=True):
    """Build the original SAM 3 image model from local official-format weights."""
    if not load_weights:
        raise ValueError("SAM3 is inference-only and requires its pretrained local checkpoint")
    settings = config.get("model", {})
    if int(settings.get("num_classes", 5)) != len(PROMPTS):
        raise ValueError("The SAM3 baseline has exactly five fixed text classes")
    source = Path(settings.get("sam3_repo", Path(__file__).resolve().parents[2] / "third_party/sam3"))
    weights = settings.get("backbone_weights")
    if not weights or not Path(weights).expanduser().is_file():
        raise FileNotFoundError("Set model.backbone_weights to the local original SAM3 sam3.pt checkpoint")
    builder, processor_class = _official_modules(source)
    # Upstream eagerly precomputes position encodings on hardcoded CUDA even
    # when device='cpu'. Lazy evaluation uses the same official implementation,
    # avoids touching GPUs during construction, and supports a subsequent .to().
    with _BUILD_LOCK:
        original_position_builder = builder._create_position_encoding
        original_decoder = builder.TransformerDecoder
        def lazy_position_encoding(precompute_resolution=None):
            return original_position_builder(precompute_resolution=None)
        def lazy_decoder(*args, **kwargs):
            # The official decoder likewise eagerly caches coordinates on CUDA.
            # None defers the same coordinate formula to its first forward.
            return original_decoder(*args, **(kwargs | {"resolution": None, "stride": None}))
        builder._create_position_encoding = lazy_position_encoding
        builder.TransformerDecoder = lazy_decoder
        try:
            model = builder.build_sam3_image_model(
                bpe_path=str(source / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"),
                device="cpu", eval_mode=True, checkpoint_path=str(Path(weights).expanduser()),
                load_from_HF=False, enable_segmentation=True, enable_inst_interactivity=False, compile=False,
            )
        finally:
            builder._create_position_encoding = original_position_builder
            builder.TransformerDecoder = original_decoder
    # Upstream's loader is intentionally non-strict to support image/video
    # subsets. Require every active detector tensor, including its dense head.
    # The SAM2 neck is unused with enable_inst_interactivity=False.
    inventory = torch.load(Path(weights).expanduser(), map_location="meta", weights_only=True, mmap=True)
    if "model" in inventory and isinstance(inventory["model"], dict):
        inventory = inventory["model"]
    detector = {key.removeprefix("detector."): value for key, value in inventory.items()
                if key.startswith("detector.")}
    expected = model.state_dict()
    missing = sorted(set(expected) - set(detector))
    inactive_neck = {key for key in detector if key.startswith("backbone.vision_backbone.sam2_convs.")}
    unexpected = sorted(set(detector) - set(expected) - inactive_neck)
    mismatch = sorted(key for key in expected.keys() & detector.keys()
                      if expected[key].shape != detector[key].shape)
    if missing or unexpected or mismatch:
        raise ValueError(f"Original SAM3 detector checkpoint mismatch: missing={missing}, "
                         f"unexpected={unexpected}, shapes={mismatch}")
    data = config.get("data", {})
    adapted = SAM3Semantic(model, processor_class, data.get("mean", [0.485, 0.456, 0.406]),
                           data.get("std", [0.229, 0.224, 0.225]))
    adapted.checkpoint_path = str(Path(weights).expanduser().resolve())
    adapted.checkpoint_validation = {"active_detector_keys": len(expected),
                                     "unused_sam2_neck_keys": len(inactive_neck)}
    return adapted
