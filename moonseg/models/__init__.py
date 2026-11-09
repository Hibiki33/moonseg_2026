"""First-party models behind the shared MoonSeg training/inference interface."""

from .criterion import Mask2FormerCriterion, build_criterion as _mask2former_criterion
from .dinov3_mask2former import DINOv3Mask2Former, build_model as _adapter_model


def build_model(config, load_backbone=True):
    settings = config.get("model", {})
    architecture = settings.get("architecture", "dinov3_mask2former")
    if architecture == "dinov3_mask2former":
        return _adapter_model(config, load_backbone=load_backbone)
    if architecture == "segformer_b5":
        from .segformer import build_segformer
        return build_segformer(settings, load_weights=load_backbone)
    if architecture == "sam3":
        from .sam3 import build_sam3
        return build_sam3(config, load_weights=load_backbone)
    if architecture == "deeplabv3plus_mobilenetv3":
        from .deeplabv3plus import build_deeplabv3plus
        return build_deeplabv3plus(settings, load_weights=load_backbone)
    raise ValueError(f"Unknown model.architecture: {architecture}")


def build_criterion(config):
    architecture = config.get("model", {}).get("architecture", "dinov3_mask2former")
    if architecture == "dinov3_mask2former":
        return _mask2former_criterion(config)
    if architecture == "sam3":
        raise ValueError("SAM3 is an inference-only baseline and has no training criterion")
    loss = dict(config.get("loss", {}))
    loss.setdefault("num_classes", config.get("model", {}).get("num_classes", 5))
    kind = loss.pop("type", None)
    if architecture == "segformer_b5" and kind == "semantic_cross_entropy":
        from .semantic_criterion import SemanticCrossEntropy
        return SemanticCrossEntropy(**loss)
    if architecture == "deeplabv3plus_mobilenetv3" and kind == "semantic_ce_dice":
        from .deeplab_criterion import DeepLabCrossEntropyDice
        loss.setdefault("smp_repo", config["model"].get("smp_repo"))
        return DeepLabCrossEntropyDice(**loss)
    raise ValueError(f"Unsupported loss.type {kind!r} for {architecture}")

__all__ = ["DINOv3Mask2Former", "Mask2FormerCriterion", "build_model", "build_criterion"]
