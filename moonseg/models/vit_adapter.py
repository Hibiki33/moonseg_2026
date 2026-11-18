"""Trainable wrapper around the official DINOv3 ViT-Adapter.

The interaction indexes, deformable inputs and feature pyramid match the official
adapter. This file does not assemble Mask2Former or load backbone weights.
"""
from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F


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


def build_adapter(adapter_module, backbone, settings):
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
