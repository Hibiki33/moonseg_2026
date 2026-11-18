"""Pixel cross entropy with the reduction used in NVlabs SegFormer."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SemanticCrossEntropy(nn.Module):
    # Compatibility with the evaluator's optional instance-mask synchronization.
    sync_num_masks = False

    def __init__(self, num_classes=5, ignore_index=255, loss_weight=1.0):
        super().__init__()
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.loss_weight = float(loss_weight)

    def forward(self, outputs, targets):
        logits = outputs['semantic_logits'].float()
        if targets.ndim != 3 or logits.ndim != 4 or logits.shape[0] != targets.shape[0]:
            raise ValueError('Semantic CE expects logits BCHW and targets BHW')
        if logits.shape[1] != self.num_classes:
            raise ValueError(f'Expected {self.num_classes} semantic classes, got {logits.shape[1]}')
        logits = F.interpolate(logits, size=targets.shape[-2:], mode='bilinear', align_corners=False)
        # NVlabs' cross_entropy_loss.py computes reduction='none', followed by
        # mean over *all* pixels. Ignored positions contribute zero to that mean.
        loss_ce = F.cross_entropy(logits, targets.long(), ignore_index=self.ignore_index,
                                  reduction='none').mean()
        return {'loss': loss_ce * self.loss_weight, 'loss_ce': loss_ce}
