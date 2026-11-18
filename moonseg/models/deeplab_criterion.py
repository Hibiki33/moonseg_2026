"""Unweighted pixel CE plus the official SMP multiclass Dice loss."""
from __future__ import annotations

from torch import nn
from torch.nn import functional as F

from .deeplabv3plus import _official_smp


class DeepLabCrossEntropyDice(nn.Module):
    sync_num_masks = False

    def __init__(self, num_classes=5, ignore_index=255, ce_weight=1.0, dice_weight=1.0, smp_repo=None):
        super().__init__()
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        # The competition discloses CE+Dice without class weights, but no loss
        # coefficients. Equal coefficients are an explicit local default.
        self.dice = _official_smp(smp_repo).losses.DiceLoss(
            mode='multiclass', from_logits=True, ignore_index=self.ignore_index)

    def forward(self, outputs, targets):
        logits = outputs['semantic_logits'].float()
        if targets.ndim != 3 or logits.ndim != 4 or logits.shape[0] != targets.shape[0]:
            raise ValueError('DeepLab CE+Dice expects logits BCHW and targets BHW')
        if logits.shape[1] != self.num_classes:
            raise ValueError(f'Expected {self.num_classes} semantic classes, got {logits.shape[1]}')
        logits = F.interpolate(logits, size=targets.shape[-2:], mode='bilinear', align_corners=False)
        targets = targets.long()
        per_pixel = F.cross_entropy(logits, targets, ignore_index=self.ignore_index, reduction='none')
        # Standard unweighted CE mean over valid pixels, including a finite zero
        # for an entirely ignored batch (ordinary torch CE mean returns NaN).
        loss_ce = per_pixel.sum() / targets.ne(self.ignore_index).sum().clamp_min(1)
        loss_dice = self.dice(logits, targets)
        loss = self.ce_weight * loss_ce + self.dice_weight * loss_dice
        return {'loss': loss, 'loss_ce': loss_ce, 'loss_dice': loss_dice}
