"""Pixel-confusion metrics using the official ground-truth-present class set."""
from __future__ import annotations

import torch
import torch.distributed as dist

from .data.labels import CLASS_NAMES


class SegmentationMetrics:
    def __init__(self, num_classes=5, device="cpu"):
        self.num_classes = num_classes
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

    @torch.no_grad()
    def update(self, prediction, target):
        prediction = prediction.to(self.confusion.device).reshape(-1).long()
        target = target.to(self.confusion.device).reshape(-1).long()
        valid = (target >= 0) & (target < self.num_classes)
        if torch.any((prediction[valid] < 0) | (prediction[valid] >= self.num_classes)):
            raise ValueError("Prediction contains invalid class IDs")
        self.confusion += torch.bincount(target[valid] * self.num_classes + prediction[valid],
                                         minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)

    def synchronize(self):
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self.confusion)

    def compute(self):
        matrix = self.confusion.double()
        intersection = matrix.diag()
        support = matrix.sum(1)
        union = matrix.sum(0) + support - intersection
        # Official C_s includes only classes present in this scene's ground truth.
        # Predicting an absent class still contributes FN to the actual GT class.
        valid = support > 0
        ious = torch.where(valid, intersection / union.clamp_min(1), torch.nan)
        recalls = torch.where(valid, intersection / support.clamp_min(1), torch.nan)
        mean = ious[valid].mean().item() if valid.any() else None
        class_recall = {name: (recalls[index].item() if valid[index] else None)
                        for index, name in enumerate(CLASS_NAMES[:self.num_classes])}
        return {"mIoU": mean,
                "mAcc": recalls[valid].mean().item() if valid.any() else None,
                "class_IoU": {name: (ious[index].item() if valid[index] else None)
                              for index, name in enumerate(CLASS_NAMES[:self.num_classes])},
                "class_accuracy": class_recall.copy(),
                "class_recall": class_recall,
                "class_support": {name: int(support[index].item())
                                  for index, name in enumerate(CLASS_NAMES[:self.num_classes])},
                "pixel_accuracy": (intersection.sum() / matrix.sum()).item() if matrix.sum() else None,
                "confusion_matrix": self.confusion.cpu().tolist(),
                "absent_class_policy": "Average only classes present in ground truth; sky is included when present.",
                "rules_source": "MoonSeg AI Challenge 2026 evaluation rules",
                "metric_scope": "Confusion counts pooled over the evaluated samples; use each scene separately for official scene mIoU. Official weighted total requires organizer weights."}
