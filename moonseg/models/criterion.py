"""Mask2Former set prediction losses, with explicit ignored-pixel support.

Follows facebookresearch/Mask2Former's HungarianMatcher and SetCriterion.
Targets are semantic class masks (one target for each class present in an image).
Only valid pixel centers are sampled, so padding/ignore=255 cannot enter the loss.
Valid pixel indices are computed once per image; point sampling needs no further
host synchronization.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F


def _sample(logits: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    """Sample N masks at shared [P,2] or individual [N,P,2] coordinates."""
    if coordinates.ndim == 2:
        coordinates = coordinates.unsqueeze(0).expand(logits.shape[0], -1, -1)
    return F.grid_sample(
        logits.unsqueeze(1), coordinates.unsqueeze(2) * 2 - 1,
        mode="bilinear", align_corners=False,
    )[:, 0, :, 0]


def _coordinates(target: dict, shape: tuple[int, ...]) -> torch.Tensor:
    """Uniform random valid pixel centers as normalized (x, y) with shape [*shape, 2]."""
    valid = target["valid_mask"]
    height, width = valid.shape
    if target["all_valid"]:
        flat = torch.randint(height * width, shape, device=valid.device)
    else:
        indices = target["valid_indices"]
        if indices.numel() == 0:
            raise ValueError("Cannot sample points from an entirely ignored image")
        flat = indices[torch.randint(indices.numel(), shape, device=valid.device)]
    return torch.stack(((flat % width + 0.5) / width, (flat // width + 0.5) / height), -1)


def _target_points(masks: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Nearest sampling keeps semantic targets binary and excludes ignored pixels."""
    if coords.ndim == 2:
        coords = coords.unsqueeze(0).expand(masks.shape[0], -1, -1)
    return F.grid_sample(
        masks[:, None].float(), coords[:, :, None] * 2 - 1,
        mode="nearest", align_corners=False,
    )[:, 0, :, 0]


class Mask2FormerCriterion(nn.Module):
    def __init__(self, num_classes=5, ce_weight=2.0, mask_weight=5.0,
                 dice_weight=5.0, no_object_weight=0.1, num_points=12544,
                 oversample_ratio=3.0, importance_sample_ratio=0.75, ignore_index=255):
        super().__init__()
        if num_points < 1 or oversample_ratio < 1 or not 0 <= importance_sample_ratio <= 1:
            raise ValueError("Invalid Mask2Former point sampling settings")
        self.num_classes = int(num_classes)
        self.weights = {"loss_ce": ce_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}
        self.num_points = int(num_points)
        self.oversample_ratio = float(oversample_ratio)
        self.importance_sample_ratio = float(importance_sample_ratio)
        self.ignore_index = int(ignore_index)
        class_weights = torch.ones(num_classes + 1)
        class_weights[-1] = no_object_weight
        self.register_buffer("class_weights", class_weights)

    def _prepare_targets(self, targets, device):
        prepared = []
        if isinstance(targets, torch.Tensor):
            if targets.ndim != 3:
                raise ValueError("Semantic targets must have shape [B,H,W]")
            for semantic in targets.to(device):
                valid = semantic != self.ignore_index
                labels = semantic[valid].unique().long()
                masks = semantic[None] == labels[:, None, None]
                prepared.append({"labels": labels, "masks": masks.float(), "valid_mask": valid})
        else:
            for target in targets:
                masks = target["masks"].to(device).float()
                valid = target.get("valid_mask", torch.ones(masks.shape[-2:], dtype=torch.bool, device=device))
                prepared.append({"labels": target["labels"].to(device).long(), "masks": masks,
                                 "valid_mask": valid.to(device).bool()})
        for target in prepared:
            labels, masks, valid = target["labels"], target["masks"], target["valid_mask"]
            if masks.ndim != 3 or masks.shape[0] != labels.numel() or masks.shape[-2:] != valid.shape:
                raise ValueError("Target labels, masks and valid_mask have inconsistent shapes")
            if ((labels < 0) | (labels >= self.num_classes)).any():
                raise ValueError("Target labels must be in [0, num_classes)")
            # Drop targets entirely inside ignored regions, including fully ignored images.
            keep = (masks * valid[None]).flatten(1).sum(1) > 0
            target["labels"], target["masks"] = labels[keep], masks[keep]
            # Cache valid pixel indices once per image (one host sync) for all point sampling.
            flat = valid.flatten()
            target["all_valid"] = bool(flat.all())
            target["valid_indices"] = None if target["all_valid"] else flat.nonzero()[:, 0]
            target["has_valid"] = target["all_valid"] or target["valid_indices"].numel() > 0
        return prepared

    @torch.no_grad()
    def _match(self, outputs, targets):
        matches = []
        for logits, masks, target in zip(outputs["pred_logits"], outputs["pred_masks"], targets):
            n_targets = target["labels"].numel()
            if n_targets == 0:
                empty = torch.empty(0, dtype=torch.long, device=logits.device)
                matches.append((empty, empty))
                continue
            # Matching shares one set of valid points among all query/target pairs, as upstream.
            coordinates = _coordinates(target, (self.num_points,))
            predicted = _sample(masks.float(), coordinates)
            truth = _target_points(target["masks"], coordinates)
            cost_ce = -logits.float().softmax(-1)[:, target["labels"]]
            cost_mask = F.softplus(predicted).mean(1)[:, None] - predicted @ truth.T / self.num_points
            probabilities = predicted.sigmoid()
            cost_dice = 1 - (2 * probabilities @ truth.T + 1) / (
                probabilities.sum(1)[:, None] + truth.sum(1)[None] + 1
            )
            cost = self.weights["loss_ce"] * cost_ce + self.weights["loss_mask"] * cost_mask + self.weights["loss_dice"] * cost_dice
            query_indices, target_indices = linear_sum_assignment(cost.cpu().numpy())
            matches.append((torch.as_tensor(query_indices, device=logits.device),
                            torch.as_tensor(target_indices, device=logits.device)))
        return matches

    def _losses(self, outputs, targets, normalizer):
        matches = self._match(outputs, targets)
        logits, masks = outputs["pred_logits"].float(), outputs["pred_masks"]
        classes = torch.full(logits.shape[:2], self.num_classes, dtype=torch.long, device=logits.device)
        valid_images = torch.tensor([t["has_valid"] for t in targets], device=logits.device)
        # Keeps every mask logit in the graph; mean cannot overflow FP16 like a sum of [B,Q,h,w].
        zero = logits.sum() * 0 + masks.mean().float() * 0
        mask_loss, dice_loss = zero, zero
        for batch_index, (query_indices, target_indices) in enumerate(matches):
            target = targets[batch_index]
            classes[batch_index, query_indices] = target["labels"][target_indices]
            if query_indices.numel() == 0:
                continue
            # Cast only the matched queries, never the complete [B,Q,h,w] mask logits.
            predicted = masks[batch_index, query_indices].float()
            truth = target["masks"][target_indices]
            with torch.no_grad():
                # Mask2Former get_uncertain_point_coords_with_randomness: candidate and random
                # points are drawn independently for each matched mask, here from valid pixels.
                count = predicted.shape[0]
                candidate_count = max(self.num_points, int(self.num_points * self.oversample_ratio))
                candidate = _coordinates(target, (count, candidate_count))
                uncertainty = -_sample(predicted, candidate).abs()
                important_count = int(self.num_points * self.importance_sample_ratio)
                uncertain_indices = uncertainty.topk(important_count, dim=1).indices
                important = candidate.gather(1, uncertain_indices[..., None].expand(-1, -1, 2))
                random = _coordinates(target, (count, self.num_points - important_count))
                coordinates = torch.cat((important, random), dim=1)
                truth_points = _target_points(truth, coordinates)
            predicted_points = _sample(predicted, coordinates)
            mask_loss = mask_loss + F.binary_cross_entropy_with_logits(predicted_points, truth_points, reduction="none").mean(1).sum()
            probabilities = predicted_points.sigmoid()
            dice_loss = dice_loss + (1 - (2 * (probabilities * truth_points).sum(1) + 1) / (
                probabilities.sum(1) + truth_points.sum(1) + 1
            )).sum()
        if valid_images.any():
            ce = F.cross_entropy(logits[valid_images].transpose(1, 2), classes[valid_images], weight=self.class_weights)
        else:
            ce = zero
        return {"loss_ce": ce, "loss_mask": mask_loss / normalizer, "loss_dice": dice_loss / normalizer}

    def forward(self, outputs, targets):
        # Explicit FP32 prevents half-precision overflow in matching and point losses.
        with torch.autocast(device_type=outputs["pred_logits"].device.type, enabled=False):
            targets = self._prepare_targets(targets, outputs["pred_logits"].device)
            if len(targets) != outputs["pred_logits"].shape[0]:
                raise ValueError("Output and target batch sizes differ")
            count = torch.tensor(sum(t["labels"].numel() for t in targets), device=outputs["pred_logits"].device, dtype=torch.float32)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(count)
                count /= dist.get_world_size()
            normalizer = count.clamp(min=1)
            losses = self._losses(outputs, targets, normalizer)
            total = sum(self.weights[key] * value for key, value in losses.items())
            for index, auxiliary in enumerate(outputs.get("aux_outputs", [])):
                aux_losses = self._losses(auxiliary, targets, normalizer)
                total = total + sum(self.weights[key] * value for key, value in aux_losses.items())
                losses.update({f"{key}_{index}": value for key, value in aux_losses.items()})
            return {"loss": total, **{key: value.detach() for key, value in losses.items()}}


def build_criterion(config: dict) -> Mask2FormerCriterion:
    model = config.get("model", {})
    loss = dict(config.get("loss", {}))
    loss.setdefault("num_classes", model.get("num_classes", 5))
    return Mask2FormerCriterion(**loss)
