"""DAFormer-style Rare Class Sampling (Hoyer et al., CVPR 2022) as a distributed sampler.

Class c is drawn with P(c) proportional to exp((1 - f_c) / T), where f_c is class c's share
of all training label pixels; then one image with at least min_pixels pixels of c is drawn
uniformly. Draws are with replacement. Every rank builds the same epoch list from
(seed, epoch) and takes indices[rank::world_size]; the epoch length equals that of
DistributedSampler, so the training batch plan is unchanged.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from torch.utils.data import Sampler

from .labels import CLASS_NAMES


def load_label_stats(path, records, num_classes):
    """Return per-record class pixel counts [N,C] and label pixel totals [N] from a label_stats JSON."""
    if not path:
        raise ValueError("train.sampling.mode=rare_class requires data.label_stats")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("kind") != "label_stats" or not isinstance(value.get("samples"), dict):
        raise ValueError(f"{path} must be a JSON object with kind=label_stats and a samples mapping")
    if value.get("num_classes") != num_classes:
        raise ValueError(f"label_stats num_classes={value.get('num_classes')} differs from model.num_classes={num_classes}")
    pixels = np.zeros((len(records), num_classes), dtype=np.int64)
    totals = np.zeros(len(records), dtype=np.int64)
    missing = []
    for index, record in enumerate(records):
        entry = value["samples"].get(str(record["sample_id"]))
        if entry is None:
            missing.append(str(record["sample_id"]))
            continue
        counts, total = (entry.get("class_pixels"), entry.get("total")) if isinstance(entry, dict) else (None, None)
        if (not isinstance(counts, list) or len(counts) != num_classes or not isinstance(total, int)
                or any(not isinstance(count, int) or count < 0 for count in counts) or sum(counts) > total):
            raise ValueError(f"Invalid label_stats entry for sample {record['sample_id']}: {entry}")
        pixels[index], totals[index] = counts, total
    if missing:
        raise ValueError(f"label_stats {path} lacks {len(missing)} training samples; first: {missing[:8]}")
    return pixels, totals


class RareClassSampler(Sampler[int]):
    def __init__(self, pixels, probabilities, min_pixels, num_replicas, rank, seed=0, drop_last=True):
        if not 0 <= rank < num_replicas:
            raise ValueError("Sampler rank must be in [0, num_replicas)")
        self.probabilities = np.asarray(probabilities, dtype=np.float64)
        self.eligible = [np.flatnonzero(pixels[:, c] >= min_pixels) for c in range(pixels.shape[1])]
        empty = [CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c)
                 for c, members in enumerate(self.eligible) if len(members) == 0]
        if empty:
            raise ValueError(f"Rare class sampling: no training image has >= {min_pixels} pixels of {empty}")
        self.size, self.num_replicas, self.rank, self.seed = len(pixels), num_replicas, rank, int(seed)
        # Same per-rank length and tail policy as DistributedSampler(drop_last=...).
        self.num_samples = self.size // num_replicas if drop_last else math.ceil(self.size / num_replicas)
        self.total_size = self.num_samples * num_replicas
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def epoch_indices(self, epoch):
        """Identical on every rank: len(dataset) draws (more only to pad when drop_last is false)."""
        generator = np.random.default_rng([self.seed, int(epoch)])
        draws = max(self.size, self.total_size)
        classes = generator.choice(len(self.probabilities), size=draws, p=self.probabilities)
        indices = np.empty(draws, dtype=np.int64)
        for c, members in enumerate(self.eligible):
            chosen = np.flatnonzero(classes == c)
            indices[chosen] = members[generator.integers(len(members), size=len(chosen))]
        return indices[:self.total_size]

    def __iter__(self):
        return iter(self.epoch_indices(self.epoch)[self.rank::self.num_replicas].tolist())

    def __len__(self):
        return self.num_samples


def build_rare_class_sampler(records, stats_path, sampling, num_classes, num_replicas, rank, seed, drop_last):
    """Return (sampler, summary) for train.sampling={mode: rare_class, temperature, min_pixels}."""
    temperature, min_pixels = sampling.get("temperature", 0.01), sampling.get("min_pixels", 1)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not temperature > 0:
        raise ValueError("train.sampling.temperature must be a positive number")
    if isinstance(min_pixels, bool) or not isinstance(min_pixels, int) or min_pixels < 1:
        raise ValueError("train.sampling.min_pixels must be a positive integer")
    pixels, totals = load_label_stats(stats_path, records, num_classes)
    frequency = pixels.sum(0) / max(int(totals.sum()), 1)
    logits = (1.0 - frequency) / float(temperature)
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    sampler = RareClassSampler(pixels, probabilities, min_pixels, num_replicas, rank, seed, drop_last)
    # Expected draw probability of each image, then of drawing an image that contains class c.
    image_probability = np.zeros(len(records))
    for c, members in enumerate(sampler.eligible):
        image_probability[members] += probabilities[c] / len(members)
    present = pixels > 0
    names = [CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c) for c in range(num_classes)]
    as_dict = lambda values: {name: float(f"{value:.6g}") for name, value in zip(names, values)}  # noqa: E731
    summary = {
        "mode": "rare_class", "temperature": float(temperature), "min_pixels": min_pixels,
        "label_stats": str(stats_path), "class_pixel_frequency": as_dict(frequency),
        "class_sampling_probability": as_dict(probabilities),
        "eligible_images": {name: int(len(members)) for name, members in zip(names, sampler.eligible)},
        "expected_image_frequency": as_dict(image_probability @ present),
        "uniform_image_frequency": as_dict(present.mean(0)),
        "expected_pixel_frequency": as_dict((image_probability @ pixels) / max(float(image_probability @ totals), 1.0)),
        "max_image_probability_ratio_to_uniform": float(image_probability.max() * len(records)),
    }
    return sampler, summary
