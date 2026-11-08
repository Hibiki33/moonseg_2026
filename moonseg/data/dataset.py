from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from torch.utils.data import Dataset

from .labels import decode_label
from .manifest import TRAIN_SCENES, load_manifest, manifest_provenance

SKY_ID = 4

# Optional augmentation v2 defaults. Probabilities default to 0, so the default
# pipeline (and its random-number stream) is unchanged unless a config opts in.
AUGMENTATION_V2_DEFAULTS = {
    "scale_jitter_probability": 0.0, "scale_jitter_range": [0.75, 1.25],
    "gamma_probability": 0.0, "gamma_range": [0.7, 1.5],
    "shadow_probability": 0.0, "shadow_count_range": [1, 3], "shadow_strength_range": [0.3, 0.8],
    "shadow_blur_range": [8.0, 48.0], "shadow_size_range": [0.05, 0.35],
    "pixel_floor": 0,
}


def _probability(config, key, default):
    value = float(config.get(key, default))
    if not 0 <= value <= 1:
        raise ValueError(f"data.{key} must be in [0,1]")
    return value


def _ordered_pair(config, key, default, low=None, high=None, positive=False, integer=False):
    value = config.get(key, default)
    try:
        first, second = (float(item) for item in value)
    except (TypeError, ValueError):
        raise ValueError(f"data.{key} must contain two numbers") from None
    if not (math.isfinite(first) and math.isfinite(second)) or first > second:
        raise ValueError(f"data.{key} must contain two finite, ordered values")
    if integer and (first != int(first) or second != int(second)):
        raise ValueError(f"data.{key} must contain integers")
    if positive and first <= 0:
        raise ValueError(f"data.{key} values must be positive")
    if (low is not None and first < low) or (high is not None and second > high):
        raise ValueError(f"data.{key} values must lie in [{low},{high}]")
    return first, second


class MoonDataset(Dataset):
    def __init__(self, config, split="train"):
        self.config = config
        self.split = split
        # Resolve storage links once, before workers repeatedly open image pairs.
        # Keep config and manifest records unchanged for checkpoint comparisons.
        self.root = Path(config["root"]).resolve()
        self.training = split == "train"
        scenes = set(config.get(f"{split}_scenes", [1, 3, 6, 7, 8] if self.training else [5]))
        if not scenes <= TRAIN_SCENES:
            raise ValueError("Only scenes 1,3,5,6,7,8 may provide task labels")
        manifest = config[f"{split}_manifest"]
        self.records = load_manifest(manifest, self.root, scenes, config.get("official_allowlist"), need_label=True,
                                     allow_development_pool=bool(config.get("development_pool", False)))
        # "official_allowlist" or "public_development_pool" (method development only).
        self.provenance = manifest_provenance(manifest)
        self.mean = np.asarray(config.get("mean", [0.485, 0.456, 0.406]), np.float32)
        self.std = np.asarray(config.get("std", [0.229, 0.224, 0.225]), np.float32)
        self.train_mode = config.get("train_mode", "full_image")
        if self.training:
            if self.train_mode not in ("full_image", "random_resize_crop"):
                raise ValueError("data.train_mode must be full_image or random_resize_crop")
            size_key = "image_size" if self.train_mode == "full_image" else "crop_size"
            size = config.get(size_key, [1024, 1024])
            if len(size) != 2 or any(int(value) != value or value <= 0 or value % 32 for value in size):
                raise ValueError(f"data.{size_key} must contain two positive multiples of 32 in [H,W] order")
            if self.train_mode == "random_resize_crop":
                if int(config.get("crop_attempts", 10)) < 1:
                    raise ValueError("crop_attempts must be at least 1")
                low, high = map(float, config.get("resize_range", [0.5, 2.0]))
                if not 0 < low <= high:
                    raise ValueError("resize_range must be positive and ordered")
                if not 0 < float(config.get("cat_max_ratio", 0.75)) <= 1:
                    raise ValueError("cat_max_ratio must be in (0,1]")
            probability = float(config.get("photometric_probability", 0.5))
            for key in ("hflip_probability", "photometric_probability",
                        "brightness_probability", "contrast_probability"):
                default = 0.5 if key == "hflip_probability" else probability
                if not 0 <= float(config.get(key, default)) <= 1:
                    raise ValueError(f"data.{key} must be in [0,1]")
            brightness = float(config.get("brightness_delta", 16.0))
            if not 0 <= brightness <= 255:
                raise ValueError("brightness_delta must be in [0,255]")
            contrast = config.get("contrast_range", [0.9, 1.1])
            if (len(contrast) != 2 or not 0 <= float(contrast[0]) <= float(contrast[1])
                    or not np.isfinite(contrast).all()):
                raise ValueError("contrast_range must contain two finite, nonnegative ordered values")
            self._validate_augmentation_v2(config)

    def _validate_augmentation_v2(self, config):
        defaults = AUGMENTATION_V2_DEFAULTS
        if _probability(config, "scale_jitter_probability", defaults["scale_jitter_probability"]) > 0 \
                and self.train_mode != "full_image":
            raise ValueError("scale_jitter_* applies only to train_mode=full_image; "
                             "random_resize_crop has its own resize_range")
        _ordered_pair(config, "scale_jitter_range", defaults["scale_jitter_range"], positive=True)
        _probability(config, "gamma_probability", defaults["gamma_probability"])
        _ordered_pair(config, "gamma_range", defaults["gamma_range"], positive=True)
        _probability(config, "shadow_probability", defaults["shadow_probability"])
        _ordered_pair(config, "shadow_count_range", defaults["shadow_count_range"], low=1, integer=True)
        _ordered_pair(config, "shadow_strength_range", defaults["shadow_strength_range"], low=0, high=1)
        _ordered_pair(config, "shadow_blur_range", defaults["shadow_blur_range"], low=0)
        _ordered_pair(config, "shadow_size_range", defaults["shadow_size_range"], high=1, positive=True)
        floor = config.get("pixel_floor", defaults["pixel_floor"])
        if isinstance(floor, bool) or int(floor) != floor or not 0 <= int(floor) <= 255:
            raise ValueError("data.pixel_floor must be an integer in [0,255]")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(self.root / record["image"]) as handle:
            image = handle.convert("RGB")
        with Image.open(self.root / record["label"]) as handle:
            label = Image.fromarray(decode_label(handle))
        if image.size != label.size:
            raise ValueError(f"Image/label dimensions differ: {record['sample_id']}")
        if self.training:
            image, label = self.augment(image, label)
        pixels = (np.asarray(image, np.float32) / 255.0 - self.mean) / self.std
        return {"image": torch.from_numpy(pixels.transpose(2, 0, 1).copy()),
                "target": torch.from_numpy(np.array(label, dtype=np.int64)),
                "sample_id": str(record["sample_id"]), "scene": int(record["scene"])}

    def _setting(self, key):
        return self.config.get(key, AUGMENTATION_V2_DEFAULTS[key])

    def augment(self, image, label):
        """PIL RGB/L in; uint8 HxWx3 pixels and uint8 HxW label out.

        Order: [full_image resize] -> scale jitter -> hflip -> brightness -> contrast
        -> gamma -> shadow -> pixel floor (normalization follows in __getitem__).
        Labels only receive the geometric operations.
        """
        image, label = self.geometry(image, label)
        pixels = self._photometric(np.asarray(image))
        label = np.asarray(label)
        pixels = self._gamma(pixels)
        pixels = self._shadow(pixels, label)
        floor = int(self._setting("pixel_floor"))
        if floor > 0:
            # Synthetic darkening never goes below the renderer's black floor (22 in LuSNAR).
            pixels = np.maximum(pixels, np.uint8(floor))
        return pixels, label

    def geometry(self, image, label):
        """Synchronized geometry: full_image resize/scale jitter or random_resize_crop, then hflip."""
        if self.train_mode == "full_image":
            # Keep the complete field of view. In particular, legacy crop/zoom
            # settings cannot implicitly turn full-image training into crops.
            height, width = map(int, self.config.get("image_size", [1024, 1024]))
            if image.size != (width, height):
                image = image.resize((width, height), Image.Resampling.BILINEAR)
                label = label.resize((width, height), Image.Resampling.NEAREST)
            probability = float(self._setting("scale_jitter_probability"))
            # Draw only when enabled so the default random stream is unchanged.
            if probability > 0 and random.random() < probability:
                image, label = self._scale_jitter(image, label)
        else:
            image, label = self._random_resize_crop(image, label)
        if random.random() < float(self.config.get("hflip_probability", 0.5)):
            image, label = ImageOps.mirror(image), ImageOps.mirror(label)
        return image, label

    def _scale_jitter(self, image, label):
        """Resize by s (RGB bilinear, label nearest) and restore the canvas size.

        s > 1: random crop back to the canvas; s < 1: pad bottom/right with the mean
        colour (label 255, ignored). Only used in full_image mode.
        """
        low, high = map(float, self._setting("scale_jitter_range"))
        scale = random.uniform(low, high)
        width, height = image.size
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        if size != (width, height):
            image = image.resize(size, Image.Resampling.BILINEAR)
            label = label.resize(size, Image.Resampling.NEAREST)
        pad_w, pad_h = max(0, width - size[0]), max(0, height - size[1])
        if pad_w or pad_h:
            fill = tuple(round(x * 255) for x in self.mean)
            image = ImageOps.expand(image, (0, 0, pad_w, pad_h), fill=fill)
            label = ImageOps.expand(label, (0, 0, pad_w, pad_h), fill=255)
        if image.size != (width, height):
            x = random.randint(0, image.width - width)
            y = random.randint(0, image.height - height)
            box = (x, y, x + width, y + height)
            image, label = image.crop(box), label.crop(box)
        return image, label

    def _photometric(self, pixels):
        """Brightness/contrast subset of DINOv3/MMSeg PhotoMetricDistortion on uint8 RGB.

        Each operation has its own probability, without an outer random gate.
        photometric_probability supplies the default for both operation gates;
        brightness_probability/contrast_probability can override them separately.
        brightness_delta and contrast_range come from the config (the baseline sets the
        standard DINOv3/MMSeg strengths, +/-32 and 0.5-1.5); when omitted they default to
        the code fallback values below (+/-16 and 0.9-1.1).
        """
        probability = float(self.config.get("photometric_probability", 0.5))
        pixels = np.asarray(pixels)
        if random.random() < float(self.config.get("brightness_probability", probability)):
            delta = float(self.config.get("brightness_delta", 16.0))
            pixels = np.clip(pixels.astype(np.float32) + random.uniform(-delta, delta),
                             0, 255).astype(np.uint8)
        if random.random() < float(self.config.get("contrast_probability", probability)):
            low, high = map(float, self.config.get("contrast_range", [0.9, 1.1]))
            pixels = np.clip(pixels.astype(np.float32) * random.uniform(low, high),
                             0, 255).astype(np.uint8)
        return pixels

    def _gamma(self, pixels):
        """I' = 255 * (I/255) ** gamma, float32 then clip and uint8 (via a 256-entry table)."""
        probability = float(self._setting("gamma_probability"))
        if not (probability > 0 and random.random() < probability):
            return pixels
        low, high = map(float, self._setting("gamma_range"))
        gamma = random.uniform(low, high)
        table = np.clip(255.0 * (np.arange(256, dtype=np.float32) / 255.0) ** np.float32(gamma), 0, 255)
        return table.astype(np.uint8)[pixels]

    def _shadow(self, pixels, label):
        """Spatially varying multiplicative darkening of non-sky pixels only.

        1..N random polygons (3-7 vertices) or rotated ellipses are placed anywhere,
        their masks Gaussian-blurred for soft edges, and pixels multiplied by
        1 - strength * mask (overlaps keep the darkest factor). Sky (label 4) and
        ignore padding (255) are left exactly unchanged; the label is not modified.
        """
        probability = float(self._setting("shadow_probability"))
        if not (probability > 0 and random.random() < probability):
            return pixels
        factor = self.shadow_factor(label.shape)
        factor[label >= SKY_ID] = 1.0
        return np.clip(pixels.astype(np.float32) * factor[..., None], 0, 255).astype(np.uint8)

    def shadow_factor(self, shape):
        height, width = shape
        count_low, count_high = map(int, self._setting("shadow_count_range"))
        strength_low, strength_high = map(float, self._setting("shadow_strength_range"))
        blur_low, blur_high = map(float, self._setting("shadow_blur_range"))
        size_low, size_high = map(float, self._setting("shadow_size_range"))
        factor = np.ones((height, width), np.float32)
        for _ in range(random.randint(count_low, count_high)):
            radius = random.uniform(size_low, size_high) * min(height, width)
            center_x, center_y = random.uniform(0, width), random.uniform(0, height)
            if random.random() < 0.5:
                vertices = random.randint(3, 7)
                angles = sorted(random.uniform(0, 2 * math.pi) for _ in range(vertices))
                points = [(center_x + radius * random.uniform(0.4, 1.0) * math.cos(angle),
                           center_y + radius * random.uniform(0.4, 1.0) * math.sin(angle)) for angle in angles]
            else:
                minor = radius * random.uniform(0.15, 1.0)
                rotation = random.uniform(0, math.pi)
                cos_r, sin_r = math.cos(rotation), math.sin(rotation)
                points = []
                for step in range(64):
                    t = 2 * math.pi * step / 64
                    x, y = radius * math.cos(t), minor * math.sin(t)
                    points.append((center_x + x * cos_r - y * sin_r, center_y + x * sin_r + y * cos_r))
            blur = random.uniform(blur_low, blur_high)
            strength = random.uniform(strength_low, strength_high)
            # Rasterize and blur only the shape's bounding box plus the blur support.
            margin = int(math.ceil(3 * blur)) + 2
            xs, ys = [point[0] for point in points], [point[1] for point in points]
            left, top = max(0, int(math.floor(min(xs))) - margin), max(0, int(math.floor(min(ys))) - margin)
            right, bottom = min(width, int(math.ceil(max(xs))) + margin + 1), min(height, int(math.ceil(max(ys))) + margin + 1)
            if right <= left or bottom <= top:
                continue
            mask = Image.new("L", (right - left, bottom - top), 0)
            ImageDraw.Draw(mask).polygon([(x - left, y - top) for x, y in points], fill=255)
            if blur > 0:
                mask = mask.filter(ImageFilter.GaussianBlur(blur))
            region = 1.0 - np.float32(strength) * (np.asarray(mask, np.float32) / 255.0)
            np.minimum(factor[top:bottom, left:right], region, out=factor[top:bottom, left:right])
        return factor

    def _random_resize_crop(self, image, label):
        """Optional scale jitter and class-balanced crop; never used in full_image."""
        crop_h, crop_w = map(int, self.config.get("crop_size", [1024, 1024]))
        base_h, base_w = self.config.get("resize_size", [crop_h, crop_w])
        low, high = self.config.get("resize_range", [0.5, 2.0])
        # Preserve terrain geometry. At scale=1, the resized image covers the reference crop.
        scale = max(base_w / image.width, base_h / image.height) * random.uniform(low, high)
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(size, Image.Resampling.BILINEAR)
        label = label.resize(size, Image.Resampling.NEAREST)
        pad_w, pad_h = max(0, crop_w - size[0]), max(0, crop_h - size[1])
        if pad_w or pad_h:
            fill = tuple(round(x * 255) for x in self.mean)
            image = ImageOps.expand(image, (0, 0, pad_w, pad_h), fill=fill)
            label = ImageOps.expand(label, (0, 0, pad_w, pad_h), fill=255)
        maximum = float(self.config.get("cat_max_ratio", 0.75))
        for _ in range(int(self.config.get("crop_attempts", 10))):
            x = random.randint(0, image.width - crop_w)
            y = random.randint(0, image.height - crop_h)
            box = (x, y, x + crop_w, y + crop_h)
            crop = np.asarray(label.crop(box))
            _, counts = np.unique(crop[crop != 255], return_counts=True)
            if len(counts) > 1 and counts.max() / counts.sum() < maximum:
                break
        image, label = image.crop(box), label.crop(box)
        return image, label


def collate_samples(batch):
    """Pad variable image sizes; all padded label pixels are ignored by the criterion."""
    height = max(item["image"].shape[-2] for item in batch)
    width = max(item["image"].shape[-1] for item in batch)
    images = batch[0]["image"].new_zeros((len(batch), 3, height, width))
    targets = torch.full((len(batch), height, width), 255, dtype=torch.long)
    sizes = []
    for index, item in enumerate(batch):
        h, w = item["target"].shape
        images[index, :, :h, :w] = item["image"]
        targets[index, :h, :w] = item["target"]
        sizes.append((h, w))
    return {"images": images, "targets": targets, "sizes": sizes,
            "sample_ids": [item["sample_id"] for item in batch],
            "scenes": [item["scene"] for item in batch]}
