from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from .labels import decode_label
from .manifest import TRAIN_SCENES, load_manifest


class MoonDataset(Dataset):
    def __init__(self, config, split="train"):
        self.config = config
        self.split = split
        self.root = Path(config["root"])
        self.training = split == "train"
        scenes = set(config.get(f"{split}_scenes", [1, 3, 6, 7, 8] if self.training else [5]))
        if not scenes <= TRAIN_SCENES:
            raise ValueError("Only scenes 1,3,5,6,7,8 may provide task labels")
        self.records = load_manifest(config[f"{split}_manifest"], self.root, scenes,
                                     config.get("official_allowlist"), need_label=True)
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

    def augment(self, image, label):
        if self.train_mode == "full_image":
            # Keep the complete field of view. In particular, legacy crop/zoom
            # settings cannot implicitly turn full-image training into crops.
            height, width = map(int, self.config.get("image_size", [1024, 1024]))
            if image.size != (width, height):
                image = image.resize((width, height), Image.Resampling.BILINEAR)
                label = label.resize((width, height), Image.Resampling.NEAREST)
        else:
            image, label = self._random_resize_crop(image, label)
        if random.random() < float(self.config.get("hflip_probability", 0.5)):
            image, label = ImageOps.mirror(image), ImageOps.mirror(label)
        image = self._photometric(image)
        return image, label

    def _photometric(self, image):
        """Brightness/contrast subset of DINOv3/MMSeg PhotoMetricDistortion.

        Each operation has its own probability, without an outer random gate.
        photometric_probability supplies the default for both operation gates;
        brightness_probability/contrast_probability can override them separately.
        The reduced intensity ranges are local baseline hyperparameters.
        """
        probability = float(self.config.get("photometric_probability", 0.5))
        pixels = np.asarray(image)
        if random.random() < float(self.config.get("brightness_probability", probability)):
            delta = float(self.config.get("brightness_delta", 16.0))
            pixels = np.clip(pixels.astype(np.float32) + random.uniform(-delta, delta),
                             0, 255).astype(np.uint8)
        if random.random() < float(self.config.get("contrast_probability", probability)):
            low, high = map(float, self.config.get("contrast_range", [0.9, 1.1]))
            pixels = np.clip(pixels.astype(np.float32) * random.uniform(low, high),
                             0, 255).astype(np.uint8)
        return Image.fromarray(pixels)

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
