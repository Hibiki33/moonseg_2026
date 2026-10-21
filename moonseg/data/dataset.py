from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
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
        if self.training:
            if any(int(size) <= 0 or int(size) % 32 for size in config.get("crop_size", [512, 1024])):
                raise ValueError("Training crop dimensions must be positive multiples of 32")
            if int(config.get("crop_attempts", 10)) < 1:
                raise ValueError("crop_attempts must be at least 1")

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
        crop_h, crop_w = self.config.get("crop_size", [512, 1024])
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
        if random.random() < float(self.config.get("hflip_probability", 0.5)):
            image, label = ImageOps.mirror(image), ImageOps.mirror(label)
        # Mild photometric changes retain the geometry and strong lunar shadows.
        if random.random() < float(self.config.get("photometric_probability", 0.5)):
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.9, 1.1))
            gamma = random.uniform(0.9, 1.1)
            table = [round(255 * (value / 255) ** gamma) for value in range(256)]
            image = image.point(table * 3)
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
