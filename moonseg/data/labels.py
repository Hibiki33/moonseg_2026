"""Official challenge IDs. Sky is a scored class, never ignore."""
import numpy as np

CLASS_NAMES = ("regolith", "rock", "crater", "mountain", "sky")
PALETTE = np.array([(187, 70, 156), (232, 250, 80), (120, 0, 200),
                    (173, 69, 31), (34, 201, 248)], dtype=np.uint8)


def decode_label(image):
    """Decode exact official RGB colors or already-converted class IDs; reject unknowns."""
    # P-mode files can carry arbitrary palette indices, so decode through their RGB palette.
    if image.mode == "P":
        image = image.convert("RGB")
    pixels = np.asarray(image)
    if pixels.ndim == 2:
        if not np.isin(pixels, np.arange(5)).all():
            raise ValueError("Single-channel label contains IDs outside 0..4")
        return pixels.astype(np.uint8)
    if pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
        raise ValueError(f"Unsupported label shape: {pixels.shape}")
    pixels = pixels[:, :, :3]
    result = np.full(pixels.shape[:2], 255, dtype=np.uint8)
    for class_id, color in enumerate(PALETTE):
        result[np.all(pixels == color, axis=-1)] = class_id
    if np.any(result == 255):
        colors, counts = np.unique(pixels[result == 255], axis=0, return_counts=True)
        order = np.argsort(counts)[::-1][:8]
        raise ValueError(f"Unknown label colors (RGB/count): {list(zip(colors[order].tolist(), counts[order].tolist()))}")
    return result
