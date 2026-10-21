"""Official challenge IDs. Sky is a scored class, never ignore."""
import numpy as np

CLASS_NAMES = ("regolith", "rock", "crater", "mountain", "sky")
PALETTE = np.array([(187, 70, 156), (232, 250, 80), (120, 0, 200),
                    (173, 69, 31), (34, 201, 248)], dtype=np.uint8)
_PACKED_PALETTE = (PALETTE[:, 0].astype(np.uint32) << 16) | (PALETTE[:, 1].astype(np.uint32) << 8) | PALETTE[:, 2]


def _unknown_error(colors, counts):
    order = np.argsort(counts)[::-1][:8]
    return ValueError(f"Unknown label colors (RGB/count): {list(zip(np.asarray(colors)[order].tolist(), np.asarray(counts)[order].tolist()))}")


def _decode_palette(image):
    """P-mode: map palette entries to class IDs by their RGB color (palette index != class ID)."""
    indices = np.asarray(image)
    palette = np.asarray(image.getpalette() or [], dtype=np.uint8).reshape(-1, 3)
    lookup = np.full(256, 255, dtype=np.uint8)
    for class_id, color in enumerate(PALETTE):
        lookup[:len(palette)][np.all(palette == color, axis=1)] = class_id
    result = lookup[indices]
    if np.any(result == 255):
        counts = np.bincount(indices.ravel(), minlength=256)
        unknown = np.flatnonzero((lookup == 255) & (counts > 0))
        colors = [palette[index].tolist() if index < len(palette) else f"index {index} beyond palette" for index in unknown]
        raise _unknown_error(colors, counts[unknown])
    return result


def decode_label(image):
    """Decode exact official RGB colors or already-converted class IDs; reject unknowns."""
    # P-mode files can carry arbitrary palette indices, so decode through their RGB palette.
    if image.mode == "P":
        return _decode_palette(image)
    pixels = np.asarray(image)
    if pixels.ndim == 2:
        if not np.isin(pixels, np.arange(5)).all():
            raise ValueError("Single-channel label contains IDs outside 0..4")
        return pixels.astype(np.uint8)
    if pixels.ndim != 3 or pixels.shape[2] not in (3, 4) or pixels.dtype != np.uint8:
        raise ValueError(f"Unsupported label shape: {pixels.shape}")
    packed = (pixels[:, :, 0].astype(np.uint32) << 16) | (pixels[:, :, 1].astype(np.uint32) << 8) | pixels[:, :, 2]
    result = np.full(pixels.shape[:2], 255, dtype=np.uint8)
    for class_id, key in enumerate(_PACKED_PALETTE):
        result[packed == key] = class_id
    if np.any(result == 255):
        colors, counts = np.unique(pixels[:, :, :3][result == 255], axis=0, return_counts=True)
        raise _unknown_error(colors, counts)
    return result
