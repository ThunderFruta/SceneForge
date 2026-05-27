from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


def polygon_to_mask(
    polygon: list[tuple[float, float]],
    image_width: int,
    image_height: int,
) -> np.ndarray:
    mask = Image.new("L", (image_width, image_height), 0)
    if len(polygon) >= 3:
        ImageDraw.Draw(mask).polygon(polygon, fill=255)
    return np.asarray(mask, dtype=bool)


def sampled_mask_pixels(
    mask: np.ndarray,
    max_samples: int = 8000,
) -> np.ndarray:
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return np.empty((0, 2), dtype=np.int32)

    pixels = np.column_stack((cols, rows)).astype(np.int32)
    if len(pixels) <= max_samples:
        return pixels

    indices = np.linspace(0, len(pixels) - 1, max_samples, dtype=np.int64)
    return pixels[indices]
