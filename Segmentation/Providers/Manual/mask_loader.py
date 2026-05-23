from __future__ import annotations

from pathlib import Path

from PIL import Image

from Segmentation.Core.segmentation_labels import color_to_label
from Segmentation.Core.segmentation_mask import SegmentationMask


def load_segmentation_mask(mask_path: str | Path) -> SegmentationMask:
    path = Path(mask_path)
    if not path.exists():
        raise FileNotFoundError(f"Segmentation mask file does not exist: {path}")

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        pixels = rgb.load()
        labels = [
            [color_to_label(pixels[x, y]) for x in range(width)]
            for y in range(height)
        ]
    return SegmentationMask.from_labels(labels)
