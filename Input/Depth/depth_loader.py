from __future__ import annotations

from pathlib import Path

from PIL import Image


DepthMap = list[list[float]]


def load_depth_map(depth_path: str | Path) -> DepthMap:
    path = Path(depth_path)
    if not path.exists():
        raise FileNotFoundError(f"Depth file does not exist: {path}")

    with Image.open(path) as image:
        gray = image.convert("L")
        return _image_to_normalized_depth(gray)


def derive_depth_from_luminance(image: Image.Image) -> DepthMap:
    return _image_to_normalized_depth(image.convert("L"))


def _image_to_normalized_depth(image: Image.Image) -> DepthMap:
    width, height = image.size
    pixels = image.load()
    return [
        [float(pixels[x, y]) / 255.0 for x in range(width)]
        for y in range(height)
    ]

