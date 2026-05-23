from __future__ import annotations

from pathlib import Path

from PIL import Image


def load_rgb_image(image_path: str | Path) -> Image.Image:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {path}")

    with Image.open(path) as image:
        return image.convert("RGB")

