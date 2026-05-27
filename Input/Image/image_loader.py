from __future__ import annotations

from pathlib import Path

from PIL import Image, UnidentifiedImageError


class ImageLoadError(ValueError):
    pass


def load_rgb_image(path: str | Path) -> Image.Image:
    image_path = Path(path)
    if not image_path.is_file():
        raise ImageLoadError(f"Image path does not exist or is not a file: {image_path}")

    try:
        with Image.open(image_path) as image:
            return image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise ImageLoadError(f"Image file could not be decoded: {image_path}") from exc
