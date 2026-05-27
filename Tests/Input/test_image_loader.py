from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from Input.Image.image_loader import ImageLoadError, load_rgb_image


def test_load_rgb_image_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ImageLoadError):
        load_rgb_image(tmp_path / "missing.png")


def test_load_rgb_image_rejects_non_image_file(tmp_path: Path) -> None:
    path = tmp_path / "not_image.txt"
    path.write_text("not an image", encoding="utf-8")

    with pytest.raises(ImageLoadError):
        load_rgb_image(path)


def test_load_rgb_image_converts_to_rgb(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    Image.new("L", (4, 3), 128).save(path)

    image = load_rgb_image(path)

    assert image.mode == "RGB"
    assert image.size == (4, 3)
