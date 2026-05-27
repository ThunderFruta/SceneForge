from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from Input.Depth.depth_loader import DepthLoadError, load_grayscale_depth


def test_depth_loader_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DepthLoadError, match="Depth path does not exist"):
        load_grayscale_depth(tmp_path / "missing.png")


def test_depth_loader_validates_expected_size(tmp_path: Path) -> None:
    depth_path = tmp_path / "depth.png"
    Image.new("L", (4, 3), 128).save(depth_path)

    with pytest.raises(DepthLoadError, match="does not match"):
        load_grayscale_depth(depth_path, expected_size=(5, 3))


def test_depth_loader_normalizes_white_and_black(tmp_path: Path) -> None:
    depth_path = tmp_path / "depth.png"
    Image.new("L", (2, 1)).save(depth_path)
    image = Image.open(depth_path)
    image.putpixel((0, 0), 255)
    image.putpixel((1, 0), 0)
    image.save(depth_path)

    depth = load_grayscale_depth(depth_path)

    assert depth[0, 0] == 1.0
    assert depth[0, 1] == 0.0
