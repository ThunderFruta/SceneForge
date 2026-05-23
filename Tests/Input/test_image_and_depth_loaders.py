from __future__ import annotations

from pathlib import Path

from Input.Depth.depth_loader import derive_depth_from_luminance, load_depth_map
from Input.Image.image_loader import load_rgb_image


FIXTURES = Path("Assets/Fixtures")


def test_load_rgb_image_converts_fixture_to_rgb() -> None:
    image = load_rgb_image(FIXTURES / "tiny_rgb.ppm")

    assert image.mode == "RGB"
    assert image.size == (2, 2)
    assert image.getpixel((0, 0)) == (255, 0, 0)


def test_load_depth_map_normalizes_grayscale_values() -> None:
    depth = load_depth_map(FIXTURES / "tiny_depth.pgm")

    assert depth == [
        [0.0, 128.0 / 255.0],
        [192.0 / 255.0, 1.0],
    ]


def test_derive_depth_from_luminance_matches_image_size() -> None:
    image = load_rgb_image(FIXTURES / "tiny_rgb.ppm")
    depth = derive_depth_from_luminance(image)

    assert len(depth) == 2
    assert len(depth[0]) == 2
    assert all(0.0 <= value <= 1.0 for row in depth for value in row)

