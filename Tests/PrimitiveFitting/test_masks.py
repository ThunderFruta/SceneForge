from __future__ import annotations

from PrimitiveFitting.masks import polygon_to_mask, sampled_mask_pixels


def test_simple_polygon_mask_produces_points() -> None:
    mask = polygon_to_mask(
        polygon=[(2, 2), (7, 2), (7, 7), (2, 7)],
        image_width=10,
        image_height=10,
    )
    pixels = sampled_mask_pixels(mask)

    assert mask.shape == (10, 10)
    assert len(pixels) > 0
    assert pixels.shape[1] == 2
