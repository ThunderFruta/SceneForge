from __future__ import annotations

from Geometry.UV.uv_projector import build_grid_uvs


def test_build_grid_uvs_uses_normalized_image_coordinates() -> None:
    assert build_grid_uvs(3, 2) == [
        (0.0, 1.0),
        (0.5, 1.0),
        (1.0, 1.0),
        (0.0, 0.0),
        (0.5, 0.0),
        (1.0, 0.0),
    ]

