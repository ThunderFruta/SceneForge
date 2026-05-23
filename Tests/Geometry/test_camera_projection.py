from __future__ import annotations

from Geometry.Projection.camera_projection import (
    image_uv,
    project_image_depth_to_point,
    ray_through_image_point,
)


def test_top_left_image_point_maps_to_left_up_forward() -> None:
    point = project_image_depth_to_point(
        u=0.0,
        raw_v=0.0,
        depth=0.5,
        aspect_ratio=2.0,
        depth_strength=1.0,
    )

    assert point == (-1.5, 1.5, 0.75)


def test_bottom_right_image_point_maps_to_right_down_forward() -> None:
    point = project_image_depth_to_point(
        u=1.0,
        raw_v=1.0,
        depth=0.5,
        aspect_ratio=2.0,
        depth_strength=1.0,
    )

    assert point == (1.5, 1.5, -0.75)


def test_image_uv_flips_raw_image_y() -> None:
    assert image_uv(0.25, 0.75) == (0.25, 0.25)


def test_camera_ray_uses_canonical_positive_y_depth() -> None:
    assert ray_through_image_point(1.0, 0.0, aspect_ratio=2.0) == (1.0, 1.0, 0.5)
