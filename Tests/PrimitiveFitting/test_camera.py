from __future__ import annotations

import numpy as np

from PrimitiveFitting.camera import PinholeCamera


def test_white_depth_is_closer_than_black_depth() -> None:
    camera = PinholeCamera(image_width=100, image_height=100, near_depth=1.0, far_depth=6.0)

    assert camera.depth_value_to_scene_depth(1.0) == 1.0
    assert camera.depth_value_to_scene_depth(0.0) == 6.0


def test_pinhole_unprojection_is_deterministic() -> None:
    camera = PinholeCamera(image_width=100, image_height=100, fov_degrees=90.0, near_depth=1.0, far_depth=5.0)
    pixels = np.array([[49, 49], [99, 0]], dtype=np.int32)
    depth = np.array([1.0, 0.0], dtype=np.float32)

    points = camera.unproject_pixels(pixels, depth)

    assert points.shape == (2, 3)
    assert np.allclose(points[0], (-0.01, 1.0, 0.01), atol=1e-6)
    assert np.allclose(points[1], (4.95, 5.0, 4.95), atol=1e-6)


def test_horizontal_sensor_fit_uses_image_width_for_focal_length() -> None:
    horizontal = PinholeCamera(image_width=200, image_height=100, fov_degrees=90.0, sensor_fit="horizontal")
    vertical = PinholeCamera(image_width=200, image_height=100, fov_degrees=90.0, sensor_fit="vertical")

    assert np.isclose(horizontal.focal_length_pixels, 100.0)
    assert np.isclose(vertical.focal_length_pixels, 50.0)


def test_camera_metadata_includes_shared_fusion_contract() -> None:
    camera = PinholeCamera(image_width=640, image_height=480, fov_degrees=70.0, near_depth=1.0, far_depth=8.0)

    data = camera.to_dict()
    contract = data["fusion_contract"]

    assert contract["camera_model"] == "pinhole"
    assert contract["fov_degrees"] == 70.0
    assert contract["sensor_fit"] == "horizontal"
    assert contract["scene"]["coordinate_system"] == "sceneforge_camera_v1"
    assert contract["scene"]["axes"] == {
        "x": "image_right",
        "y": "depth_away_from_camera",
        "z": "image_up",
    }
    assert contract["depth"]["convention"] == "white_close_black_far"
