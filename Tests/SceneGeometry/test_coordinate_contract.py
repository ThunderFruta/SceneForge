from __future__ import annotations

from SceneGeometry.coordinate_contract import (
    camera_fusion_contract,
    crop_coordinate_contract,
    normalize_sensor_fit,
)


def test_camera_fusion_contract_pins_axes_fov_and_depth() -> None:
    contract = camera_fusion_contract(
        image_width=640,
        image_height=640,
        fov_degrees=70.0,
        sensor_fit="HORIZONTAL",
        near_depth=1.0,
        far_depth=8.0,
    )

    assert contract["sensor_fit"] == "horizontal"
    assert contract["scene"]["axes"]["x"] == "image_right"
    assert contract["scene"]["axes"]["y"] == "depth_away_from_camera"
    assert contract["scene"]["axes"]["z"] == "image_up"
    assert contract["scene"]["upright_axis"] == "z"
    assert contract["depth"]["near_depth"] == 1.0
    assert contract["depth"]["far_depth"] == 8.0


def test_unknown_sensor_fit_normalizes_to_horizontal() -> None:
    assert normalize_sensor_fit("AUTO") == "horizontal"


def test_crop_contract_records_parent_and_local_axes() -> None:
    contract = crop_coordinate_contract(
        parent_image_width=640,
        parent_image_height=480,
        crop_box_xyxy=(10, 20, 110, 220),
        detection_id=7,
    )

    assert contract["parent_image_size"] == [640, 480]
    assert contract["crop_box_xyxy"] == [10, 20, 110, 220]
    assert contract["crop_size"] == [100, 200]
    assert contract["x"] == "right"
    assert contract["y"] == "down"
