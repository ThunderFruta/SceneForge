from __future__ import annotations

from copy import deepcopy
from typing import Any


CONTRACT_VERSION = 1
DEFAULT_FOV_DEGREES = 70.0
DEFAULT_SENSOR_FIT = "horizontal"
DEFAULT_NEAR_DEPTH = 1.0
DEFAULT_FAR_DEPTH = 8.0
DEPTH_CONVENTION = "white_close_black_far"
SCENE_COORDINATE_SYSTEM = "sceneforge_camera_v1"
PIXEL_COORDINATE_SYSTEM = "image_pixel_top_left_v1"


def normalize_sensor_fit(sensor_fit: str | None) -> str:
    value = (sensor_fit or DEFAULT_SENSOR_FIT).lower()
    if value not in {"horizontal", "vertical"}:
        return DEFAULT_SENSOR_FIT
    return value


def scene_coordinate_contract() -> dict[str, Any]:
    return {
        "schema_version": CONTRACT_VERSION,
        "coordinate_system": SCENE_COORDINATE_SYSTEM,
        "units": "blender_units",
        "axes": {
            "x": "image_right",
            "y": "depth_away_from_camera",
            "z": "image_up",
        },
        "handedness": "right_handed",
        "origin": "source_camera_center_for_camera_layout",
        "upright_axis": "z",
        "blender_world_mapping": {
            "x": "+X",
            "y": "+Y",
            "z": "+Z",
            "camera_layout_camera_location": [0.0, 0.0, 0.0],
            "camera_layout_camera_forward": "+Y",
            "blender_camera_local_forward": "-Z",
        },
    }


def pixel_coordinate_contract() -> dict[str, Any]:
    return {
        "schema_version": CONTRACT_VERSION,
        "coordinate_system": PIXEL_COORDINATE_SYSTEM,
        "origin": "top_left_pixel_corner",
        "sample_position": "pixel_center",
        "x": "right",
        "y": "down",
    }


def depth_contract(near_depth: float = DEFAULT_NEAR_DEPTH, far_depth: float = DEFAULT_FAR_DEPTH) -> dict[str, Any]:
    return {
        "schema_version": CONTRACT_VERSION,
        "format": "8bit_grayscale_png_L",
        "convention": DEPTH_CONVENTION,
        "near_depth": float(near_depth),
        "far_depth": float(far_depth),
        "orientation": "rgb_aligned_no_flip",
    }


def camera_fusion_contract(
    *,
    image_width: int,
    image_height: int,
    fov_degrees: float = DEFAULT_FOV_DEGREES,
    sensor_fit: str = DEFAULT_SENSOR_FIT,
    near_depth: float = DEFAULT_NEAR_DEPTH,
    far_depth: float = DEFAULT_FAR_DEPTH,
) -> dict[str, Any]:
    return {
        "schema_version": CONTRACT_VERSION,
        "purpose": "all_source_detection_enrichment_fit_metrics_artifacts_share_this_frame",
        "image_width": int(image_width),
        "image_height": int(image_height),
        "camera_model": "pinhole",
        "fov_degrees": float(fov_degrees),
        "sensor_fit": normalize_sensor_fit(sensor_fit),
        "scene": scene_coordinate_contract(),
        "pixel": pixel_coordinate_contract(),
        "depth": depth_contract(near_depth, far_depth),
    }


def crop_coordinate_contract(
    *,
    parent_image_width: int,
    parent_image_height: int,
    crop_box_xyxy: tuple[int, int, int, int],
    detection_id: int,
) -> dict[str, Any]:
    x0, y0, x1, y1 = crop_box_xyxy
    return {
        "schema_version": CONTRACT_VERSION,
        "coordinate_system": "crop_pixel_top_left_v1",
        "parent_coordinate_system": PIXEL_COORDINATE_SYSTEM,
        "detection_id": int(detection_id),
        "parent_image_size": [int(parent_image_width), int(parent_image_height)],
        "crop_box_xyxy": [int(x0), int(y0), int(x1), int(y1)],
        "crop_size": [int(x1 - x0), int(y1 - y0)],
        "origin": "crop_top_left_pixel_corner",
        "x": "right",
        "y": "down",
        "depth_alignment": "crop_pixels_match_rgb_mask_pixels",
    }


def load_fusion_contract_from_camera_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    existing = metadata.get("fusion_contract")
    if isinstance(existing, dict):
        return deepcopy(existing)
    resolution = metadata.get("resolution") or [metadata.get("image_width", 0), metadata.get("image_height", 0)]
    return camera_fusion_contract(
        image_width=int(resolution[0]),
        image_height=int(resolution[1]),
        fov_degrees=float(metadata.get("fov_degrees", DEFAULT_FOV_DEGREES)),
        sensor_fit=str(metadata.get("sensor_fit", DEFAULT_SENSOR_FIT)),
        near_depth=float(metadata.get("near_depth", DEFAULT_NEAR_DEPTH)),
        far_depth=float(metadata.get("far_depth", DEFAULT_FAR_DEPTH)),
    )

