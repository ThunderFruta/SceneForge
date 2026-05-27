from __future__ import annotations

from dataclasses import dataclass
from math import radians, tan

import numpy as np

from SceneGeometry.coordinate_contract import camera_fusion_contract


@dataclass(frozen=True)
class PinholeCamera:
    image_width: int
    image_height: int
    fov_degrees: float = 70.0
    sensor_fit: str = "horizontal"
    near_depth: float = 1.0
    far_depth: float = 6.0

    def __post_init__(self) -> None:
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("Camera image dimensions must be positive.")
        if self.fov_degrees <= 0.0 or self.fov_degrees >= 179.0:
            raise ValueError("--fov-degrees must be between 0 and 179.")
        if self.sensor_fit not in {"horizontal", "vertical"}:
            raise ValueError("--sensor-fit must be horizontal or vertical.")
        if self.near_depth <= 0.0:
            raise ValueError("--near-depth must be greater than 0.")
        if self.far_depth <= self.near_depth:
            raise ValueError("--far-depth must be greater than --near-depth.")

    @property
    def focal_length_pixels(self) -> float:
        sensor_pixels = self.image_width if self.sensor_fit == "horizontal" else self.image_height
        return sensor_pixels / (2.0 * tan(radians(self.fov_degrees) / 2.0))

    def depth_value_to_scene_depth(self, value: np.ndarray | float) -> np.ndarray | float:
        return self.near_depth + (1.0 - value) * (self.far_depth - self.near_depth)

    def unproject_pixels(
        self,
        pixel_xy: np.ndarray,
        depth_values: np.ndarray,
    ) -> np.ndarray:
        depth = self.depth_value_to_scene_depth(depth_values.astype(np.float64))
        return self.unproject_scene_depth_pixels(pixel_xy, depth)

    def unproject_scene_depth_pixels(
        self,
        pixel_xy: np.ndarray,
        scene_depths: np.ndarray,
    ) -> np.ndarray:
        if len(pixel_xy) == 0:
            return np.empty((0, 3), dtype=np.float64)

        pixels = pixel_xy.astype(np.float64)
        depth = scene_depths.astype(np.float64)
        focal = self.focal_length_pixels
        center_x = self.image_width / 2.0
        center_y = self.image_height / 2.0

        x = ((pixels[:, 0] + 0.5) - center_x) * depth / focal
        y = depth
        z = (center_y - (pixels[:, 1] + 0.5)) * depth / focal
        return np.column_stack((x, y, z))

    def to_dict(self) -> dict:
        fusion_contract = camera_fusion_contract(
            image_width=self.image_width,
            image_height=self.image_height,
            fov_degrees=self.fov_degrees,
            sensor_fit=self.sensor_fit,
            near_depth=self.near_depth,
            far_depth=self.far_depth,
        )
        return {
            "model": "pinhole",
            "fov_degrees": self.fov_degrees,
            "sensor_fit": self.sensor_fit,
            "near_depth": self.near_depth,
            "far_depth": self.far_depth,
            "depth_convention": "white_close_black_far",
            "coordinate_system": "x_right_y_depth_away_z_up",
            "fusion_contract": fusion_contract,
        }
