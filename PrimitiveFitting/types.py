from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrimitiveFit:
    id: int
    primitive_label: str
    confidence: float
    center_xyz: tuple[float, float, float]
    rotation_matrix: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    dimensions_xyz: tuple[float, float, float]
    fit_quality: dict
    primitive_label_source: str = "detector"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "primitive_label": self.primitive_label,
            "primitive_label_source": self.primitive_label_source,
            "confidence": round(self.confidence, 6),
            "center_xyz": [round(value, 6) for value in self.center_xyz],
            "rotation_matrix": [
                [round(value, 6) for value in row]
                for row in self.rotation_matrix
            ],
            "dimensions_xyz": [round(value, 6) for value in self.dimensions_xyz],
            "fit_quality": self.fit_quality,
        }


@dataclass(frozen=True)
class PrimitiveFitReport:
    image_path: str
    depth_path: str
    detections_path: str
    image_width: int
    image_height: int
    camera: dict
    objects: list[PrimitiveFit]
    model_info: dict

    def to_dict(self) -> dict:
        return {
            "camera": self.camera,
            "depth_path": self.depth_path,
            "detections_path": self.detections_path,
            "image_height": self.image_height,
            "image_path": self.image_path,
            "image_width": self.image_width,
            "model_info": dict(self.model_info),
            "objects": [item.to_dict() for item in self.objects],
        }
