from __future__ import annotations

from dataclasses import dataclass

from ShapeDetection.primitive_labels import validate_primitive_label


@dataclass(frozen=True)
class ObjectShapeDetection:
    id: int
    bbox_xyxy: tuple[float, float, float, float]
    mask_polygon: list[tuple[float, float]]
    detector_label: str
    detector_confidence: float
    primitive_label: str
    primitive_confidence: float
    primitive_label_source: str = "classifier"

    def to_dict(self) -> dict:
        validate_primitive_label(self.primitive_label)
        return {
            "id": self.id,
            "bbox_xyxy": [round(value, 3) for value in self.bbox_xyxy],
            "mask_polygon": [
                [round(x, 3), round(y, 3)]
                for x, y in self.mask_polygon
            ],
            "detector_label": self.detector_label,
            "detector_confidence": round(self.detector_confidence, 6),
            "primitive_label": self.primitive_label,
            "primitive_confidence": round(self.primitive_confidence, 6),
            "primitive_label_source": self.primitive_label_source,
        }


@dataclass(frozen=True)
class DetectionReport:
    image_path: str
    image_width: int
    image_height: int
    objects: list[ObjectShapeDetection]
    model_info: dict

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "objects": [item.to_dict() for item in self.objects],
            "model_info": dict(self.model_info),
        }
