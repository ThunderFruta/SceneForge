from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SegmentDetection:
    bbox_xyxy: tuple[float, float, float, float]
    mask_polygon: list[tuple[float, float]]
    detector_label: str
    detector_confidence: float

    def normalized(self, image_width: int, image_height: int) -> "SegmentDetection":
        left, top, right, bottom = self.bbox_xyxy
        clamped_box = (
            max(0.0, min(float(image_width), left)),
            max(0.0, min(float(image_height), top)),
            max(0.0, min(float(image_width), right)),
            max(0.0, min(float(image_height), bottom)),
        )
        clamped_polygon = [
            (
                max(0.0, min(float(image_width), x)),
                max(0.0, min(float(image_height), y)),
            )
            for x, y in self.mask_polygon
        ]
        return SegmentDetection(
            bbox_xyxy=clamped_box,
            mask_polygon=clamped_polygon,
            detector_label=self.detector_label,
            detector_confidence=max(0.0, min(1.0, self.detector_confidence)),
        )
