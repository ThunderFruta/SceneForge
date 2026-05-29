from __future__ import annotations

from PIL import Image

from Segmentation.types import SegmentDetection
from ShapeDetection.types import PrimitivePrediction


class FakeSegmenter:
    def __init__(self, mode: str = "sample") -> None:
        if mode not in {"sample", "none"}:
            raise ValueError(f"Unsupported fake segmenter mode: {mode}")
        self.mode = mode

    def detect(self, image: Image.Image) -> list[SegmentDetection]:
        if self.mode == "none":
            return []

        width, height = image.size
        left = round(width * 0.2, 2)
        top = round(height * 0.2, 2)
        right = round(width * 0.8, 2)
        bottom = round(height * 0.8, 2)
        polygon = [(left, top), (right, top), (right, bottom), (left, bottom)]
        return [
            SegmentDetection(
                bbox_xyxy=(left, top, right, bottom),
                mask_polygon=polygon,
                detector_label="object",
                detector_confidence=0.9,
            )
        ]


class FakePrimitiveClassifier:
    def classify(self, image: Image.Image, detection: SegmentDetection) -> PrimitivePrediction:
        del image
        left, top, right, bottom = detection.bbox_xyxy
        width = max(1.0, right - left)
        height = max(1.0, bottom - top)
        ratio = width / height
        label = "box" if 0.75 <= ratio <= 1.35 else "cylinder"
        return PrimitivePrediction(label=label, confidence=0.75)

