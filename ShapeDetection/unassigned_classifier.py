from __future__ import annotations

from PIL import Image

from Segmentation.types import SegmentDetection
from ShapeDetection.types import PrimitivePrediction


class UnassignedPrimitiveClassifier:
    def classify(self, image: Image.Image, detection: SegmentDetection) -> PrimitivePrediction:
        return PrimitivePrediction(label="unknown", confidence=0.0, source="unassigned")
