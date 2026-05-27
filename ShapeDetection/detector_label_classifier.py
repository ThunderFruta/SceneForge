from __future__ import annotations

from PIL import Image

from Segmentation.types import SegmentDetection
from ShapeDetection.primitive_labels import PRIMITIVE_LABELS
from ShapeDetection.types import PrimitivePrediction


class DetectorLabelPrimitiveClassifier:
    def classify(self, image: Image.Image, detection: SegmentDetection) -> PrimitivePrediction:
        label = detection.detector_label
        if label not in PRIMITIVE_LABELS:
            label = "unknown"
        return PrimitivePrediction(
            label=label,
            confidence=detection.detector_confidence,
            source="detector_label_legacy",
        )
