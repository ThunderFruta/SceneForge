from __future__ import annotations

from PIL import Image

from Segmentation.types import SegmentDetection
from ShapeDetection.detector_label_classifier import DetectorLabelPrimitiveClassifier


def test_detector_label_classifier_trusts_primitive_detector_labels() -> None:
    detection = SegmentDetection(
        bbox_xyxy=(0, 0, 10, 10),
        mask_polygon=[],
        detector_label="cone",
        detector_confidence=0.88,
    )

    prediction = DetectorLabelPrimitiveClassifier().classify(Image.new("RGB", (10, 10)), detection)

    assert prediction.label == "cone"
    assert prediction.confidence == 0.88


def test_detector_label_classifier_maps_non_primitive_labels_to_unknown() -> None:
    detection = SegmentDetection(
        bbox_xyxy=(0, 0, 10, 10),
        mask_polygon=[],
        detector_label="person",
        detector_confidence=0.5,
    )

    prediction = DetectorLabelPrimitiveClassifier().classify(Image.new("RGB", (10, 10)), detection)

    assert prediction.label == "unknown"
    assert prediction.confidence == 0.5


def test_detector_label_classifier_keeps_extended_geometry_label() -> None:
    detection = SegmentDetection(
        bbox_xyxy=(0, 0, 10, 10),
        mask_polygon=[(0, 0), (10, 0), (10, 10)],
        detector_label="torus",
        detector_confidence=0.82,
    )

    prediction = DetectorLabelPrimitiveClassifier().classify(Image.new("RGB", (16, 16)), detection)

    assert prediction.label == "torus"
    assert prediction.confidence == 0.82
