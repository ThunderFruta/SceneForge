from __future__ import annotations

import math
from pathlib import Path

from PIL import Image

from Segmentation.types import SegmentDetection
from ShapeDetection.depth_geometry_classifier import DepthGeometryPrimitiveClassifier


def make_detection(polygon: list[tuple[float, float]]) -> SegmentDetection:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return SegmentDetection(
        bbox_xyxy=(min(xs), min(ys), max(xs), max(ys)),
        mask_polygon=polygon,
        detector_label="unknown",
        detector_confidence=0.5,
    )


def circle_polygon(cx: float, cy: float, radius: float, points: int = 32) -> list[tuple[float, float]]:
    return [
        (
            cx + math.cos(2.0 * math.pi * index / points) * radius,
            cy + math.sin(2.0 * math.pi * index / points) * radius,
        )
        for index in range(points)
    ]


def capsule_polygon(
    left_cx: float,
    right_cx: float,
    cy: float,
    radius: float,
    points: int = 12,
) -> list[tuple[float, float]]:
    right = [
        (
            right_cx + math.cos(-0.5 * math.pi + math.pi * index / points) * radius,
            cy + math.sin(-0.5 * math.pi + math.pi * index / points) * radius,
        )
        for index in range(points + 1)
    ]
    left = [
        (
            left_cx + math.cos(0.5 * math.pi + math.pi * index / points) * radius,
            cy + math.sin(0.5 * math.pi + math.pi * index / points) * radius,
        )
        for index in range(points + 1)
    ]
    return right + left


def test_depth_geometry_classifier_labels_synthetic_mask_primitives(tmp_path: Path) -> None:
    depth_path = tmp_path / "depth.png"
    Image.new("L", (32, 32), 160).save(depth_path)
    classifier = DepthGeometryPrimitiveClassifier(depth_path)
    image = Image.new("RGB", (32, 32), (150, 150, 150))

    cases = {
        "sphere": circle_polygon(16.0, 16.0, 10.0),
        "cylinder": capsule_polygon(9.0, 23.0, 16.0, 6.0),
        "box": [(5.0, 6.0), (27.0, 6.0), (27.0, 26.0), (5.0, 26.0)],
        "cone": [(16.0, 4.0), (28.0, 28.0), (4.0, 28.0)],
    }
    for expected, polygon in cases.items():
        prediction = classifier.classify(image, make_detection(polygon))
        assert prediction.label == expected
        assert prediction.source == "depth_geometry_weak"
