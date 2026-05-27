from ShapeDetection.report import ObjectShapeDetection
from OutputWriter.overlay import _display_values


def test_overlay_shows_detector_label_for_unassigned_detection() -> None:
    detection = ObjectShapeDetection(
        id=1,
        bbox_xyxy=(1, 2, 3, 4),
        mask_polygon=[(1, 2), (3, 2), (3, 4)],
        detector_label="cylinder",
        detector_confidence=0.98,
        primitive_label="unknown",
        primitive_confidence=0.0,
        primitive_label_source="unassigned",
    )

    assert _display_values(detection) == ("cylinder", 0.98, "cylinder")


def test_overlay_keeps_primitive_label_when_assigned() -> None:
    detection = ObjectShapeDetection(
        id=1,
        bbox_xyxy=(1, 2, 3, 4),
        mask_polygon=[(1, 2), (3, 2), (3, 4)],
        detector_label="object",
        detector_confidence=0.98,
        primitive_label="box",
        primitive_confidence=0.73,
        primitive_label_source="geometry_classifier",
    )

    assert _display_values(detection) == ("box", 0.73, "box")
