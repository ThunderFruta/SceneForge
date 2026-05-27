from __future__ import annotations

import json
from pathlib import Path

from OutputWriter.report_writer import write_report
from ShapeDetection.report import DetectionReport, ObjectShapeDetection


def test_detection_report_serializes_stable_json(tmp_path: Path) -> None:
    report = DetectionReport(
        image_path="example.png",
        image_width=20,
        image_height=10,
        objects=[
            ObjectShapeDetection(
                id=1,
                bbox_xyxy=(1.12345, 2.0, 9.0, 8.0),
                mask_polygon=[(1.0, 2.0), (9.0, 2.0), (9.0, 8.0)],
                detector_label="object",
                detector_confidence=0.987654321,
                primitive_label="box",
                primitive_confidence=0.7654321,
            )
        ],
        model_info={"detector_backend": "fake"},
    )

    output = tmp_path / "detections.json"
    write_report(report, output)

    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["objects"][0]["bbox_xyxy"] == [1.123, 2.0, 9.0, 8.0]
    assert data["objects"][0]["detector_confidence"] == 0.987654
    assert data["objects"][0]["primitive_label"] == "box"
    assert list(data.keys()) == ["image_height", "image_path", "image_width", "model_info", "objects"]
