from __future__ import annotations

import json
from pathlib import Path

from ShapeDetection.report import DetectionReport, ObjectShapeDetection


def load_detection_report(path: str | Path) -> DetectionReport:
    report_path = Path(path)
    if not report_path.is_file():
        raise ValueError(f"Detections path does not exist or is not a file: {report_path}")

    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed detections.json: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ValueError("Malformed detections.json: top-level value must be an object.")
    if "image_width" not in data or "image_height" not in data:
        raise ValueError("Malformed detections.json: image_width and image_height are required.")
    objects = [
        ObjectShapeDetection(
            id=int(item["id"]),
            bbox_xyxy=tuple(float(value) for value in item["bbox_xyxy"]),
            mask_polygon=[
                (float(point[0]), float(point[1]))
                for point in item.get("mask_polygon", [])
            ],
            detector_label=str(item.get("detector_label", "")),
            detector_confidence=float(item.get("detector_confidence", 0.0)),
            primitive_label=str(item["primitive_label"]),
            primitive_confidence=float(item.get("primitive_confidence", 0.0)),
            primitive_label_source=str(item.get("primitive_label_source", "classifier")),
        )
        for item in data.get("objects", [])
    ]
    return DetectionReport(
        image_path=str(data.get("image_path", "")),
        image_width=int(data["image_width"]),
        image_height=int(data["image_height"]),
        objects=objects,
        model_info=dict(data.get("model_info", {})),
    )
