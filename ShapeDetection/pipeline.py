from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from Input.Image.image_loader import load_rgb_image
from OutputWriter.overlay import write_overlay
from OutputWriter.report_writer import write_report
from SceneGeometry.coordinate_contract import camera_fusion_contract, load_fusion_contract_from_camera_metadata
from Segmentation.proposal_quality import is_open_vocab_model_info, summarize_open_vocab_proposals
from ShapeDetection.report import DetectionReport, ObjectShapeDetection


def run_shape_detection(
    image_path: str | Path,
    output_dir: str | Path,
    segmenter,
    classifier,
    model_info: dict,
) -> DetectionReport:
    resolved_image_path = Path(image_path)
    image = load_rgb_image(resolved_image_path)
    segments = segmenter.detect(image)
    proposal_quality = None
    if is_open_vocab_model_info(model_info):
        proposal_quality = summarize_open_vocab_proposals(segments, image_width=image.width, image_height=image.height)

    objects: list[ObjectShapeDetection] = []
    for index, segment in enumerate(segments, start=1):
        prediction = classifier.classify(image, segment)
        objects.append(
            ObjectShapeDetection(
                id=index,
                bbox_xyxy=segment.bbox_xyxy,
                mask_polygon=segment.mask_polygon,
                detector_label=segment.detector_label,
                detector_confidence=segment.detector_confidence,
                primitive_label=prediction.label,
                primitive_confidence=prediction.confidence,
                primitive_label_source=prediction.source,
            )
        )

    model_info_with_time = dict(model_info)
    if proposal_quality is not None:
        model_info_with_time["proposal_quality"] = proposal_quality
    model_info_with_time["fusion_contract"] = load_source_fusion_contract(resolved_image_path, image.width, image.height)
    model_info_with_time["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    report = DetectionReport(
        image_path=str(resolved_image_path),
        image_width=image.width,
        image_height=image.height,
        objects=objects,
        model_info=model_info_with_time,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    write_report(report, output_path / "detections.json")
    if proposal_quality is not None:
        (output_path / "proposal_quality.json").write_text(
            json.dumps(proposal_quality, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    write_overlay(image, objects, output_path / "overlay.png")
    return report


def load_source_fusion_contract(image_path: Path, width: int, height: int) -> dict:
    camera_path = image_path.parent / "camera.json"
    if camera_path.is_file():
        try:
            return load_fusion_contract_from_camera_metadata(json.loads(camera_path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            pass
    return camera_fusion_contract(image_width=width, image_height=height)
