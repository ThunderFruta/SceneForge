from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from ObjectEnrichment.types import (
    EdgeEvidence,
    FusedState,
    GeometryEvidence,
    MeshEvidence,
    ObjectEnrichment,
    WireframeEvidence,
)
from OutputWriter.evidence_overlay import write_evidence_overlay
from OutputWriter.evidence_overlay import _selected_overlay_label


def write_reports(root: Path) -> tuple[Path, Path, Path]:
    image_path = root / "image.png"
    detections_path = root / "detections.json"
    enrich_root = root / "enrich"
    object_root = enrich_root / "objects" / "01"
    enrichment_path = enrich_root / "object_enrichment.json"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    object_root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), (30, 30, 30)).save(image_path)
    Image.new("L", (32, 32), 0).save(enrich_root / "edge_map.png")
    Image.new("L", (16, 16), 255).save(object_root / "mask.png")
    (object_root / "crop_metadata.json").write_text(
        json.dumps({"crop_box_xyxy": [8, 8, 24, 24]}),
        encoding="utf-8",
    )
    (object_root / "wireframe.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "lines": [[0, 0, 15, 15, 1.0]],
            }
        ),
        encoding="utf-8",
    )
    (object_root / "mesh_candidate.obj").write_text("v 0 0 0\nv 1 0 0\n", encoding="utf-8")
    detections_path.write_text(
        json.dumps(
            {
                "image_path": str(image_path),
                "image_width": 32,
                "image_height": 32,
                "model_info": {},
                "objects": [
                    {
                        "id": 1,
                        "bbox_xyxy": [8, 8, 24, 24],
                        "mask_polygon": [[8, 8], [24, 8], [24, 24], [8, 24]],
                        "detector_label": "box",
                        "detector_confidence": 0.9,
                        "primitive_label": "unknown",
                        "primitive_confidence": 0.0,
                        "primitive_label_source": "unassigned",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    enrichment_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "image_path": str(image_path),
                "depth_path": str(root / "depth.png"),
                "detections_path": str(detections_path),
                "model_info": {},
                "objects": [
                    {
                        "id": 1,
                        "status": "ok",
                        "error": None,
                        "original_detector_label": "box",
                        "detector_confidence": 0.9,
                        "paths": {
                            "crop_metadata": "objects/01/crop_metadata.json",
                            "wireframe_json": "objects/01/wireframe.json",
                            "mesh_candidate": "objects/01/mesh_candidate.obj",
                        },
                        "edge": {"status": "ok", "boundary_agreement": 1.0, "edge_density": 0.1},
                        "wireframe": {"status": "ok", "line_count": 1, "junction_count": 2},
                        "mesh": {"status": "ok", "path": "objects/01/mesh_candidate.obj", "reason": None},
                        "geometry": {
                            "schema_version": 1,
                            "selected_label": "box",
                            "confidence": 0.8,
                            "candidate_scores": {"box": 0.8},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return image_path, detections_path, enrichment_path


def test_evidence_overlay_draws_combined_artifact(tmp_path: Path) -> None:
    image_path, detections_path, enrichment_path = write_reports(tmp_path)
    output_path = tmp_path / "evidence_overlay.png"

    write_evidence_overlay(
        image_path=image_path,
        detections_path=detections_path,
        enrichment_path=enrichment_path,
        output_path=output_path,
    )

    assert output_path.is_file()
    output = Image.open(output_path).convert("RGB")
    assert output.size == (32, 32)
    assert output.getpixel((8, 8)) != (30, 30, 30)


def test_selected_overlay_label_prefers_fused_state_when_contract_exists() -> None:
    obj = ObjectEnrichment(
        id=1,
        status="ok",
        error=None,
        original_detector_label="box",
        detector_confidence=0.9,
        paths={},
        edge=EdgeEvidence(status="not_available", boundary_agreement=0.0, edge_density=0.0),
        wireframe=WireframeEvidence(),
        mesh=MeshEvidence(status="missing", path=None, reason=None),
        geometry=GeometryEvidence(selected_label="box", confidence=0.6, candidate_scores={"box": 0.6}),
        fused_state=FusedState(
            fused_label="cylinder",
            fused_confidence=0.93,
            fused_contributions={"detector": {"status": "ok"}},
            needs_review=False,
            needs_review_reason=[],
        ),
        fused_label="cylinder",
        fused_confidence=0.93,
        fused_contributions={"detector": {"status": "ok"}},
        needs_review=False,
    )

    label, score = _selected_overlay_label(obj)
    assert label == "cylinder"
    assert score == 0.93
