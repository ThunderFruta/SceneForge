from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from EdgeDetection.no_edge_provider import NoEdgeProvider
from EdgeDetection.simple_edge_provider import SimpleEdgeProvider
from MeshReconstruction.no_mesh_provider import NoMeshProvider
from ObjectEnrichment.report_loader import load_enrichment_report
from ObjectEnrichment.pipeline import run_object_enrichment
from Tests.Fakes.providers import FakeMeshProvider, FakeWireframeProvider
from ObjectEnrichment.types import FUSED_LABELS


def write_detections(path: Path, image_path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "image_path": str(image_path),
                "image_width": 64,
                "image_height": 64,
                "model_info": {"detector_backend": "fake"},
                "objects": [
                    {
                        "id": 7,
                        "bbox_xyxy": [16, 16, 48, 48],
                        "mask_polygon": [[16, 16], [48, 16], [48, 48], [16, 48]],
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


def test_fake_enrichment_writes_object_evidence_pack(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    depth_path = tmp_path / "depth.png"
    detections_path = tmp_path / "detections.json"
    output_dir = tmp_path / "enrich"
    Image.new("RGB", (64, 64), "white").save(image_path)
    Image.new("L", (64, 64), 180).save(depth_path)
    write_detections(detections_path, image_path)

    report = run_object_enrichment(
        image_path=image_path,
        depth_path=depth_path,
        detections_path=detections_path,
        output_dir=output_dir,
        edge_provider=SimpleEdgeProvider(),
        mesh_provider=FakeMeshProvider(),
    )

    data = json.loads((output_dir / "object_enrichment.json").read_text(encoding="utf-8"))
    assert report.objects[0].id == 7
    assert data["objects"][0]["id"] == 7
    assert data["objects"][0]["original_detector_label"] == "box"
    assert data["objects"][0]["geometry"]["selected_label"] in {"box", "plane", "unknown", "sphere", "cylinder", "cone"}
    assert data["objects"][0]["wireframe"]["status"] == "not_available"
    assert data["objects"][0]["mesh"]["status"] == "ok"
    assert data["model_info"]["fusion_contract"]["scene"]["coordinate_system"] == "sceneforge_camera_v1"
    assert data["objects"][0]["paths"]["crop_metadata"] == "objects/01/crop_metadata.json"
    assert (output_dir / "edge_map.png").is_file()
    assert (output_dir / "objects" / "01" / "rgb_crop.png").is_file()
    assert (output_dir / "objects" / "01" / "mask.png").is_file()
    assert (output_dir / "objects" / "01" / "crop_metadata.json").is_file()
    assert (output_dir / "objects" / "01" / "mesh_candidate.obj").is_file()
    assert set(data["objects"][0]["fused_contributions"].keys()) == {
        "detector",
        "depth",
        "edge",
        "wireframe",
        "mesh",
        "fusion",
    }
    for modality in ("detector", "depth", "edge", "wireframe", "mesh"):
        scores = data["objects"][0]["fused_contributions"][modality]["label_scores"]
        assert set(scores) == set(FUSED_LABELS)


def test_fake_wireframe_provider_writes_object_wireframe_pack(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    depth_path = tmp_path / "depth.png"
    detections_path = tmp_path / "detections.json"
    output_dir = tmp_path / "enrich"
    Image.new("RGB", (64, 64), "white").save(image_path)
    Image.new("L", (64, 64), 180).save(depth_path)
    write_detections(detections_path, image_path)

    run_object_enrichment(
        image_path=image_path,
        depth_path=depth_path,
        detections_path=detections_path,
        output_dir=output_dir,
        edge_provider=SimpleEdgeProvider(),
        mesh_provider=FakeMeshProvider(),
        wireframe_provider=FakeWireframeProvider(),
    )

    data = json.loads((output_dir / "object_enrichment.json").read_text(encoding="utf-8"))
    wireframe = data["objects"][0]["wireframe"]
    assert wireframe["status"] == "ok"
    assert wireframe["line_count"] == 4
    assert wireframe["junction_count"] == 4
    assert data["objects"][0]["paths"]["wireframe_crop"] == "objects/01/wireframe_crop.png"
    assert data["objects"][0]["paths"]["wireframe_json"] == "objects/01/wireframe.json"
    assert (output_dir / "objects" / "01" / "wireframe_crop.png").is_file()
    assert (output_dir / "objects" / "01" / "wireframe.json").is_file()


def test_fusion_uses_assigned_primitive_label_when_detector_label_is_unknown(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    depth_path = tmp_path / "depth.png"
    detections_path = tmp_path / "detections.json"
    output_dir = tmp_path / "enrich"
    Image.new("RGB", (64, 64), "white").save(image_path)
    Image.new("L", (64, 64), 180).save(depth_path)
    detections_path.write_text(
        json.dumps(
            {
                "image_path": str(image_path),
                "image_width": 64,
                "image_height": 64,
                "model_info": {"detector_backend": "depth-edge-object-detector"},
                "objects": [
                    {
                        "id": 1,
                        "bbox_xyxy": [16, 16, 48, 48],
                        "mask_polygon": [[16, 16], [48, 16], [48, 48], [16, 48]],
                        "detector_label": "unknown",
                        "detector_confidence": 0.5,
                        "primitive_label": "box",
                        "primitive_confidence": 0.88,
                        "primitive_label_source": "depth_geometry_weak",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    run_object_enrichment(
        image_path=image_path,
        depth_path=depth_path,
        detections_path=detections_path,
        output_dir=output_dir,
        edge_provider=NoEdgeProvider(),
        mesh_provider=NoMeshProvider(),
    )

    data = json.loads((output_dir / "object_enrichment.json").read_text(encoding="utf-8"))
    detector = data["objects"][0]["fused_contributions"]["detector"]
    assert detector["selected_label"] == "box"
    assert detector["evidence"]["selected_evidence"] == "primitive_label"
    assert detector["evidence"]["primitive_label_source"] == "depth_geometry_weak"


def test_fusion_keeps_medium_confidence_primitive_label_on_close_disagreement(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    depth_path = tmp_path / "depth.png"
    detections_path = tmp_path / "detections.json"
    output_dir = tmp_path / "enrich"
    Image.new("RGB", (64, 64), "white").save(image_path)
    Image.new("L", (64, 64), 180).save(depth_path)
    detections_path.write_text(
        json.dumps(
            {
                "image_path": str(image_path),
                "image_width": 64,
                "image_height": 64,
                "model_info": {"detector_backend": "depth-edge-object-detector"},
                "objects": [
                    {
                        "id": 1,
                        "bbox_xyxy": [16, 16, 48, 48],
                        "mask_polygon": [[16, 16], [48, 16], [48, 48], [16, 48]],
                        "detector_label": "unknown",
                        "detector_confidence": 0.5,
                        "primitive_label": "cone",
                        "primitive_confidence": 0.84,
                        "primitive_label_source": "depth_geometry_weak",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    run_object_enrichment(
        image_path=image_path,
        depth_path=depth_path,
        detections_path=detections_path,
        output_dir=output_dir,
        edge_provider=NoEdgeProvider(),
        mesh_provider=NoMeshProvider(),
    )

    data = json.loads((output_dir / "object_enrichment.json").read_text(encoding="utf-8"))
    assert data["objects"][0]["fused_contributions"]["detector"]["selected_label"] == "cone"
    assert data["objects"][0]["fused_label"] == "cone"


def test_load_enrichment_report_normalizes_fused_state_shape(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "image_path": "input/image.png",
        "depth_path": "input/depth.png",
        "detections_path": "input/detections.json",
        "model_info": {},
        "objects": [
            {
                "id": 1,
                "status": "ok",
                "error": None,
                "original_detector_label": "box",
                "detector_confidence": 0.9,
                "paths": {},
                "edge": {"status": "ok", "boundary_agreement": 0.4, "edge_density": 0.2},
                "wireframe": {"status": "not_available", "line_count": 0, "junction_count": 0},
                "mesh": {"status": "missing", "path": None, "reason": None},
                "geometry": {"schema_version": 1, "selected_label": "box", "confidence": 0.77, "candidate_scores": {"box": 0.77}},
                "fused_state": {
                    "fused_label": "cylinder",
                    "fused_confidence": 0.9,
                    "fused_contributions": {
                        "depth": {
                            "selected_label": "cylinder",
                            "selected_score": 0.8,
                            "label_scores": {"cylinder": 0.8},
                        }
                    },
                    "needs_review": True,
                    "needs_review_reason": ["unit_test"],
                },
                "fused_label": "cylinder",
                "fused_confidence": 0.9,
                "fused_contributions": {"depth": {"selected_label": "cylinder", "selected_score": 0.8}},
                "needs_review": True,
                "needs_review_reason": ["unit_test"],
            }
        ],
    }
    report_path = tmp_path / "object_enrichment.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    report = load_enrichment_report(report_path)

    assert len(report.objects) == 1
    obj = report.objects[0]
    assert obj.fused_state.fused_label == "cylinder"
    assert obj.fused_state.fused_contributions["detector"]["label_scores"]["unknown"] == 0.0
    assert set(obj.fused_state.fused_contributions.keys()) == {"detector", "depth", "edge", "wireframe", "mesh", "fusion"}
    assert set(obj.fused_state.fused_contributions["fusion"]["label_scores"].keys()) == set(FUSED_LABELS)


def test_empty_detections_writes_empty_enrichment(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    depth_path = tmp_path / "depth.png"
    detections_path = tmp_path / "detections.json"
    output_dir = tmp_path / "enrich"
    Image.new("RGB", (16, 16), "white").save(image_path)
    Image.new("L", (16, 16), 180).save(depth_path)
    detections_path.write_text(
        json.dumps(
            {
                "image_path": str(image_path),
                "image_width": 16,
                "image_height": 16,
                "model_info": {},
                "objects": [],
            }
        ),
        encoding="utf-8",
    )

    run_object_enrichment(
        image_path=image_path,
        depth_path=depth_path,
        detections_path=detections_path,
        output_dir=output_dir,
        edge_provider=SimpleEdgeProvider(),
        mesh_provider=FakeMeshProvider(),
    )

    data = json.loads((output_dir / "object_enrichment.json").read_text(encoding="utf-8"))
    assert data["objects"] == []


def test_enrichment_orders_objects_by_numeric_id_and_uses_stable_folders(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    depth_path = tmp_path / "depth.png"
    detections_path = tmp_path / "detections.json"
    output_dir = tmp_path / "enrich"
    Image.new("RGB", (64, 64), "white").save(image_path)
    Image.new("L", (64, 64), 180).save(depth_path)
    detections_path.write_text(
        json.dumps(
            {
                "image_path": str(image_path),
                "image_width": 64,
                "image_height": 64,
                "model_info": {},
                "objects": [
                    {
                        "id": 9,
                        "bbox_xyxy": [34, 34, 54, 54],
                        "mask_polygon": [[34, 34], [54, 34], [54, 54], [34, 54]],
                        "detector_label": "sphere",
                        "detector_confidence": 0.8,
                        "primitive_label": "unknown",
                        "primitive_confidence": 0.0,
                        "primitive_label_source": "unassigned",
                    },
                    {
                        "id": 3,
                        "bbox_xyxy": [8, 8, 28, 28],
                        "mask_polygon": [[8, 8], [28, 8], [28, 28], [8, 28]],
                        "detector_label": "box",
                        "detector_confidence": 0.9,
                        "primitive_label": "unknown",
                        "primitive_confidence": 0.0,
                        "primitive_label_source": "unassigned",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    run_object_enrichment(
        image_path=image_path,
        depth_path=depth_path,
        detections_path=detections_path,
        output_dir=output_dir,
        edge_provider=SimpleEdgeProvider(),
        mesh_provider=FakeMeshProvider(),
    )

    data = json.loads((output_dir / "object_enrichment.json").read_text(encoding="utf-8"))
    assert [item["id"] for item in data["objects"]] == [3, 9]
    assert data["objects"][0]["paths"]["rgb_crop"] == "objects/01/rgb_crop.png"
    assert data["objects"][1]["paths"]["rgb_crop"] == "objects/02/rgb_crop.png"


def test_noop_providers_write_missing_unavailable_statuses(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    depth_path = tmp_path / "depth.png"
    detections_path = tmp_path / "detections.json"
    output_dir = tmp_path / "enrich"
    Image.new("RGB", (64, 64), "white").save(image_path)
    Image.new("L", (64, 64), 180).save(depth_path)
    write_detections(detections_path, image_path)

    run_object_enrichment(
        image_path=image_path,
        depth_path=depth_path,
        detections_path=detections_path,
        output_dir=output_dir,
        edge_provider=NoEdgeProvider(),
        mesh_provider=NoMeshProvider(),
    )

    data = json.loads((output_dir / "object_enrichment.json").read_text(encoding="utf-8"))
    obj = data["objects"][0]
    assert data["model_info"]["edge_backend"] == "none"
    assert data["model_info"]["mesh_backend"] == "none"
    assert obj["edge"]["status"] == "not_available"
    assert obj["mesh"]["status"] == "missing"
    assert obj["mesh"]["path"] is None
    assert obj["paths"]["mesh_candidate"] is None
    assert not (output_dir / "objects" / "01" / "mesh_candidate.obj").exists()
