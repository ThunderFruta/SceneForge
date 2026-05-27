from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import Mock

import pytest
from PIL import Image

from PrimitiveFitting.pipeline import run_primitive_fitting


def write_detection_report(path: Path, image_path: Path, width: int, height: int, objects: list[dict]) -> None:
    path.write_text(
        json.dumps(
            {
                "image_path": str(image_path),
                "image_width": width,
                "image_height": height,
                "model_info": {"detector_backend": "fake"},
                "objects": objects,
            }
        ),
        encoding="utf-8",
    )


def write_enrichment_report(path: Path, image_path: Path, depth_path: Path, detections_path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "image_path": str(image_path),
                "depth_path": str(depth_path),
                "detections_path": str(detections_path),
                "model_info": {},
                "objects": [
                    {
                        "id": 1,
                        "status": "ok",
                        "error": None,
                        "original_detector_label": "box",
                        "detector_confidence": 0.9,
                        "paths": {},
                        "edge": {"status": "not_available", "boundary_agreement": 0.0, "edge_density": 0.0},
                        "wireframe": {"status": "not_available", "line_count": 0, "junction_count": 0},
                        "mesh": {"status": "missing", "path": None, "reason": None},
                        "geometry": {
                            "schema_version": 1,
                            "selected_label": "box",
                            "confidence": 0.66,
                            "candidate_scores": {"box": 0.66, "sphere": 0.2},
                        },
                        "fused_state": {
                            "fused_label": "cylinder",
                            "fused_confidence": 0.91,
                            "fused_contributions": {
                                "detector": {
                                    "status": "ok",
                                    "selected_label": "box",
                                    "selected_score": 0.9,
                                    "label_scores": {"box": 0.9},
                                    "evidence": {"detector_label": "box", "detector_confidence": 0.9},
                                },
                                "depth": {
                                    "status": "ok",
                                    "selected_label": "cylinder",
                                    "selected_score": 0.75,
                                    "label_scores": {"cylinder": 0.75, "box": 0.1},
                                    "evidence": {"geometry_label": "box", "geometry_confidence": 0.66},
                                },
                                "fusion": {
                                    "label_scores": {"cylinder": 0.8, "box": 0.4},
                                    "weights": {"detector": 0.2, "depth": 0.36},
                                    "active_modalities": ["detector", "depth"],
                                },
                            },
                            "needs_review": False,
                            "needs_review_reason": [],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def write_fused_enrichment_report(path: Path, image_path: Path, depth_path: Path, detections_path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "image_path": str(image_path),
                "depth_path": str(depth_path),
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
                            "rgb_crop": "objects/01/rgb_crop.png",
                            "mask": "objects/01/mask.png",
                            "depth_crop": "objects/01/depth_crop.png",
                            "edge_crop": "objects/01/edge_crop.png",
                            "crop_metadata": "objects/01/crop_metadata.json",
                            "wireframe_crop": None,
                            "wireframe_json": None,
                            "mesh_candidate": "objects/01/mesh_candidate.obj",
                            "evidence_stack": "objects/01/evidence_stack.png",
                        },
                        "edge": {"status": "ok", "boundary_agreement": 0.6, "edge_density": 0.2},
                        "wireframe": {"status": "ok", "line_count": 3, "junction_count": 1},
                        "mesh": {"status": "ok", "path": "objects/01/mesh_candidate.obj", "reason": None},
                        "geometry": {"schema_version": 1, "selected_label": "cone", "confidence": 0.55, "candidate_scores": {"cone": 0.55}},
                        "fused_state": {
                            "fused_label": "cylinder",
                            "fused_confidence": 0.91,
                            "fused_contributions": {
                                "detector": {
                                    "status": "ok",
                                    "selected_label": "box",
                                    "selected_score": 0.9,
                                    "label_scores": {"box": 0.9},
                                },
                                "depth": {
                                    "status": "ok",
                                    "selected_label": "cylinder",
                                    "selected_score": 0.72,
                                    "label_scores": {"cylinder": 0.72},
                                },
                                "edge": {
                                    "status": "ok",
                                    "selected_label": "cylinder",
                                    "selected_score": 0.66,
                                    "label_scores": {"cylinder": 0.66},
                                },
                                "wireframe": {
                                    "status": "ok",
                                    "selected_label": "cylinder",
                                    "selected_score": 0.64,
                                    "label_scores": {"cylinder": 0.64},
                                },
                                "mesh": {"status": "ok", "selected_label": "sphere", "selected_score": 0.81, "label_scores": {"sphere": 0.81}},
                                "fusion": {
                                    "label_scores": {"cylinder": 0.8, "box": 0.4},
                                    "weights": {"detector": 0.20, "depth": 0.36},
                                    "active_modalities": ["detector", "depth", "edge", "wireframe", "mesh"],
                                },
                            },
                            "needs_review": False,
                            "needs_review_reason": [],
                        },
                        "fused_label": "cylinder",
                        "fused_confidence": 0.91,
                        "fused_contributions": {},
                        "needs_review": False,
                        "needs_review_reason": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_primitive_fitting_writes_report_and_blend_for_fake_detection(tmp_path: Path) -> None:
    if shutil.which("blender") is None:
        pytest.skip("Blender is not installed")

    image_path = tmp_path / "image.png"
    depth_path = tmp_path / "depth.png"
    detections_path = tmp_path / "detections.json"
    output_dir = tmp_path / "out"
    Image.new("RGB", (32, 32), (60, 70, 80)).save(image_path)
    Image.new("L", (32, 32), 220).save(depth_path)
    write_detection_report(
        detections_path,
        image_path,
        32,
        32,
        [
            {
                "id": 1,
                "bbox_xyxy": [8, 8, 24, 24],
                "mask_polygon": [[8, 8], [24, 8], [24, 24], [8, 24]],
                "detector_label": "box",
                "detector_confidence": 0.9,
                "primitive_label": "box",
                "primitive_confidence": 0.8,
            }
        ],
    )

    report = run_primitive_fitting(
        image_path=image_path,
        depth_path=depth_path,
        detections_path=detections_path,
        output_dir=output_dir,
        camera_shift_x=0.03,
        camera_shift_y=-0.02,
    )

    data = json.loads((output_dir / "primitive_fits.json").read_text(encoding="utf-8"))
    assert report.objects[0].primitive_label == "box"
    assert data["camera"]["shift_x"] == 0.03
    assert data["camera"]["shift_y"] == -0.02
    assert data["camera"]["fusion_contract"]["scene"]["coordinate_system"] == "sceneforge_camera_v1"
    assert data["model_info"]["final_blend_layout"] == "camera"
    assert data["objects"][0]["primitive_label"] == "box"
    assert data["objects"][0]["primitive_label_source"] in {"detector", "depth_override"}
    assert "selected_fit_mode" in data["objects"][0]["fit_quality"]
    assert "bad_pixel_ratio_010" in data["objects"][0]["fit_quality"]
    assert (output_dir / "fit_overlay.png").is_file()
    assert (output_dir / "fitted_scene.blend").is_file()
    assert not (output_dir / "fitted_scene_layout.blend").exists()
    assert (output_dir / "depth_check" / "depth_check.json").is_file()
    assert (output_dir / "depth_check" / "depth_check_side_by_side.png").is_file()
    assert (output_dir / "metrics_summary.json").is_file()


def test_primitive_fitting_prefers_fused_state_when_enrichment_is_provided(tmp_path: Path) -> None:
    if shutil.which("blender") is None:
        pytest.skip("Blender is not installed")

    image_path = tmp_path / "image.png"
    depth_path = tmp_path / "depth.png"
    detections_path = tmp_path / "detections.json"
    enrichment_path = tmp_path / "object_enrichment.json"
    output_dir = tmp_path / "out"
    Image.new("RGB", (32, 32), (60, 70, 80)).save(image_path)
    Image.new("L", (32, 32), 220).save(depth_path)
    write_detection_report(
        detections_path,
        image_path,
        32,
        32,
        [
            {
                "id": 1,
                "bbox_xyxy": [8, 8, 24, 24],
                "mask_polygon": [[8, 8], [24, 8], [24, 24], [8, 24]],
                "detector_label": "box",
                "detector_confidence": 0.9,
                "primitive_label": "box",
                "primitive_confidence": 0.8,
            }
        ],
    )
    write_enrichment_report(enrichment_path, image_path, depth_path, detections_path)

    report = run_primitive_fitting(
        image_path=image_path,
        depth_path=depth_path,
        detections_path=detections_path,
        enrichment_path=enrichment_path,
        output_dir=output_dir,
        camera_shift_x=0.01,
        camera_shift_y=0.0,
    )

    data = json.loads((output_dir / "primitive_fits.json").read_text(encoding="utf-8"))
    assert report.objects[0].primitive_label == "cylinder"
    assert report.objects[0].primitive_label_source == "fused"
    assert data["objects"][0]["primitive_label"] == "cylinder"
    assert data["objects"][0]["primitive_label_source"] == "fused"
    assert data["objects"][0]["fit_quality"]["fused_label"] == "cylinder"
    assert data["objects"][0]["fit_quality"]["label_source"] == "fused"


def test_primitive_fitting_uses_fused_contract_with_no_mesh_dominance_and_writes_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "image.png"
    depth_path = tmp_path / "depth.png"
    detections_path = tmp_path / "detections.json"
    enrichment_path = tmp_path / "object_enrichment.json"
    output_dir = tmp_path / "out"
    Image.new("RGB", (48, 48), (60, 70, 80)).save(image_path)
    Image.new("L", (48, 48), 220).save(depth_path)
    write_detection_report(
        detections_path,
        image_path,
        48,
        48,
        [
            {
                "id": 1,
                "bbox_xyxy": [16, 16, 32, 32],
                "mask_polygon": [[16, 16], [32, 16], [32, 32], [16, 32]],
                "detector_label": "box",
                "detector_confidence": 0.9,
                "primitive_label": "box",
                "primitive_confidence": 0.8,
            }
        ],
    )
    write_fused_enrichment_report(enrichment_path, image_path, depth_path, detections_path)

    def fake_export(**kwargs) -> None:
        Path(kwargs["output_path"]).write_bytes(b"blend")

    def fake_depth(
        source_depth_path: str | Path,
        fitted_blend_path: str | Path,
        output_dir: str | Path,
        near_depth: float,
        far_depth: float,
        blender_executable: str = "blender",
        detections=None,
    ) -> dict:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "depth_check.json").write_text(
            json.dumps({"objects": []}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {
            "objects": [
                {
                    "id": 1,
                    "depth_mae": 0.02,
                    "depth_rmse": 0.01,
                    "bad_pixel_ratio_010": 0.0,
                    "mask_pixel_count": 256,
                }
            ]
        }
    # keep callable history
    fake_export = Mock(side_effect=fake_export)
    fake_depth = Mock(side_effect=fake_depth)
    monkeypatch.setattr("PrimitiveFitting.pipeline.export_fit_report_to_blend", fake_export)
    monkeypatch.setattr("PrimitiveFitting.pipeline.write_depth_check", fake_depth)

    report = run_primitive_fitting(
        image_path=image_path,
        depth_path=depth_path,
        detections_path=detections_path,
        enrichment_path=enrichment_path,
        output_dir=output_dir,
    )

    data = json.loads((output_dir / "primitive_fits.json").read_text(encoding="utf-8"))
    fused_payload = json.loads(enrichment_path.read_text(encoding="utf-8"))
    assert report.objects[0].primitive_label == "cylinder"
    assert report.objects[0].primitive_label_source == "fused"
    assert data["objects"][0]["primitive_label"] in {"sphere", "cylinder", "cone", "box", "plane", "unknown"}
    assert data["objects"][0]["primitive_label"] == "cylinder"
    assert data["objects"][0]["primitive_label_source"] == "fused"
    assert data["objects"][0]["fit_quality"]["label_source"] == "fused"
    assert data["objects"][0]["fit_quality"]["fused_label"] == data["objects"][0]["primitive_label"]
    assert data["objects"][0]["primitive_label"] != fused_payload["objects"][0]["fused_state"]["fused_contributions"]["mesh"]["selected_label"]
    assert data["objects"][0]["fit_quality"]["geometry_selected_label"] == "cone"
    assert data["objects"][0]["fit_quality"]["mesh_status"] == "ok"
    assert not output_dir.joinpath("fitted_scene_layout.blend").exists()
    assert not output_dir.joinpath("fitted_scene_camera_space.blend").exists()
    assert (output_dir / "fit_overlay.png").is_file()
    assert (output_dir / "primitive_fits.json").is_file()
    assert (output_dir / "fitted_scene.blend").is_file()
    assert (output_dir / "depth_check" / "depth_check.json").is_file()
    assert fake_export.call_count == 2
    assert fake_depth.call_count == 1


def test_primitive_fitting_empty_detections_writes_empty_scene(tmp_path: Path) -> None:
    if shutil.which("blender") is None:
        pytest.skip("Blender is not installed")

    image_path = tmp_path / "image.png"
    depth_path = tmp_path / "depth.png"
    detections_path = tmp_path / "detections.json"
    output_dir = tmp_path / "out"
    Image.new("RGB", (16, 16), "white").save(image_path)
    Image.new("L", (16, 16), 128).save(depth_path)
    write_detection_report(detections_path, image_path, 16, 16, [])

    run_primitive_fitting(
        image_path=image_path,
        depth_path=depth_path,
        detections_path=detections_path,
        output_dir=output_dir,
    )

    assert json.loads((output_dir / "primitive_fits.json").read_text(encoding="utf-8"))["objects"] == []
    assert (output_dir / "fitted_scene.blend").is_file()
    assert not (output_dir / "fitted_scene_layout.blend").exists()


def test_primitive_fitting_rejects_malformed_detections(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    depth_path = tmp_path / "depth.png"
    detections_path = tmp_path / "detections.json"
    Image.new("RGB", (16, 16), "white").save(image_path)
    Image.new("L", (16, 16), 128).save(depth_path)
    detections_path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed detections.json"):
        run_primitive_fitting(
            image_path=image_path,
            depth_path=depth_path,
            detections_path=detections_path,
            output_dir=tmp_path / "out",
        )


def test_primitive_fitting_rejects_empty_enrichment_for_non_empty_detections(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    depth_path = tmp_path / "depth.png"
    detections_path = tmp_path / "detections.json"
    enrichment_path = tmp_path / "object_enrichment.json"
    Image.new("RGB", (16, 16), "white").save(image_path)
    Image.new("L", (16, 16), 128).save(depth_path)
    write_detection_report(
        detections_path,
        image_path,
        16,
        16,
        [
            {
                "id": 2,
                "bbox_xyxy": [2, 2, 10, 10],
                "mask_polygon": [[2, 2], [10, 2], [10, 10], [2, 10]],
                "detector_label": "box",
                "detector_confidence": 0.9,
                "primitive_label": "unknown",
                "primitive_confidence": 0.0,
                "primitive_label_source": "unassigned",
            }
        ],
    )
    enrichment_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "image_path": str(image_path),
                "depth_path": str(depth_path),
                "detections_path": str(detections_path),
                "model_info": {},
                "objects": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no objects for non-empty detections"):
        run_primitive_fitting(
            image_path=image_path,
            depth_path=depth_path,
            detections_path=detections_path,
            enrichment_path=enrichment_path,
            output_dir=tmp_path / "out",
        )


def test_primitive_fitting_rejects_detection_enrichment_id_mismatch(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    depth_path = tmp_path / "depth.png"
    detections_path = tmp_path / "detections.json"
    enrichment_path = tmp_path / "object_enrichment.json"
    Image.new("RGB", (16, 16), "white").save(image_path)
    Image.new("L", (16, 16), 128).save(depth_path)
    write_detection_report(
        detections_path,
        image_path,
        16,
        16,
        [
            {
                "id": 2,
                "bbox_xyxy": [2, 2, 10, 10],
                "mask_polygon": [[2, 2], [10, 2], [10, 10], [2, 10]],
                "detector_label": "box",
                "detector_confidence": 0.9,
                "primitive_label": "unknown",
                "primitive_confidence": 0.0,
                "primitive_label_source": "unassigned",
            }
        ],
    )
    enrichment_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "image_path": str(image_path),
                "depth_path": str(depth_path),
                "detections_path": str(detections_path),
                "model_info": {},
                "objects": [
                    {
                        "id": 9,
                        "status": "ok",
                        "error": None,
                        "original_detector_label": "box",
                        "detector_confidence": 0.9,
                        "paths": {},
                        "edge": {"status": "not_available", "boundary_agreement": 0.0, "edge_density": 0.0},
                        "wireframe": {"status": "not_available", "line_count": 0, "junction_count": 0},
                        "mesh": {"status": "missing", "path": None, "reason": None},
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

    with pytest.raises(ValueError, match="ids do not match"):
        run_primitive_fitting(
            image_path=image_path,
            depth_path=depth_path,
            detections_path=detections_path,
            enrichment_path=enrichment_path,
            output_dir=tmp_path / "out",
        )
