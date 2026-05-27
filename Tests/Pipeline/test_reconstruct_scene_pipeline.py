from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
from PIL import Image

from run import (
    CliError,
    _prepare_latest_output,
    _replace_stage_output,
    _run_reconstruct_fit,
)


def reconstruct_args(**overrides):
    values = {
        "reference_blend": "Assets/Samples/shapes.blend",
        "output": Path("Latest"),
        "camera_name": None,
        "detector_backend": "depth-edge",
        "detector_weights": None,
        "edge_backend": "simple",
        "edge_model_dir": None,
        "mesh_backend": "none",
        "mesh_model_dir": None,
        "final_layout": "camera",
        "blender": "blender",
        "device": None,
        "seed": 20260525,
        "width": 640,
        "height": 640,
        "render_samples": 16,
        "near_depth": 1.0,
        "far_depth": 8.0,
        "edge_timeout_seconds": 120,
        "mesh_timeout_seconds": 180,
        "resume": False,
        "force": False,
        "no_archive": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_run_reconstruct_fit_writes_reconstruction_artifacts(tmp_path, monkeypatch) -> None:
    output_dir = tmp_path / "Latest"
    output_dir.mkdir()
    enrichment = output_dir / "enrich" / "object_enrichment.json"
    enrichment.parent.mkdir(parents=True)
    enrichment.write_text(
        json.dumps(
            {
                "schema_version": 1,
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
                            "mesh_candidate": None,
                            "evidence_stack": "objects/01/evidence_stack.png",
                        },
                        "edge": {"status": "ok", "boundary_agreement": 0.72, "edge_density": 0.22},
                        "wireframe": {"status": "ok", "line_count": 8, "junction_count": 3, "reason": None},
                        "mesh": {"status": "ok", "path": "objects/01/mesh_candidate.obj", "reason": None},
                        "geometry": {
                            "schema_version": 1,
                            "selected_label": "cone",
                            "confidence": 0.5,
                            "candidate_scores": {"cone": 0.5},
                        },
                        "fused_state": {
                            "fused_label": "cylinder",
                            "fused_confidence": 0.93,
                            "fused_contributions": {
                                "detector": {
                                    "status": "ok",
                                    "selected_label": "box",
                                    "selected_score": 0.8,
                                    "label_scores": {"box": 0.8},
                                },
                                "depth": {
                                    "status": "ok",
                                    "selected_label": "cylinder",
                                    "selected_score": 0.75,
                                    "label_scores": {"cylinder": 0.75},
                                },
                                "edge": {
                                    "status": "ok",
                                    "selected_label": "cylinder",
                                    "selected_score": 0.58,
                                    "label_scores": {"cylinder": 0.58, "box": 0.26},
                                },
                                "wireframe": {
                                    "status": "ok",
                                    "selected_label": "cylinder",
                                    "selected_score": 0.64,
                                    "label_scores": {"cylinder": 0.64},
                                },
                                "mesh": {"status": "ok", "selected_label": "sphere", "selected_score": 0.12, "label_scores": {"unknown": 0.0}},
                                "fusion": {
                                    "label_scores": {"cylinder": 0.78, "box": 0.46},
                                    "weights": {"detector": 0.20, "depth": 0.36},
                                    "active_modalities": ["detector", "depth", "edge", "wireframe", "mesh"],
                                },
                            },
                            "needs_review": False,
                            "needs_review_reason": [],
                        },
                        "fused_label": "cylinder",
                        "fused_confidence": 0.93,
                        "fused_contributions": {},
                        "needs_review": False,
                        "needs_review_reason": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_run_primitive_fitting(*, output_dir: Path, **kwargs):
        out = Path(output_dir)
        enrichment_payload = json.loads(Path(kwargs["enrichment_path"]).read_text(encoding="utf-8"))
        assert len(enrichment_payload["objects"]) == 1
        obj = enrichment_payload["objects"][0]
        fused = obj["fused_state"]
        assert fused["fused_label"] == "cylinder"
        assert "detector" in fused["fused_contributions"]
        assert fused["fused_contributions"]["fusion"]["label_scores"]["cylinder"] > fused["fused_contributions"]["fusion"]["label_scores"]["box"]
        assert obj["mesh"]["status"] == "ok"
        assert kwargs["detections_path"].name == "detections.json"
        out.mkdir(parents=True, exist_ok=True)
        out.joinpath("primitive_fits.json").write_text(
            json.dumps(
                {
                    "objects": [
                        {
                            "id": 1,
                            "primitive_label": "cylinder",
                            "primitive_label_source": "fused",
                            "fit_quality": {
                                "label_source": "fused",
                                "fused_label": "cylinder",
                                "needs_review": False,
                                "geometry_selected_label": "cone",
                                "mesh_status": "ok",
                                "mesh_candidate_path": "objects/01/mesh_candidate.obj",
                            },
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        Image.new("RGB", (16, 16), (10, 20, 30)).save(out / "fit_overlay.png")
        out.joinpath("fitted_scene.blend").write_bytes(b"blend")

    monkeypatch.setattr("run.run_primitive_fitting", fake_run_primitive_fitting)
    args_final = reconstruct_args(output=str(output_dir), blender=str(tmp_path / "missing-blender"))

    _run_reconstruct_fit(args_final, output_dir, fov_degrees=70.0)

    fit_dir = output_dir / "fit"
    data = json.loads((fit_dir / "primitive_fits.json").read_text(encoding="utf-8"))
    enrichment_payload = json.loads((output_dir / "enrich" / "object_enrichment.json").read_text(encoding="utf-8"))
    fused_object = enrichment_payload["objects"][0]

    assert data["objects"][0]["primitive_label"] == data["objects"][0]["fit_quality"]["fused_label"] == "cylinder"
    assert data["objects"][0]["primitive_label"] in {"sphere", "cylinder", "cone", "box", "plane", "unknown"}
    assert data["objects"][0]["fit_quality"]["label_source"] == "fused"
    assert data["objects"][0]["primitive_label_source"] == "fused"
    assert data["objects"][0]["fit_quality"]["mesh_status"] == "ok"
    assert data["objects"][0]["fit_quality"]["fused_label"] != fused_object["fused_state"]["fused_contributions"]["mesh"]["selected_label"]
    assert data["objects"][0]["fit_quality"]["geometry_selected_label"] == "cone"
    assert (fit_dir / "fit_overlay.png").is_file()
    assert (fit_dir / "primitive_fits.json").is_file()
    assert not (fit_dir / "fitted_scene.blend").is_file()
    assert (output_dir / "fitted_scene.blend").is_file()


def test_prepare_latest_archives_existing_output_before_new_run(tmp_path: Path) -> None:
    latest = tmp_path / "Latest"
    latest.mkdir()
    (latest / "old.txt").write_text("old", encoding="utf-8")

    _prepare_latest_output(reconstruct_args(), latest)

    archive_dirs = sorted((tmp_path / "Archive").iterdir())
    assert len(archive_dirs) == 1
    assert (archive_dirs[0] / "old.txt").read_text(encoding="utf-8") == "old"
    assert (latest / "run_manifest.json").is_file()


def test_prepare_latest_no_archive_fails_on_non_empty_latest(tmp_path: Path) -> None:
    latest = tmp_path / "Latest"
    latest.mkdir()
    (latest / "old.txt").write_text("old", encoding="utf-8")

    with pytest.raises(CliError, match="not empty"):
        _prepare_latest_output(reconstruct_args(no_archive=True), latest)

    assert (latest / "old.txt").is_file()


def test_prepare_latest_resume_rejects_stale_manifest(tmp_path: Path) -> None:
    latest = tmp_path / "Latest"
    latest.mkdir()
    (latest / "run_manifest.json").write_text(
        json.dumps({"schema_version": 1, "seed": 1}),
        encoding="utf-8",
    )

    with pytest.raises(CliError, match="run_manifest.json does not match"):
        _prepare_latest_output(reconstruct_args(resume=True), latest)


def test_stage_output_replaces_only_after_temp_success(tmp_path: Path) -> None:
    final = tmp_path / "detect"
    final.mkdir()
    (final / "old.txt").write_text("old", encoding="utf-8")
    temp = tmp_path / ".tmp_detect"
    temp.mkdir()
    (temp / "new.txt").write_text("new", encoding="utf-8")

    _replace_stage_output(temp, final)

    assert not temp.exists()
    assert not (final / "old.txt").exists()
    assert (final / "new.txt").read_text(encoding="utf-8") == "new"
