from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "run.py"), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_fake_backend_is_not_public(tmp_path: Path) -> None:
    result = run_cli(
        "detect-shapes",
        "--backend",
        "fake",
        "--image",
        str(tmp_path / "input.png"),
        "--output",
        str(tmp_path / "out"),
    )

    assert result.returncode == 2
    assert "invalid choice: 'fake'" in result.stderr


def test_cli_missing_real_model_paths_fail_clearly(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    Image.new("RGB", (8, 8), "white").save(image_path)

    result = run_cli(
        "detect-shapes",
        "--backend",
        "rgb-yolo",
        "--image",
        str(image_path),
        "--output",
        str(tmp_path / "out"),
    )

    assert result.returncode == 2
    assert "--detector-weights is required" in result.stderr


def test_cli_depth_edge_backend_does_not_require_yolo_weights(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    depth_path = tmp_path / "depth.png"
    output_dir = tmp_path / "out"
    Image.new("RGB", (32, 32), "black").save(image_path)
    Image.new("L", (32, 32), 180).save(depth_path)

    result = run_cli(
        "detect-shapes",
        "--backend",
        "depth-edge",
        "--image",
        str(image_path),
        "--depth",
        str(depth_path),
        "--output",
        str(output_dir),
    )

    assert result.returncode == 0
    data = json.loads((output_dir / "detections.json").read_text(encoding="utf-8"))
    assert data["model_info"]["detector_backend"] == "depth-edge-instance-scaffold"
    assert data["model_info"]["legacy_yolo"] is False
    assert data["model_info"]["primitive_label_policy"] == "geometry_fitting_downstream"
    assert data["model_info"]["detector_backend_info"]["proposal_only"] is True
    assert data["model_info"]["detector_backend_info"]["output_contract"] == "instance_masks_only"


def test_cli_depth_edge_object_backend_reports_object_detector(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    depth_path = tmp_path / "depth.png"
    output_dir = tmp_path / "out"
    Image.new("RGB", (32, 32), "black").save(image_path)
    Image.new("L", (32, 32), 180).save(depth_path)

    result = run_cli(
        "detect-shapes",
        "--backend",
        "depth-edge-object",
        "--image",
        str(image_path),
        "--depth",
        str(depth_path),
        "--output",
        str(output_dir),
    )

    assert result.returncode == 0
    data = json.loads((output_dir / "detections.json").read_text(encoding="utf-8"))
    assert data["model_info"]["detector_backend"] == "depth-edge-object-detector"
    assert data["model_info"]["detector_input_channels"] == ["rgb", "depth", "edge"]
    assert data["model_info"]["classifier_backend"] == "depth-geometry-weak"


def test_cli_detector_label_source_does_not_require_clip_model_dir(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    weights_path = tmp_path / "weights.pt"
    Image.new("RGB", (8, 8), "white").save(image_path)
    weights_path.write_text("not real weights", encoding="utf-8")

    result = run_cli(
        "detect-shapes",
        "--backend",
        "rgb-yolo",
        "--image",
        str(image_path),
        "--detector-weights",
        str(weights_path),
        "--primitive-source",
        "detector-label",
        "--output",
        str(tmp_path / "out"),
    )

    assert "--clip-model-dir is required" not in result.stderr


def test_cli_fit_primitives_writes_outputs(tmp_path: Path) -> None:
    if shutil.which("blender") is None:
        return

    image_path = tmp_path / "input.png"
    depth_path = tmp_path / "depth.png"
    detections_path = tmp_path / "detections.json"
    output_dir = tmp_path / "fit"
    Image.new("RGB", (32, 32), "white").save(image_path)
    Image.new("L", (32, 32), 220).save(depth_path)
    detections_path.write_text(
        json.dumps(
            {
                "image_path": str(image_path),
                "image_width": 32,
                "image_height": 32,
                "model_info": {"detector_backend": "fake"},
                "objects": [
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
            }
        ),
        encoding="utf-8",
    )

    result = run_cli(
        "fit-primitives",
        "--image",
        str(image_path),
        "--depth",
        str(depth_path),
        "--detections",
        str(detections_path),
        "--output",
        str(output_dir),
    )

    assert result.returncode == 0
    assert (output_dir / "primitive_fits.json").is_file()
    assert (output_dir / "fit_overlay.png").is_file()
    assert (output_dir / "fitted_scene.blend").is_file()
    assert not (output_dir / "fitted_scene_layout.blend").exists()


def test_cli_compare_metrics_writes_summary(tmp_path: Path) -> None:
    original = tmp_path / "original" / "depth"
    generated = tmp_path / "generated" / "depth"
    original.mkdir(parents=True)
    generated.mkdir(parents=True)
    Image.new("RGB", (8, 8), (255, 255, 255)).save(original / "pos_z.png")
    Image.new("RGB", (8, 8), (0, 0, 0)).save(generated / "pos_z.png")
    output_dir = tmp_path / "metrics"

    result = run_cli(
        "compare-metrics",
        "--original-metrics",
        str(tmp_path / "original"),
        "--generated-metrics",
        str(tmp_path / "generated"),
        "--output",
        str(output_dir),
    )

    assert result.returncode == 0
    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "comparison" / "metrics_comparison.csv").is_file()


def test_cli_sam3_missing_repo_fails_clearly(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    Image.new("RGB", (8, 8), "white").save(image_path)

    result = run_cli(
        "detect-shapes",
        "--backend",
        "sam3",
        "--image",
        str(image_path),
        "--sam3-repo-dir",
        str(tmp_path / "missing-sam3"),
        "--sam3-model-dir",
        str(tmp_path / "missing-model"),
        "--output",
        str(tmp_path / "out"),
    )

    assert result.returncode == 2
    assert "--sam3-repo-dir does not exist" in result.stderr
