from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import Image

from run import CliError, _require_pipeline_artifact, _run_pipeline_stage, build_parser


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


def test_cli_retired_rgb_yolo_backend_is_not_public(tmp_path: Path) -> None:
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
    assert "invalid choice: 'rgb-yolo'" in result.stderr


def test_cli_retired_depth_edge_backend_is_not_public(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    depth_path = tmp_path / "depth.png"
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
        str(tmp_path / "out"),
    )

    assert result.returncode == 2
    assert "invalid choice: 'depth-edge'" in result.stderr


def test_cli_retired_depth_edge_object_backend_is_not_public(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    depth_path = tmp_path / "depth.png"
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
        str(tmp_path / "out"),
    )

    assert result.returncode == 2
    assert "invalid choice: 'depth-edge-object'" in result.stderr


def test_process_image_parser_accepts_single_image_automation(tmp_path: Path) -> None:
    parser = build_parser()
    image_path = tmp_path / "input.jpg"
    Image.new("RGB", (8, 8), "white").save(image_path)

    args = parser.parse_args(
        [
            "process-image",
            "--image",
            str(image_path),
            "--output-root",
            str(tmp_path / "out"),
            "--empty-room-backend",
            "fake",
            "--vggt-backend",
            "fake",
            "--no-render-source-camera",
        ]
    )

    assert args.command == "process-image"
    assert args.image == str(image_path)
    assert args.output_root == str(tmp_path / "out")
    assert args.empty_room_backend == "fake"
    assert args.vggt_backend == "fake"
    assert args.render_source_camera is False


def test_texture_objects_parser_defaults_to_existing_objects() -> None:
    parser = build_parser()

    args = parser.parse_args(["texture-objects"])

    assert args.command == "texture-objects"
    assert args.objects == "Output/Latest/objects"
    assert args.texture_resolution == 512
    assert args.texture_views == 6
    assert args.texture_remesh is True


def test_pipeline_stage_raises_on_nonzero_status() -> None:
    try:
        _run_pipeline_stage("broken-stage", lambda _args: 3, object())
    except CliError as exc:
        assert "broken-stage" in str(exc)
        assert "exit code 3" in str(exc)
    else:
        raise AssertionError("Expected nonzero pipeline stage to raise.")


def test_pipeline_artifact_check_names_missing_producer(tmp_path: Path) -> None:
    missing = tmp_path / "objects_vggt" / "vggt_points.npy"

    try:
        _require_pipeline_artifact(missing, "run-vggt")
    except CliError as exc:
        assert "run-vggt" in str(exc)
        assert str(missing) in str(exc)
    else:
        raise AssertionError("Expected missing pipeline artifact to raise.")



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
