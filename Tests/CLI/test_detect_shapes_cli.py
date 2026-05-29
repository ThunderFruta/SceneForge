from __future__ import annotations

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
