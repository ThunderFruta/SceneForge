from __future__ import annotations

import json
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


def write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    image_path = tmp_path / "image.png"
    depth_path = tmp_path / "depth.png"
    detections_path = tmp_path / "detections.json"
    Image.new("RGB", (16, 16), "white").save(image_path)
    Image.new("L", (16, 16), 128).save(depth_path)
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
    return image_path, depth_path, detections_path


def test_enrich_lightweight_providers_require_no_model_paths(tmp_path: Path) -> None:
    image_path, depth_path, detections_path = write_inputs(tmp_path)
    output_dir = tmp_path / "enrich"

    result = run_cli(
        "enrich-objects",
        "--image",
        str(image_path),
        "--depth",
        str(depth_path),
        "--detections",
        str(detections_path),
        "--edge-backend",
        "simple",
        "--mesh-backend",
        "none",
        "--wireframe-backend",
        "none",
        "--output",
        str(output_dir),
    )

    assert result.returncode == 0
    assert (output_dir / "object_enrichment.json").is_file()


def test_enrich_fake_backends_are_not_public(tmp_path: Path) -> None:
    image_path, depth_path, detections_path = write_inputs(tmp_path)

    result = run_cli(
        "enrich-objects",
        "--image",
        str(image_path),
        "--depth",
        str(depth_path),
        "--detections",
        str(detections_path),
        "--edge-backend",
        "fake",
        "--output",
        str(tmp_path / "enrich"),
    )

    assert result.returncode == 2
    assert "invalid choice: 'fake'" in result.stderr


def test_enrich_missing_real_wireframe_model_fails_before_output(tmp_path: Path) -> None:
    image_path, depth_path, detections_path = write_inputs(tmp_path)
    output_dir = tmp_path / "enrich"

    result = run_cli(
        "enrich-objects",
        "--image",
        str(image_path),
        "--depth",
        str(depth_path),
        "--detections",
        str(detections_path),
        "--edge-backend",
        "simple",
        "--mesh-backend",
        "none",
        "--wireframe-backend",
        "hawp",
        "--wireframe-model-dir",
        str(tmp_path / "missing-wireframe"),
        "--output",
        str(output_dir),
    )

    assert result.returncode == 2
    assert "--wireframe-model-dir does not exist" in result.stderr
    assert not output_dir.exists()


def test_enrich_missing_real_edge_model_fails_before_output(tmp_path: Path) -> None:
    image_path, depth_path, detections_path = write_inputs(tmp_path)
    output_dir = tmp_path / "enrich"

    result = run_cli(
        "enrich-objects",
        "--image",
        str(image_path),
        "--depth",
        str(depth_path),
        "--detections",
        str(detections_path),
        "--edge-backend",
        "dexined",
        "--edge-model-dir",
        str(tmp_path / "missing-edge"),
        "--mesh-backend",
        "none",
        "--output",
        str(output_dir),
    )

    assert result.returncode == 2
    assert "--edge-model-dir does not exist" in result.stderr
    assert not output_dir.exists()


def test_enrich_missing_real_mesh_model_fails_before_output(tmp_path: Path) -> None:
    image_path, depth_path, detections_path = write_inputs(tmp_path)
    output_dir = tmp_path / "enrich"

    result = run_cli(
        "enrich-objects",
        "--image",
        str(image_path),
        "--depth",
        str(depth_path),
        "--detections",
        str(detections_path),
        "--edge-backend",
        "simple",
        "--mesh-backend",
        "triposr",
        "--mesh-model-dir",
        str(tmp_path / "missing-mesh"),
        "--output",
        str(output_dir),
    )

    assert result.returncode == 2
    assert "--mesh-model-dir does not exist" in result.stderr
    assert not output_dir.exists()
