from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "run.py"), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_reconstruct_legacy_rgbd_yolo_requires_weights_before_output(tmp_path: Path) -> None:
    reference_blend = tmp_path / "reference.blend"
    reference_blend.write_bytes(b"not a real blend")
    output_dir = tmp_path / "Latest"

    result = run_cli(
        "reconstruct-scene",
        "--reference-blend",
        str(reference_blend),
        "--detector-backend",
        "rgbd-yolo",
        "--edge-backend",
        "simple",
        "--mesh-backend",
        "none",
        "--output",
        str(output_dir),
        "--device",
        "0",
    )

    assert result.returncode == 2
    assert "--detector-weights is required" in result.stderr
    assert not output_dir.exists()


def test_reconstruct_lightweight_providers_can_preflight_without_model_paths(tmp_path: Path) -> None:
    reference_blend = tmp_path / "reference.blend"
    reference_blend.write_bytes(b"not a real blend")
    weights = tmp_path / "weights.pt"
    weights.write_text("not real weights", encoding="utf-8")
    output_dir = tmp_path / "Latest"

    result = run_cli(
        "reconstruct-scene",
        "--reference-blend",
        str(reference_blend),
        "--detector-weights",
        str(weights),
        "--edge-backend",
        "simple",
        "--mesh-backend",
        "none",
        "--output",
        str(output_dir),
        "--blender",
        str(tmp_path / "missing-blender"),
        "--device",
        "0",
    )

    assert result.returncode == 2
    assert (output_dir / "run_status.json").is_file()


def test_reconstruct_fake_detector_backend_is_not_public(tmp_path: Path) -> None:
    reference_blend = tmp_path / "reference.blend"
    reference_blend.write_bytes(b"not a real blend")

    result = run_cli(
        "reconstruct-scene",
        "--reference-blend",
        str(reference_blend),
        "--detector-backend",
        "fake",
        "--output",
        str(tmp_path / "Latest"),
    )

    assert result.returncode == 2
    assert "invalid choice: 'fake'" in result.stderr


def test_reconstruct_missing_real_edge_model_fails_before_output_mutation(tmp_path: Path) -> None:
    reference_blend = tmp_path / "reference.blend"
    reference_blend.write_bytes(b"not a real blend")
    weights = tmp_path / "weights.pt"
    weights.write_text("not real weights", encoding="utf-8")
    output_dir = tmp_path / "Latest"

    result = run_cli(
        "reconstruct-scene",
        "--reference-blend",
        str(reference_blend),
        "--detector-weights",
        str(weights),
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


def test_reconstruct_missing_real_wireframe_model_fails_before_output_mutation(tmp_path: Path) -> None:
    reference_blend = tmp_path / "reference.blend"
    reference_blend.write_bytes(b"not a real blend")
    weights = tmp_path / "weights.pt"
    weights.write_text("not real weights", encoding="utf-8")
    output_dir = tmp_path / "Latest"

    result = run_cli(
        "reconstruct-scene",
        "--reference-blend",
        str(reference_blend),
        "--detector-weights",
        str(weights),
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
        "--device",
        "0",
    )

    assert result.returncode == 2
    assert "--wireframe-model-dir does not exist" in result.stderr
    assert not output_dir.exists()


def test_reconstruct_real_provider_uses_auto_device_after_model_preflight(tmp_path: Path) -> None:
    reference_blend = tmp_path / "reference.blend"
    reference_blend.write_bytes(b"not a real blend")
    weights = tmp_path / "weights.pt"
    weights.write_text("not real weights", encoding="utf-8")
    output_dir = tmp_path / "Latest"

    result = run_cli(
        "reconstruct-scene",
        "--reference-blend",
        str(reference_blend),
        "--detector-weights",
        str(weights),
        "--edge-backend",
        "simple",
        "--mesh-backend",
        "none",
        "--output",
        str(output_dir),
    )

    assert result.returncode == 2
    assert "--device is required" not in result.stderr
    assert (output_dir / "run_status.json").is_file()


def test_reconstruct_groundingdino_sam3_open_vocab_root_not_ready_fails_before_output(tmp_path: Path) -> None:
    reference_blend = tmp_path / "reference.blend"
    reference_blend.write_bytes(b"not a real blend")
    output_dir = tmp_path / "Latest"

    result = run_cli(
        "reconstruct-scene",
        "--reference-blend",
        str(reference_blend),
        "--detector-backend",
        "groundingdino-sam3",
        "--open-vocab-root",
        str(tmp_path / "OpenVocabulary"),
        "--edge-backend",
        "simple",
        "--mesh-backend",
        "none",
        "--output",
        str(output_dir),
    )

    assert result.returncode == 2
    assert "Open-vocabulary integration is not ready for reconstruction" in result.stderr
    assert not output_dir.exists()
