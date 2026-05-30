from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from SceneGeometry.VGGT.pipeline import (
    convert_vggt_points_to_sceneforge_camera,
    find_local_hf_snapshot,
    run_vggt_image_geometry,
    scene_point_to_blender_obj_vertex,
    scene_point_to_gltf_vertex,
)


ROOT = Path(__file__).resolve().parents[2]


def run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "run.py"), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_run_vggt_fake_writes_geometry_artifacts(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    output_dir = tmp_path / "objects_vggt"
    Image.new("RGB", (16, 12), (80, 120, 160)).save(image_path)

    result = run_cli(
        [
            "run-vggt",
            "--backend",
            "fake",
            "--image",
            str(image_path),
            "--output",
            str(output_dir),
            "--vggt-cache-dir",
            str(tmp_path / "hf-cache"),
            "--vggt-local-only",
            "--obj-stride",
            "4",
        ]
    )

    assert result.returncode == 0, result.stderr
    for name in (
        "vggt_geometry.json",
        "vggt_camera.json",
        "vggt_depth.png",
        "vggt_depth.npy",
        "vggt_points.npy",
        "vggt_points.xyz",
        "vggt_mesh.obj",
        "vggt_mesh.glb",
        "vggt_confidence.png",
    ):
        assert (output_dir / name).is_file()

    report = json.loads((output_dir / "vggt_geometry.json").read_text(encoding="utf-8"))
    assert report["backend"] == "fake"
    assert report["image_width"] == 16
    assert report["image_height"] == 12
    assert report["coordinate_contract"]["scene"]["axes"]["x"] == "image_right"
    assert report["summary"]["point_count"] == 16 * 12
    assert report["artifacts"]["mesh_obj"] == "vggt_mesh.obj"
    assert report["artifacts"]["mesh_glb"] == "vggt_mesh.glb"
    assert report["summary"]["obj_stride"] == 4
    assert report["summary"]["obj_vertex_count"] == 12
    assert report["summary"]["obj_face_count"] == 12
    assert report["summary"]["obj_winding"] == "camera_facing"
    assert report["summary"]["obj_point_source"] == "sceneforge_camera_points"
    assert report["summary"]["glb_vertex_count"] == 12
    assert report["summary"]["glb_face_count"] == 12
    assert report["summary"]["glb_point_source"] == "sceneforge_camera_points"
    assert report["summary"]["glb_vertex_colors"] is True

    depth = np.load(output_dir / "vggt_depth.npy")
    points = np.load(output_dir / "vggt_points.npy")
    assert depth.shape == (12, 16)
    assert points.shape == (12, 16, 3)
    obj_lines = (output_dir / "vggt_mesh.obj").read_text(encoding="utf-8").splitlines()
    assert sum(line.startswith("v ") for line in obj_lines) == 12
    assert sum(line.startswith("f ") for line in obj_lines) == 12
    assert (output_dir / "vggt_mesh.glb").stat().st_size > 0


def test_scene_point_to_blender_obj_vertex_compensates_for_default_obj_import_axes() -> None:
    assert scene_point_to_blender_obj_vertex(np.array([1.0, 2.0, 3.0], dtype=np.float32)) == (1.0, -2.0, -3.0)


def test_scene_point_to_gltf_vertex_uses_y_up_gltf_axes() -> None:
    assert scene_point_to_gltf_vertex(np.array([1.0, 2.0, 3.0], dtype=np.float32)) == (1.0, 3.0, -2.0)


def test_convert_vggt_points_to_sceneforge_camera_axes() -> None:
    points = np.array([[[1.0, -2.0, 3.0]]], dtype=np.float32)

    converted = convert_vggt_points_to_sceneforge_camera(points)

    assert converted.tolist() == [[[1.0, 3.0, 2.0]]]



def test_find_local_hf_snapshot_picks_snapshot_with_model_file(tmp_path: Path) -> None:
    snapshot = tmp_path / "models--facebook--VGGT-1B" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"stub")

    assert find_local_hf_snapshot("facebook/VGGT-1B", tmp_path) == snapshot

def test_run_vggt_real_backend_missing_import_fails_clearly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image_path = tmp_path / "input.png"
    Image.new("RGB", (8, 8), "white").save(image_path)

    original_import = __import__

    def blocked_vggt_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "vggt" or name.startswith("vggt."):
            raise ModuleNotFoundError("No module named 'vggt'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", blocked_vggt_import)

    with pytest.raises(RuntimeError, match="VGGT is not importable"):
        run_vggt_image_geometry(image_path=image_path, output_dir=tmp_path / "out")
