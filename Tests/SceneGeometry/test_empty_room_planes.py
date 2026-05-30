from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from SceneGeometry.Planes.empty_room import fit_empty_room_planes


ROOT = Path(__file__).resolve().parents[2]


def run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "run.py"), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def write_empty_room_fixture(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    width, height = 12, 8
    points = np.zeros((height, width, 3), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            scene_x = (x - width / 2) / width
            scene_y = 0.5 + y / height
            scene_z = -0.2 + y / height * 0.8
            if y > height * 0.58:
                scene_z = -0.2
            points[y, x] = [scene_x, scene_y, scene_z]
    np.save(path / "vggt_points.npy", points)
    image = Image.new("RGB", (width, height))
    for y in range(height):
        for x in range(width):
            image.putpixel((x, y), (210 + x, 205 + y * 2, 198 + (x + y) % 9))
    image.save(path / "empty_room.png")


def test_fit_empty_room_planes_writes_xyz_aligned_planes(tmp_path: Path) -> None:
    background = tmp_path / "background"
    write_empty_room_fixture(background)

    report = fit_empty_room_planes(background_dir=background, output_dir=background, stride=1)

    assert (background / "plane_detections.json").is_file()
    assert (background / "empty_room_planes.glb").is_file()
    assert report["align_xyz"] is True
    assert report["mesh"]["axis_transform"] == "gltf_x_image_right_y_image_up_z_back_toward_camera"
    planes = {plane["id"]: plane for plane in report["planes"]}
    assert planes["floor"]["normal_xyz"] == [0.0, 0.0, 1.0]
    assert planes["back_wall"]["normal_xyz"] == [0.0, -1.0, 0.0]
    assert planes["right_wall"]["normal_xyz"] == [-1.0, 0.0, 0.0]
    assert planes["floor"]["plane_subtype"] == "floor"
    assert planes["back_wall"]["plane_subtype"] == "wall"
    assert report["mesh"]["vertex_count"] > 12
    assert report["mesh"]["texture_source"] == "empty_room_image_uv_projected"
    assert report["mesh"]["vertex_colors"] == "projected_empty_room_image_fallback"


def test_fit_empty_room_planes_cli(tmp_path: Path) -> None:
    background = tmp_path / "background"
    output = tmp_path / "out"
    write_empty_room_fixture(background)

    result = run_cli(["fit-empty-room-planes", "--background", str(background), "--output", str(output), "--stride", "1"])

    assert result.returncode == 0, result.stderr
    assert (output / "plane_detections.json").is_file()
    assert (output / "empty_room_planes.glb").is_file()
    report = json.loads((output / "plane_detections.json").read_text(encoding="utf-8"))
    assert len(report["planes"]) == 3
    assert "XYZ-aligned" in result.stdout
