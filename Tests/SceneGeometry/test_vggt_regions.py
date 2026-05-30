from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from SceneGeometry.VGGT.regions import fit_box, sample_points_for_mask


ROOT = Path(__file__).resolve().parents[2]


def run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "run.py"), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_sample_points_for_mask_uses_finite_masked_points() -> None:
    points = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    points[1, 1] = np.nan
    mask = np.zeros((3, 3), dtype=bool)
    mask[0, 0] = True
    mask[1, 1] = True
    mask[2, 2] = True

    sampled = sample_points_for_mask(points, mask)

    assert sampled.tolist() == [[0.0, 1.0, 2.0], [24.0, 25.0, 26.0]]


def test_fit_aabb_values_exactly() -> None:
    points = np.array(
        [
            [0.0, 1.0, 2.0],
            [2.0, 5.0, 8.0],
            [1.0, 3.0, 4.0],
        ],
        dtype=np.float32,
    )

    result = fit_box(points, box_mode="aabb")

    assert result.box_type == "aabb"
    assert result.center_xyz == [1.0, 3.0, 5.0]
    assert result.extent_xyz == [2.0, 4.0, 6.0]
    assert result.rotation_matrix == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert result.needs_review is False
    assert result.failure_reason is None


def test_auto_obb_falls_back_to_aabb_for_degenerate_points() -> None:
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)

    result = fit_box(points, box_mode="auto")

    assert result.box_type == "aabb"
    assert result.needs_review is True
    assert result.failure_reason == "degenerate_covariance"


def test_fit_vggt_boxes_cli_writes_regions_and_failed_missing_mask(tmp_path: Path) -> None:
    detections_path = tmp_path / "detections.json"
    objects_dir = tmp_path / "objects"
    vggt_dir = tmp_path / "objects_vggt"
    output_dir = tmp_path / "out"
    object_dir = objects_dir / "01_cube"
    object_dir.mkdir(parents=True)
    vggt_dir.mkdir()

    detections = {
        "image_width": 4,
        "image_height": 4,
        "model_info": {"fusion_contract": {"scene": {"axes": {"x": "image_right"}}}},
        "objects": [
            {
                "id": 1,
                "detector_label": "cube",
                "detector_confidence": 0.9,
                "bbox_xyxy": [0, 0, 2, 2],
            },
            {
                "id": 2,
                "detector_label": "missing",
                "detector_confidence": 0.8,
                "bbox_xyxy": [2, 2, 3, 3],
            },
        ],
    }
    detections_path.write_text(json.dumps(detections), encoding="utf-8")
    (object_dir / "metadata.json").write_text(json.dumps({"id": 1}), encoding="utf-8")
    mask = Image.new("L", (4, 4), 0)
    for xy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        mask.putpixel(xy, 255)
    mask.save(object_dir / "full_mask.png")

    points = np.zeros((4, 4, 3), dtype=np.float32)
    for y in range(4):
        for x in range(4):
            points[y, x] = [float(x), float(y), float(x + y)]
    np.save(vggt_dir / "vggt_points.npy", points)
    (vggt_dir / "vggt_geometry.json").write_text(json.dumps({"coordinate_contract": {"schema_version": 1}}), encoding="utf-8")
    (vggt_dir / "vggt_camera.json").write_text("{}", encoding="utf-8")

    result = run_cli(
        [
            "fit-vggt-boxes",
            "--detections",
            str(detections_path),
            "--objects",
            str(objects_dir),
            "--vggt",
            str(vggt_dir),
            "--output",
            str(output_dir),
            "--box-mode",
            "aabb",
            "--min-valid-points",
            "1",
        ]
    )

    assert result.returncode == 0, result.stderr
    report = json.loads((output_dir / "object_geometry.json").read_text(encoding="utf-8"))
    assert report["summary"]["detection_count"] == 2
    assert report["summary"]["fit_count"] == 1
    assert report["summary"]["failed_count"] == 1
    boxes_obj = output_dir / "vggt_boxes.obj"
    assert report["artifacts"]["boxes_obj"] == str(boxes_obj)
    assert report["artifacts"]["regions_overlay_png"] == str(output_dir / "vggt_regions_overlay.png")
    assert boxes_obj.is_file()
    assert (output_dir / "vggt_regions_overlay.png").is_file()
    box_lines = boxes_obj.read_text(encoding="utf-8").splitlines()
    assert sum(line.startswith("v ") for line in box_lines) == 8
    assert sum(line.startswith("f ") for line in box_lines) == 6
    fitted, failed = report["objects"]
    assert fitted["detection_id"] == 1
    assert fitted["mask_source"] == "full_mask"
    assert fitted["point_count"] == 4
    assert fitted["box_type"] == "aabb"
    assert Path(fitted["artifacts"]["points_xyz"]).is_file()
    assert Path(fitted["artifacts"]["points_obj"]).is_file()
    assert Path(fitted["artifacts"]["mask_png"]).is_file()
    assert Path(fitted["artifacts"]["valid_points_png"]).is_file()
    assert Path(fitted["artifacts"]["point_distance_png"]).is_file()
    region_lines = Path(fitted["artifacts"]["points_obj"]).read_text(encoding="utf-8").splitlines()
    assert sum(line.startswith("v ") for line in region_lines) == 4
    assert sum(line.startswith("f ") for line in region_lines) == 1
    assert failed["detection_id"] == 2
    assert failed["box_type"] == "failed"
    assert failed["failure_reason"] == "missing_mask"
