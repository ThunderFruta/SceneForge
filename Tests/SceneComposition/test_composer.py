from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from SceneComposition.composer import compose_scene, placement_transform_to_gltf


ROOT = Path(__file__).resolve().parents[2]


def run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "run.py"), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def write_box_glb(path: Path, *, extents: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> None:
    import trimesh

    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.creation.box(extents=extents)
    mesh.export(path)


def write_background_vggt_fixture(vggt_dir: Path) -> None:
    vggt_dir.mkdir(parents=True, exist_ok=True)
    points = np.zeros((4, 4, 3), dtype=np.float32)
    for y in range(4):
        for x in range(4):
            points[y, x] = [float(x), 10.0 + float(y), float(y) / 10.0]
    np.save(vggt_dir / "vggt_points.npy", points)
    Image.new("RGB", (4, 4), (40, 80, 120)).save(vggt_dir / "empty_room.png")


def test_placement_transform_maps_sceneforge_camera_box_to_gltf_axes() -> None:
    transform = placement_transform_to_gltf(
        {
            "center_xyz": [1.0, 2.0, 3.0],
            "extent_xyz": [2.0, 4.0, 6.0],
            "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    )

    assert np.allclose(transform[:3, 3], [1.0, 3.0, -2.0])
    assert np.allclose(transform[:3, :3], [[2.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 4.0]])


def test_obb_placement_transform_preserves_raw_box_rotation_when_requested() -> None:
    transform = placement_transform_to_gltf(
        {
            "center_xyz": [1.0, 2.0, 3.0],
            "extent_xyz": [2.0, 4.0, 6.0],
            "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        },
        placement_orientation="obb",
    )

    assert np.allclose(transform[:3, 3], [1.0, 3.0, -2.0])
    assert np.allclose(transform[:3, :3], [[2.0, 0.0, 0.0], [0.0, 0.0, 6.0], [0.0, -4.0, 0.0]])


def test_compose_scene_writes_scene_glb_and_alignment_report(tmp_path: Path) -> None:
    background = tmp_path / "background" / "empty_room_mesh.glb"
    objects_dir = tmp_path / "objects"
    object_dir = objects_dir / "01_cube"
    object_geometry = tmp_path / "objects_vggt" / "object_geometry.json"
    output_dir = tmp_path / "scene"
    write_box_glb(background, extents=(0.5, 0.5, 0.5))
    write_box_glb(object_dir / "hunyuan3d_textured.glb")
    object_dir.mkdir(parents=True, exist_ok=True)
    (object_dir / "metadata.json").write_text(json.dumps({"id": 1}), encoding="utf-8")
    object_geometry.parent.mkdir(parents=True, exist_ok=True)
    object_geometry.write_text(
        json.dumps(
            {
                "coordinate_contract": {"schema_version": 1},
                "objects": [
                    {
                        "detection_id": 1,
                        "detector_label": "cube",
                        "box_type": "aabb",
                        "needs_review": False,
                        "center_xyz": [1.0, 2.0, 3.0],
                        "extent_xyz": [2.0, 4.0, 6.0],
                        "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = compose_scene(
        background_path=background,
        objects_dir=objects_dir,
        object_geometry_path=object_geometry,
        output_dir=output_dir,
        object_scale_factor=1.0,
        background_fit="placement-bounds",
    )

    assert (output_dir / "scene.glb").is_file()
    assert (output_dir / "scene_alignment.json").is_file()
    assert report["summary"] == {"placement_count": 1, "composed_count": 1, "skipped_count": 0, "failed_count": 0}
    transformed_bounds = np.asarray(report["objects"][0]["transformed_bounds"])
    background_bounds = np.asarray(report["background"]["transformed_bounds"])
    assert np.allclose(background_bounds[:, :2], [[-0.08, -0.24], [2.08, 6.24]])
    assert np.allclose(background_bounds[:, 2], [-4.44, -0.12])
    assert np.isclose(transformed_bounds[0, 1], background_bounds[0, 1])
    assert np.allclose(transformed_bounds[:, [0, 2]], [[0.0, -4.0], [2.0, 0.0]])


def test_compose_scene_camera_clipped_background_uses_masks(tmp_path: Path) -> None:
    background = tmp_path / "background" / "empty_room_mesh.glb"
    objects_dir = tmp_path / "objects"
    object_dir = objects_dir / "01_cube"
    object_geometry = tmp_path / "object_geometry.json"
    output_dir = tmp_path / "scene"
    write_box_glb(background)
    write_background_vggt_fixture(background.parent)
    write_box_glb(object_dir / "hunyuan3d_textured.glb")
    object_dir.mkdir(parents=True, exist_ok=True)
    (object_dir / "metadata.json").write_text(json.dumps({"id": 1}), encoding="utf-8")
    mask = Image.new("L", (4, 4), 0)
    mask.putpixel((1, 1), 255)
    mask.save(object_dir / "full_mask.png")
    object_geometry.write_text(
        json.dumps(
            {
                "objects": [
                    {
                        "detection_id": 1,
                        "detector_label": "cube",
                        "box_type": "aabb",
                        "needs_review": False,
                        "center_xyz": [0.0, 1.0, 0.0],
                        "extent_xyz": [1.0, 1.0, 1.0],
                        "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = compose_scene(
        background_path=background,
        objects_dir=objects_dir,
        object_geometry_path=object_geometry,
        output_dir=output_dir,
        background_fit="camera-clipped",
        background_stride=1,
        background_clip_dilation_px=0,
    )

    assert report["background"]["source"] == "vggt_points_camera_clipped"
    assert report["background"]["vertex_count"] == 15
    assert report["background"]["masked_pixel_ratio"] == 1 / 16
    assert report["summary"]["composed_count"] == 1


def test_compose_scene_room_corner_background_is_structural(tmp_path: Path) -> None:
    background = tmp_path / "background.glb"
    objects_dir = tmp_path / "objects"
    object_dir = objects_dir / "01_cube"
    object_geometry = tmp_path / "object_geometry.json"
    output_dir = tmp_path / "scene"
    write_box_glb(background)
    write_box_glb(object_dir / "hunyuan3d_textured.glb")
    object_dir.mkdir(parents=True, exist_ok=True)
    (object_dir / "metadata.json").write_text(json.dumps({"id": 1}), encoding="utf-8")
    object_geometry.write_text(
        json.dumps(
            {
                "objects": [
                    {
                        "detection_id": 1,
                        "detector_label": "cube",
                        "box_type": "aabb",
                        "needs_review": False,
                        "center_xyz": [0.0, 1.0, 0.0],
                        "extent_xyz": [1.0, 2.0, 3.0],
                        "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = compose_scene(
        background_path=background,
        objects_dir=objects_dir,
        object_geometry_path=object_geometry,
        output_dir=output_dir,
        background_fit="room-corner",
        object_scale_factor=1.0,
    )

    assert report["background"]["source"] == "procedural_room_corner_from_placement_bounds"
    assert report["background"]["mesh_count"] == 3
    assert report["background"]["face_count"] == 6
    assert report["background"]["floor_y"] < report["objects"][0]["transformed_bounds"][0][1]
    assert report["background"]["z_back"] < report["objects"][0]["transformed_bounds"][0][2]


def test_compose_scene_snaps_objects_to_floor(tmp_path: Path) -> None:
    background = tmp_path / "background.glb"
    objects_dir = tmp_path / "objects"
    object_dir = objects_dir / "01_cube"
    object_geometry = tmp_path / "object_geometry.json"
    output_dir = tmp_path / "scene"
    write_box_glb(background)
    write_box_glb(object_dir / "hunyuan3d_textured.glb")
    object_dir.mkdir(parents=True, exist_ok=True)
    (object_dir / "metadata.json").write_text(json.dumps({"id": 1}), encoding="utf-8")
    object_geometry.write_text(
        json.dumps(
            {
                "objects": [
                    {
                        "detection_id": 1,
                        "detector_label": "cube",
                        "box_type": "aabb",
                        "needs_review": False,
                        "center_xyz": [0.0, 1.0, 0.0],
                        "extent_xyz": [1.0, 1.0, 1.0],
                        "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = compose_scene(
        background_path=background,
        objects_dir=objects_dir,
        object_geometry_path=object_geometry,
        output_dir=output_dir,
        background_fit="room-corner",
    )

    floor_y = report["floor_y"]
    object_bottom_y = report["objects"][0]["transformed_bounds"][0][1]
    assert np.isclose(object_bottom_y, floor_y)
    assert report["objects"][0]["floor_snap_delta"] != 0.0


def test_compose_scene_snaps_tabletop_objects_to_table_surface(tmp_path: Path) -> None:
    background = tmp_path / "background.glb"
    objects_dir = tmp_path / "objects"
    table_dir = objects_dir / "01_table"
    flower_dir = objects_dir / "02_flower"
    object_geometry = tmp_path / "object_geometry.json"
    output_dir = tmp_path / "scene"
    write_box_glb(background)
    write_box_glb(table_dir / "hunyuan3d_textured.glb")
    write_box_glb(flower_dir / "hunyuan3d_textured.glb")
    table_dir.mkdir(parents=True, exist_ok=True)
    flower_dir.mkdir(parents=True, exist_ok=True)
    (table_dir / "metadata.json").write_text(json.dumps({"id": 1}), encoding="utf-8")
    (flower_dir / "metadata.json").write_text(json.dumps({"id": 2}), encoding="utf-8")
    object_geometry.write_text(
        json.dumps(
            {
                "objects": [
                    {
                        "detection_id": 2,
                        "detector_label": "flower",
                        "box_type": "aabb",
                        "needs_review": False,
                        "bbox_xyxy": [45.0, 5.0, 55.0, 35.0],
                        "center_xyz": [0.0, 1.0, 0.5],
                        "extent_xyz": [0.2, 0.2, 0.2],
                        "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    },
                    {
                        "detection_id": 1,
                        "detector_label": "round table",
                        "box_type": "aabb",
                        "needs_review": False,
                        "bbox_xyxy": [20.0, 30.0, 80.0, 90.0],
                        "center_xyz": [0.0, 1.0, 0.0],
                        "extent_xyz": [1.0, 1.0, 1.0],
                        "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    report = compose_scene(
        background_path=background,
        objects_dir=objects_dir,
        object_geometry_path=object_geometry,
        output_dir=output_dir,
        background_fit="room-corner",
        object_scale_factor=1.0,
    )

    by_id = {item["detection_id"]: item for item in report["objects"]}
    table_top_y = by_id[1]["transformed_bounds"][1][1]
    flower_bottom_y = by_id[2]["transformed_bounds"][0][1]
    assert np.isclose(by_id[1]["transformed_bounds"][0][1], report["floor_y"])
    assert np.isclose(flower_bottom_y, table_top_y)
    assert by_id[2]["support_kind"] == "tabletop"
    assert by_id[2]["support_detection_id"] == 1
    assert by_id[2]["floor_snap_delta"] == 0.0
    assert by_id[2]["support_snap_delta"] != 0.0


def test_compose_scene_scales_and_spreads_chairs_from_table(tmp_path: Path) -> None:
    background = tmp_path / "background.glb"
    objects_dir = tmp_path / "objects"
    table_dir = objects_dir / "01_table"
    chair_dir = objects_dir / "02_chair"
    object_geometry = tmp_path / "object_geometry.json"
    output_dir = tmp_path / "scene"
    write_box_glb(background)
    write_box_glb(table_dir / "hunyuan3d_textured.glb")
    write_box_glb(chair_dir / "hunyuan3d_textured.glb")
    table_dir.mkdir(parents=True, exist_ok=True)
    chair_dir.mkdir(parents=True, exist_ok=True)
    (table_dir / "metadata.json").write_text(json.dumps({"id": 1}), encoding="utf-8")
    (chair_dir / "metadata.json").write_text(json.dumps({"id": 2}), encoding="utf-8")
    object_geometry.write_text(
        json.dumps(
            {
                "objects": [
                    {
                        "detection_id": 1,
                        "detector_label": "round table",
                        "box_type": "aabb",
                        "needs_review": False,
                        "center_xyz": [0.0, 1.0, 0.0],
                        "extent_xyz": [1.0, 1.0, 1.0],
                        "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    },
                    {
                        "detection_id": 2,
                        "detector_label": "chair",
                        "box_type": "aabb",
                        "needs_review": False,
                        "center_xyz": [0.3, 1.0, 0.0],
                        "extent_xyz": [1.0, 1.0, 1.0],
                        "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    report = compose_scene(
        background_path=background,
        objects_dir=objects_dir,
        object_geometry_path=object_geometry,
        output_dir=output_dir,
        background_fit="room-corner",
        object_scale_factor=1.0,
    )

    by_id = {item["detection_id"]: item for item in report["objects"]}
    assert by_id[2]["label_scale_factor"] == 0.78
    assert by_id[2]["spacing_delta_gltf"][0] > 0.0
    assert by_id[2]["semantic_orientation_kind"] == "face_nearest_table"
    assert np.isclose(by_id[2]["semantic_yaw_radians"], -np.pi / 2.0)
    chair_extent_x = by_id[2]["transformed_bounds"][1][0] - by_id[2]["transformed_bounds"][0][0]
    assert np.isclose(chair_extent_x, 0.78)


def test_compose_scene_writes_support_dof_and_render_proxy_overlay(tmp_path: Path) -> None:
    source_image = tmp_path / "source.png"
    Image.new("RGB", (100, 100), (240, 240, 240)).save(source_image)
    background = tmp_path / "background.glb"
    objects_dir = tmp_path / "objects"
    object_dir = objects_dir / "01_cube"
    object_geometry = tmp_path / "object_geometry.json"
    output_dir = tmp_path / "scene"
    write_box_glb(background)
    write_box_glb(object_dir / "hunyuan3d_textured.glb")
    object_dir.mkdir(parents=True, exist_ok=True)
    (object_dir / "metadata.json").write_text(json.dumps({"id": 1}), encoding="utf-8")
    object_geometry.write_text(
        json.dumps(
            {
                "coordinate_contract": {
                    "image_width": 100,
                    "image_height": 100,
                    "fov_degrees": 70.0,
                },
                "objects": [
                    {
                        "detection_id": 1,
                        "detector_label": "cube",
                        "box_type": "aabb",
                        "needs_review": False,
                        "bbox_xyxy": [42.0, 42.0, 58.0, 58.0],
                        "center_xyz": [0.0, 1.0, 0.0],
                        "extent_xyz": [0.2, 0.2, 0.2],
                        "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = compose_scene(
        background_path=background,
        objects_dir=objects_dir,
        object_geometry_path=object_geometry,
        output_dir=output_dir,
        background_fit="room-corner",
        object_scale_factor=1.0,
        source_image_path=source_image,
    )

    record = report["objects"][0]
    optimization = record["render_to_input_optimization"]
    assert record["support_degrees_of_freedom"]["model"] == "support_plane_4dof"
    assert record["support_degrees_of_freedom"]["free_parameters"] == ["plane_x", "plane_z", "yaw_y", "uniform_scale"]
    assert optimization["enabled"] is True
    assert optimization["candidate_count"] > 0
    assert optimization["optimized_projected_bbox_xyxy"] is not None
    assert (output_dir / "input_vs_projection_overlay.png").is_file()


def test_compose_scene_rejects_bad_projection_optimization(tmp_path: Path) -> None:
    source_image = tmp_path / "source.png"
    Image.new("RGB", (100, 100), (240, 240, 240)).save(source_image)
    background = tmp_path / "background.glb"
    objects_dir = tmp_path / "objects"
    object_dir = objects_dir / "01_cube"
    object_geometry = tmp_path / "object_geometry.json"
    output_dir = tmp_path / "scene"
    write_box_glb(background)
    write_box_glb(object_dir / "hunyuan3d_textured.glb")
    object_dir.mkdir(parents=True, exist_ok=True)
    (object_dir / "metadata.json").write_text(json.dumps({"id": 1}), encoding="utf-8")
    object_geometry.write_text(
        json.dumps(
            {
                "coordinate_contract": {
                    "image_width": 100,
                    "image_height": 100,
                    "fov_degrees": 70.0,
                },
                "objects": [
                    {
                        "detection_id": 1,
                        "detector_label": "cube",
                        "box_type": "aabb",
                        "needs_review": False,
                        "bbox_xyxy": [42.0, 82.0, 58.0, 96.0],
                        "center_xyz": [0.0, 1.0, 0.0],
                        "extent_xyz": [0.2, 0.2, 0.2],
                        "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = compose_scene(
        background_path=background,
        objects_dir=objects_dir,
        object_geometry_path=object_geometry,
        output_dir=output_dir,
        background_fit="room-corner",
        object_scale_factor=1.0,
        source_image_path=source_image,
    )

    record = report["objects"][0]
    optimization = record["render_to_input_optimization"]
    assert record["needs_review"] is True
    assert record["projection_quality"]["status"] == "rejected"
    assert optimization["projection_quality"]["reason"] == "vertical_edge_error"
    assert optimization["candidate_projected_bbox_xyxy"] is not None
    assert optimization["optimized_projected_bbox_xyxy"] == optimization["initial_projected_bbox_xyxy"]
    assert report["projection_quality"]["rejected_count"] == 1


def test_compose_scene_reports_object_overlap_warnings(tmp_path: Path) -> None:
    background = tmp_path / "background.glb"
    objects_dir = tmp_path / "objects"
    first_dir = objects_dir / "01_cube"
    second_dir = objects_dir / "02_box"
    object_geometry = tmp_path / "object_geometry.json"
    output_dir = tmp_path / "scene"
    write_box_glb(background)
    write_box_glb(first_dir / "hunyuan3d_textured.glb")
    write_box_glb(second_dir / "hunyuan3d_textured.glb")
    for object_id, object_dir in ((1, first_dir), (2, second_dir)):
        object_dir.mkdir(parents=True, exist_ok=True)
        (object_dir / "metadata.json").write_text(json.dumps({"id": object_id}), encoding="utf-8")
    object_geometry.write_text(
        json.dumps(
            {
                "objects": [
                    {
                        "detection_id": 1,
                        "detector_label": "cube",
                        "box_type": "aabb",
                        "needs_review": False,
                        "center_xyz": [0.0, 1.0, 0.0],
                        "extent_xyz": [1.0, 1.0, 1.0],
                        "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    },
                    {
                        "detection_id": 2,
                        "detector_label": "box",
                        "box_type": "aabb",
                        "needs_review": False,
                        "center_xyz": [0.1, 1.0, 0.0],
                        "extent_xyz": [1.0, 1.0, 1.0],
                        "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    report = compose_scene(
        background_path=background,
        objects_dir=objects_dir,
        object_geometry_path=object_geometry,
        output_dir=output_dir,
        background_fit="room-corner",
        object_scale_factor=1.0,
    )

    assert report["object_overlap_warnings"]
    warning = report["object_overlap_warnings"][0]
    assert warning["detection_ids"] == [1, 2]
    assert warning["overlap_volume_gltf"] > 0.0


def test_compose_scene_can_keep_raw_background_transform(tmp_path: Path) -> None:
    background = tmp_path / "background.glb"
    objects_dir = tmp_path / "objects"
    object_dir = objects_dir / "01_cube"
    object_geometry = tmp_path / "object_geometry.json"
    output_dir = tmp_path / "scene"
    write_box_glb(background, extents=(2.0, 3.0, 4.0))
    write_box_glb(object_dir / "hunyuan3d_textured.glb")
    object_dir.mkdir(parents=True, exist_ok=True)
    (object_dir / "metadata.json").write_text(json.dumps({"id": 1}), encoding="utf-8")
    object_geometry.write_text(
        json.dumps(
            {
                "objects": [
                    {
                        "detection_id": 1,
                        "detector_label": "cube",
                        "box_type": "aabb",
                        "needs_review": False,
                        "center_xyz": [0.0, 1.0, 0.0],
                        "extent_xyz": [1.0, 1.0, 1.0],
                        "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = compose_scene(
        background_path=background,
        objects_dir=objects_dir,
        object_geometry_path=object_geometry,
        output_dir=output_dir,
        background_fit="raw",
    )

    assert np.allclose(report["background"]["source_bounds"], report["background"]["transformed_bounds"])


def test_compose_scene_cli(tmp_path: Path) -> None:
    background = tmp_path / "background.glb"
    objects_dir = tmp_path / "objects"
    object_dir = objects_dir / "01_cube"
    object_geometry = tmp_path / "object_geometry.json"
    output_dir = tmp_path / "scene"
    write_box_glb(background)
    write_box_glb(object_dir / "hunyuan3d_textured.glb")
    object_dir.mkdir(parents=True, exist_ok=True)
    (object_dir / "metadata.json").write_text(json.dumps({"id": 1}), encoding="utf-8")
    object_geometry.write_text(
        json.dumps(
            {
                "objects": [
                    {
                        "detection_id": 1,
                        "detector_label": "cube",
                        "box_type": "aabb",
                        "needs_review": False,
                        "center_xyz": [0.0, 1.0, 0.0],
                        "extent_xyz": [1.0, 1.0, 1.0],
                        "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = run_cli(
        [
            "compose-scene",
            "--background",
            str(background),
            "--objects",
            str(objects_dir),
            "--object-geometry",
            str(object_geometry),
            "--output",
            str(output_dir),
        ]
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "scene.glb").is_file()
    assert (output_dir / "scene_alignment.json").is_file()
    assert "Composed 1/1 objects" in result.stdout
