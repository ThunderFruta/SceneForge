from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from run import build_parser
from SceneComposition.composer import (
    candidate_transform,
    facing_prior_loss,
    mesh_facing_prior_from_target,
    physical_size_prior_from_target,
    physical_size_prior_loss,
    placement_transform_to_gltf,
    projected_transform_bbox,
    projection_quality_report,
    support_plane_pivot_local,
    transformed_bounds_from_source_bounds,
    translation_candidates_for_projection_residual,
    yaw_rotation_gltf,
    compose_scene,
)
from SceneComposition.placement import (
    build_object_fit_targets,
    choose_object_supports,
    fit_object_placements,
    floor_occupancy_refit_acceptance,
    reconcile_visibility_explained_projection_reviews,
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


def write_box_glb(path: Path, *, extents: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> None:
    import trimesh

    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.creation.box(extents=extents)
    mesh.export(path)


def write_jagged_bottom_glb(path: Path) -> None:
    import trimesh

    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    vertices = np.asarray(mesh.vertices).copy()
    vertices[0, 1] -= 0.35
    mesh.vertices = vertices
    mesh.export(path)


def write_dangling_bottom_glb(path: Path) -> None:
    import trimesh

    path.parent.mkdir(parents=True, exist_ok=True)
    body = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    body.apply_translation((0.0, 0.25, 0.0))
    dangling = trimesh.creation.box(extents=(0.08, 0.5, 0.08))
    dangling.apply_translation((0.0, -0.55, 0.0))
    mesh = trimesh.util.concatenate([body, dangling])
    mesh.export(path)


def write_background_vggt_fixture(vggt_dir: Path) -> None:
    vggt_dir.mkdir(parents=True, exist_ok=True)
    points = np.zeros((4, 4, 3), dtype=np.float32)
    for y in range(4):
        for x in range(4):
            points[y, x] = [float(x), 10.0 + float(y), float(y) / 10.0]
    np.save(vggt_dir / "vggt_points.npy", points)
    Image.new("RGB", (4, 4), (40, 80, 120)).save(vggt_dir / "empty_room.png")


def write_plane_detections_fixture(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "planes": [
                    {
                        "id": "floor",
                        "vertices_xyz": [[-1.0, 0.5, -0.2], [1.0, 0.5, -0.2], [1.0, 2.0, -0.2], [-1.0, 2.0, -0.2]],
                    },
                    {
                        "id": "back_wall",
                        "vertices_xyz": [[-1.0, 2.0, -0.2], [1.0, 2.0, -0.2], [1.0, 2.0, 1.0], [-1.0, 2.0, 1.0]],
                    },
                    {
                        "id": "right_wall",
                        "vertices_xyz": [[1.0, 0.5, -0.2], [1.0, 2.0, -0.2], [1.0, 2.0, 1.0], [1.0, 0.5, 1.0]],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )


def test_projection_quality_accepts_occluded_floor_bottom_with_review() -> None:
    target = np.asarray([100.0, 50.0, 180.0, 150.0], dtype=np.float64)
    projected = np.asarray([101.0, 54.0, 179.0, 186.0], dtype=np.float64)
    oversized = np.asarray([40.0, 55.0, 240.0, 205.0], dtype=np.float64)

    rejected = projection_quality_report(projected, target)
    accepted = projection_quality_report(projected, target, allow_occluded_bottom=True)
    oversized_rejected = projection_quality_report(oversized, target, allow_occluded_bottom=True)

    assert rejected["status"] == "rejected"
    assert rejected["reason"] == "vertical_edge_error"
    assert accepted["status"] == "accepted_occluded_bottom"
    assert accepted["accepted"] is True
    assert accepted["reason"] == "occluded_bottom_edge_tolerated"
    assert oversized_rejected["status"] == "rejected"
    assert oversized_rejected["horizontal_edge_error_ratio"] > oversized_rejected["horizontal_threshold"]


def test_projection_quality_rejects_too_small_projected_area() -> None:
    target = np.asarray([100.0, 50.0, 180.0, 150.0], dtype=np.float64)
    undersized = np.asarray([112.0, 58.0, 168.0, 142.0], dtype=np.float64)

    rejected = projection_quality_report(undersized, target)

    assert rejected["status"] == "rejected"
    assert rejected["reason"] == "area_error"
    assert rejected["area_error_ratio"] > rejected["area_error_threshold"]


def test_projection_quality_rejects_occluded_bottom_with_large_center_shift() -> None:
    target = np.asarray([100.0, 50.0, 180.0, 150.0], dtype=np.float64)
    shifted = np.asarray([98.0, 90.0, 182.0, 215.0], dtype=np.float64)

    rejected = projection_quality_report(shifted, target, allow_occluded_bottom=True)

    assert rejected["status"] == "rejected"
    assert rejected["center_y_error_ratio"] > rejected["center_threshold"]


def write_plane_report(path: Path, *, floor_z: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "planes": [
                    {
                        "id": "floor",
                        "plane_subtype": "floor",
                        "normal_xyz": [0.0, 0.0, 1.0],
                        "vertices_xyz": [[-2.0, 0.0, floor_z], [2.0, 0.0, floor_z], [2.0, 3.0, floor_z], [-2.0, 3.0, floor_z]],
                        "support_count": 128,
                        "fit_residual": 0.01,
                    },
                    {
                        "id": "back_wall",
                        "plane_subtype": "wall",
                        "normal_xyz": [0.0, -1.0, 0.0],
                        "vertices_xyz": [[-2.0, 3.0, floor_z], [2.0, 3.0, floor_z], [2.0, 3.0, 2.0], [-2.0, 3.0, 2.0]],
                        "support_count": 64,
                        "fit_residual": 0.02,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def write_tiny_floor_report(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "planes": [
                    {
                        "id": "floor",
                        "plane_subtype": "floor",
                        "normal_xyz": [0.0, 0.0, 1.0],
                        "vertices_xyz": [[-0.1, 0.9, 0.0], [0.1, 0.9, 0.0], [0.1, 1.1, 0.0], [-0.1, 1.1, 0.0]],
                        "support_count": 8,
                        "fit_residual": 0.01,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def write_empty_plane_report(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1, "planes": []}), encoding="utf-8")


def write_support_object_geometry(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
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
                        "detector_label": "round table",
                        "box_type": "aabb",
                        "needs_review": False,
                        "bbox_xyxy": [20.0, 30.0, 80.0, 90.0],
                        "center_xyz": [0.0, 1.0, 0.0],
                        "extent_xyz": [1.0, 1.0, 1.0],
                        "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    },
                    {
                        "detection_id": 2,
                        "detector_label": "vase",
                        "box_type": "aabb",
                        "needs_review": False,
                        "bbox_xyxy": [45.0, 5.0, 55.0, 35.0],
                        "center_xyz": [0.0, 1.0, 0.5],
                        "extent_xyz": [0.2, 0.2, 0.2],
                        "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    },
                    {
                        "detection_id": 3,
                        "detector_label": "chair",
                        "box_type": "aabb",
                        "needs_review": False,
                        "bbox_xyxy": [95.0, 35.0, 130.0, 96.0],
                        "center_xyz": [0.8, 1.1, 0.0],
                        "extent_xyz": [0.5, 0.5, 0.5],
                        "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )


def write_single_object_geometry(path: Path, *, label: str = "unknown object") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
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
                        "detector_label": label,
                        "box_type": "aabb",
                        "needs_review": False,
                        "bbox_xyxy": [42.0, 42.0, 58.0, 58.0],
                        "center_xyz": [0.8, 1.1, 0.0],
                        "extent_xyz": [0.5, 0.5, 0.5],
                        "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_placement_transform_maps_sceneforge_camera_box_to_gltf_axes() -> None:
    transform = placement_transform_to_gltf(
        {
            "center_xyz": [1.0, 2.0, 3.0],
            "extent_xyz": [2.0, 4.0, 6.0],
            "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    )

    assert np.allclose(transform[:3, 3], [1.0, 3.0, -2.0])
    assert np.allclose(transform[:3, :3], np.eye(3) * 4.0)


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
    assert np.allclose(transform[:3, :3], [[4.0, 0.0, 0.0], [0.0, 0.0, 4.0], [0.0, -4.0, 0.0]])


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
        background_margin=1.08,
    )

    assert (output_dir / "scene.glb").is_file()
    assert (output_dir / "scene_alignment.json").is_file()
    assert report["summary"] == {"placement_count": 1, "composed_count": 1, "skipped_count": 0, "failed_count": 0}
    transformed_bounds = np.asarray(report["objects"][0]["transformed_bounds"])
    background_bounds = np.asarray(report["background"]["transformed_bounds"])
    assert background_bounds[0, 0] <= transformed_bounds[0, 0]
    assert background_bounds[1, 0] >= transformed_bounds[1, 0]
    assert background_bounds[0, 1] <= transformed_bounds[0, 1]
    assert background_bounds[1, 1] >= transformed_bounds[1, 1]
    assert background_bounds[0, 2] <= transformed_bounds[0, 2]
    assert background_bounds[1, 2] <= transformed_bounds[1, 2]
    assert np.isclose(background_bounds[1, 0] - background_bounds[0, 0], background_bounds[1, 1] - background_bounds[0, 1])
    assert np.allclose(transformed_bounds[:, [0, 2]], [[-1.0, -4.0], [3.0, 0.0]])


def test_compose_scene_camera_clipped_background_uses_raw_vggt_mesh_with_alignment(tmp_path: Path) -> None:
    background = tmp_path / "background" / "empty_room_mesh.glb"
    objects_dir = tmp_path / "objects"
    object_dir = objects_dir / "01_cube"
    object_geometry = tmp_path / "object_geometry.json"
    output_dir = tmp_path / "scene"
    write_box_glb(background)
    write_background_vggt_fixture(background.parent)
    write_plane_detections_fixture(background.parent / "plane_detections.json")
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
        background_stride=1,
        clip_background_masks=True,
        background_clip_dilation_px=0,
    )

    assert report["background"]["source"] == "vggt_points_camera_clipped"
    assert report["background"]["alignment"] == "plane_camera_guided_room_alignment"
    assert report["background"]["room_alignment"]["method"] in {
        "plane_camera_floor_uniform_alignment_v1",
        "camera_placement_uniform_fit_without_floor_plane",
    }
    assert report["background"]["room_alignment"]["applied_transform_gltf"]
    assert report["background"]["orientation"]["method"] in {
        "fitted_floor_back_wall_normals_to_regularized_axes",
        "identity",
    }
    assert report["background"]["floor_regularization"]["status"] == "applied"
    assert report["background"]["vertex_count"] == 15
    assert report["background"]["texture_source"] == "empty_room_image_uv_projected"
    assert report["background"]["uv_count"] == 15
    assert report["background"]["masked_pixel_ratio"] == 1 / 16
    assert report["background"]["transform_gltf"] != np.eye(4, dtype=np.float64).tolist()
    assert np.isfinite(np.asarray(report["background"]["transformed_bounds"], dtype=np.float64)).all()
    assert report["summary"]["composed_count"] == 1


def test_compose_scene_cli_defaults_to_empty_room_vggt_background() -> None:
    parser = build_parser()

    args = parser.parse_args(["compose-scene"])

    assert args.background == "Output/Latest/background/empty_room_mesh.glb"
    assert args.background_fit == "camera-clipped"


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
    assert report["background"]["floor_y"] <= report["objects"][0]["transformed_bounds"][0][1] + 1e-6
    assert report["background"]["z_back"] < report["objects"][0]["transformed_bounds"][0][2]
    assert report["background"]["wall_top_y"] >= abs(report["background"]["z_back"]) * 0.55


def test_compose_scene_room_corner_projects_empty_room_texture(tmp_path: Path) -> None:
    background_dir = tmp_path / "background"
    background = background_dir / "empty_room_planes.glb"
    objects_dir = tmp_path / "objects"
    object_dir = objects_dir / "01_cube"
    object_geometry = tmp_path / "object_geometry.json"
    output_dir = tmp_path / "scene"
    background_dir.mkdir(parents=True)
    np.save(background_dir / "vggt_points.npy", np.zeros((4, 4, 3), dtype=np.float32))
    Image.new("RGB", (4, 4), (120, 160, 200)).save(background_dir / "empty_room.png")
    write_box_glb(background)
    write_box_glb(object_dir / "hunyuan3d_textured.glb")
    object_dir.mkdir(parents=True, exist_ok=True)
    (object_dir / "metadata.json").write_text(json.dumps({"id": 1}), encoding="utf-8")
    object_geometry.write_text(
        json.dumps(
            {
                "coordinate_contract": {
                    "image_width": 4,
                    "image_height": 4,
                    "fov_degrees": 70.0,
                },
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
    )

    assert report["background"]["texture_source"] == "empty_room_image_camera_projected"
    assert report["background"]["texture_image_path"] == str(background_dir / "empty_room.png")
    assert report["background"]["texture_grid_steps"] == 36
    assert report["background"]["vertex_count"] == 3 * 37 * 37
    assert report["background"]["face_count"] == 3 * 36 * 36 * 2
    assert report["background"]["vertex_colors"] == "projected_empty_room_image_fallback"


def test_compose_scene_does_not_apply_label_specific_floor_contact_cleanup(tmp_path: Path) -> None:
    background = tmp_path / "background.glb"
    objects_dir = tmp_path / "objects"
    table_dir = objects_dir / "01_table"
    object_geometry = tmp_path / "object_geometry.json"
    output_dir = tmp_path / "scene"
    write_box_glb(background)
    write_jagged_bottom_glb(table_dir / "hunyuan3d_textured.glb")
    table_dir.mkdir(parents=True, exist_ok=True)
    (table_dir / "metadata.json").write_text(json.dumps({"id": 1}), encoding="utf-8")
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

    assert report["objects"][0]["mesh_cleanup"] is None


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


def test_choose_object_supports_writes_floor_and_tabletop_records(tmp_path: Path) -> None:
    planes_path = tmp_path / "background" / "plane_detections.json"
    objects_dir = tmp_path / "objects"
    table_dir = objects_dir / "01_table"
    vase_dir = objects_dir / "02_vase"
    chair_dir = objects_dir / "03_chair"
    object_geometry = tmp_path / "objects_vggt" / "object_geometry.json"
    output_dir = tmp_path / "placement"
    write_plane_report(planes_path, floor_z=0.0)
    write_box_glb(table_dir / "hunyuan3d_textured.glb")
    write_box_glb(vase_dir / "hunyuan3d_textured.glb")
    write_box_glb(chair_dir / "hunyuan3d_textured.glb")
    for object_id, object_dir in ((1, table_dir), (2, vase_dir), (3, chair_dir)):
        object_dir.mkdir(parents=True, exist_ok=True)
        (object_dir / "metadata.json").write_text(json.dumps({"id": object_id}), encoding="utf-8")
    write_support_object_geometry(object_geometry)

    report = choose_object_supports(
        object_geometry_path=object_geometry,
        planes_path=planes_path,
        detections_path=None,
        objects_dir=objects_dir,
        output_dir=output_dir,
        object_scale_factor=1.0,
    )

    by_id = {item["detection_id"]: item for item in report["objects"]}
    assert (output_dir / "object_supports.json").is_file()
    assert by_id[1]["support"]["mode"] == "floor_4dof"
    assert by_id[2]["support"]["mode"] == "tabletop_4dof"
    assert by_id[2]["support"]["support_detection_id"] == 1
    assert by_id[3]["support"]["mode"] == "floor_4dof"
    assert report["summary"]["support_modes"] == {"floor_4dof": 2, "tabletop_4dof": 1}


def test_fit_object_placements_and_compose_explicit_records(tmp_path: Path) -> None:
    background = tmp_path / "background.glb"
    planes_path = tmp_path / "background" / "plane_detections.json"
    objects_dir = tmp_path / "objects"
    table_dir = objects_dir / "01_table"
    vase_dir = objects_dir / "02_vase"
    chair_dir = objects_dir / "03_chair"
    object_geometry = tmp_path / "objects_vggt" / "object_geometry.json"
    placement_dir = tmp_path / "placement"
    scene_dir = tmp_path / "scene"
    write_box_glb(background)
    write_plane_report(planes_path, floor_z=0.0)
    write_box_glb(table_dir / "hunyuan3d_textured.glb")
    write_box_glb(vase_dir / "hunyuan3d_textured.glb")
    write_box_glb(chair_dir / "hunyuan3d_textured.glb")
    for object_id, object_dir in ((1, table_dir), (2, vase_dir), (3, chair_dir)):
        object_dir.mkdir(parents=True, exist_ok=True)
        (object_dir / "metadata.json").write_text(json.dumps({"id": object_id}), encoding="utf-8")
    write_support_object_geometry(object_geometry)
    supports = choose_object_supports(
        object_geometry_path=object_geometry,
        planes_path=planes_path,
        detections_path=None,
        objects_dir=objects_dir,
        output_dir=placement_dir,
        object_scale_factor=1.0,
    )
    targets = build_object_fit_targets(
        object_geometry_path=object_geometry,
        supports_path=placement_dir / "object_supports.json",
        objects_dir=objects_dir,
        output_dir=placement_dir,
    )

    placements = fit_object_placements(
        supports_path=placement_dir / "object_supports.json",
        fit_targets_path=placement_dir / "object_fit_targets.json",
        output_dir=placement_dir,
        object_scale_factor=1.0,
    )
    scene_report = compose_scene(
        background_path=background,
        objects_dir=objects_dir,
        object_geometry_path=object_geometry,
        placements_path=placement_dir / "object_placements.json",
        output_dir=scene_dir,
        background_fit="room-corner",
        object_scale_factor=1.0,
        include_review=True,
    )

    assert supports["summary"]["accepted_count"] == 3
    assert targets["summary"]["ready_count"] == 3
    assert placements["summary"]["accepted_count"] == 3
    placement_by_id = {item["detection_id"]: item for item in placements["objects"]}
    table_top = placement_by_id[1]["transformed_bounds"][1][1]
    assert abs(placement_by_id[2]["support"]["support_y_gltf"] - table_top) < 1e-6
    assert placements["objects"][0]["losses"]["silhouette"] is not None
    assert placements["objects"][0]["quality"]["silhouette_proxy"]["method"] == "bbox_projection_proxy"
    assert placements["objects"][0]["quality"]["support_footprint"]["status"] in {"accepted", "warning", "rejected"}
    assert sum(placements["quality"]["status_counts"]["projection"].values()) == 3
    assert placements["quality"]["losses"]["bbox_projection"]["count"] == 3
    assert (placement_dir / "object_fit_targets.json").is_file()
    assert (placement_dir / "object_placements.json").is_file()
    assert (placement_dir / "placement_quality.json").is_file()
    assert scene_report["placement_source"] == "object_placements_json"
    assert scene_report["summary"]["composed_count"] == 3
    by_id = {item["detection_id"]: item for item in scene_report["objects"]}
    assert by_id[2]["support_kind"] == "tabletop"
    assert by_id[2]["support_detection_id"] == 1


def test_compose_explicit_records_recomputes_tabletop_support_after_floor_snap(tmp_path: Path) -> None:
    background = tmp_path / "background.glb"
    objects_dir = tmp_path / "objects"
    table_dir = objects_dir / "01_table"
    vase_dir = objects_dir / "02_vase"
    placements_path = tmp_path / "object_placements.json"
    scene_dir = tmp_path / "scene"
    write_box_glb(background)
    write_jagged_bottom_glb(table_dir / "hunyuan3d_textured.glb")
    write_box_glb(vase_dir / "hunyuan3d_textured.glb")
    table_dir.mkdir(parents=True, exist_ok=True)
    vase_dir.mkdir(parents=True, exist_ok=True)
    (table_dir / "metadata.json").write_text(json.dumps({"id": 1}), encoding="utf-8")
    (vase_dir / "metadata.json").write_text(json.dumps({"id": 2}), encoding="utf-8")
    placements_path.write_text(
        json.dumps(
            {
                "objects": [
                    {
                        "detection_id": 2,
                        "detector_label": "vase",
                        "status": "accepted",
                        "needs_review": False,
                        "mesh_path": str(vase_dir / "hunyuan3d_textured.glb"),
                        "transform_gltf": [[0.2, 0.0, 0.0, 0.0], [0.0, 0.2, 0.0, 0.6], [0.0, 0.0, 0.2, 0.0], [0.0, 0.0, 0.0, 1.0]],
                        "transformed_bounds": [[-0.1, 0.5, -0.1], [0.1, 0.7, 0.1]],
                        "support": {"mode": "tabletop_4dof", "support_kind": "tabletop", "support_detection_id": 1, "support_y_gltf": 0.5},
                    },
                    {
                        "detection_id": 1,
                        "detector_label": "round table",
                        "status": "accepted",
                        "needs_review": False,
                        "mesh_path": str(table_dir / "hunyuan3d_textured.glb"),
                        "transform_gltf": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                        "transformed_bounds": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                        "support": {"mode": "floor_4dof", "support_kind": "floor", "support_y_gltf": -0.5},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    report = compose_scene(
        background_path=background,
        objects_dir=objects_dir,
        object_geometry_path=placements_path,
        placements_path=placements_path,
        output_dir=scene_dir,
        background_fit="room-corner",
        include_review=True,
    )

    by_id = {item["detection_id"]: item for item in report["objects"]}
    table_top = by_id[1]["transformed_bounds"][1][1]
    assert by_id[2]["support_y"] == table_top
    assert by_id[2]["support_y"] != 0.5
    assert np.isclose(by_id[2]["transformed_bounds"][0][1], table_top)


def test_compose_scene_uses_stable_contact_layer_for_dangling_mesh_artifacts(tmp_path: Path) -> None:
    background = tmp_path / "background.glb"
    objects_dir = tmp_path / "objects"
    table_dir = objects_dir / "01_table"
    vase_dir = objects_dir / "02_vase"
    placements_path = tmp_path / "object_placements.json"
    scene_dir = tmp_path / "scene"
    write_box_glb(background)
    write_box_glb(table_dir / "hunyuan3d_textured.glb")
    write_dangling_bottom_glb(vase_dir / "hunyuan3d_textured.glb")
    table_dir.mkdir(parents=True, exist_ok=True)
    vase_dir.mkdir(parents=True, exist_ok=True)
    (table_dir / "metadata.json").write_text(json.dumps({"id": 1}), encoding="utf-8")
    (vase_dir / "metadata.json").write_text(json.dumps({"id": 2}), encoding="utf-8")
    placements_path.write_text(
        json.dumps(
            {
                "objects": [
                    {
                        "detection_id": 1,
                        "detector_label": "round table",
                        "status": "accepted",
                        "needs_review": False,
                        "mesh_path": str(table_dir / "hunyuan3d_textured.glb"),
                        "transform_gltf": [[1.0, 0.0, 0.0, 0.0], [0.0, 0.3, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                        "transformed_bounds": [[-0.5, -0.15, -0.5], [0.5, 0.15, 0.5]],
                        "support": {"mode": "floor_4dof", "support_kind": "floor", "support_y_gltf": -0.15},
                    },
                    {
                        "detection_id": 2,
                        "detector_label": "vase",
                        "status": "accepted",
                        "needs_review": False,
                        "mesh_path": str(vase_dir / "hunyuan3d_textured.glb"),
                        "transform_gltf": [[0.2, 0.0, 0.0, 0.0], [0.0, 0.2, 0.0, 0.5], [0.0, 0.0, 0.2, 0.0], [0.0, 0.0, 0.0, 1.0]],
                        "transformed_bounds": [[-0.1, 0.4, -0.1], [0.1, 0.6, 0.1]],
                        "support": {"mode": "tabletop_4dof", "support_kind": "tabletop", "support_detection_id": 1, "support_y_gltf": 0.6},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    report = compose_scene(
        background_path=background,
        objects_dir=objects_dir,
        object_geometry_path=placements_path,
        placements_path=placements_path,
        output_dir=scene_dir,
        background_fit="room-corner",
        include_review=True,
    )

    by_id = {item["detection_id"]: item for item in report["objects"]}
    table_top = by_id[1]["transformed_bounds"][1][1]
    contact = by_id[2]["support_contact"]
    assert by_id[2]["support_y"] == table_top
    assert np.isclose(contact["contact_y"], table_top)
    assert contact["raw_bottom_y"] < table_top
    assert contact["selected_quantile"] > 0.5
    assert by_id[2]["transformed_bounds"][0][1] < table_top


def test_fit_object_placements_has_unknown_5dof_fallback(tmp_path: Path) -> None:
    planes_path = tmp_path / "background" / "plane_detections.json"
    objects_dir = tmp_path / "objects"
    object_dir = objects_dir / "01_object"
    object_geometry = tmp_path / "objects_vggt" / "object_geometry.json"
    placement_dir = tmp_path / "placement"
    write_empty_plane_report(planes_path)
    write_box_glb(object_dir / "hunyuan3d_textured.glb")
    object_dir.mkdir(parents=True, exist_ok=True)
    (object_dir / "metadata.json").write_text(json.dumps({"id": 1}), encoding="utf-8")
    write_single_object_geometry(object_geometry)

    supports = choose_object_supports(
        object_geometry_path=object_geometry,
        planes_path=planes_path,
        detections_path=None,
        objects_dir=objects_dir,
        output_dir=placement_dir,
        object_scale_factor=1.0,
    )
    build_object_fit_targets(
        object_geometry_path=object_geometry,
        supports_path=placement_dir / "object_supports.json",
        objects_dir=objects_dir,
        output_dir=placement_dir,
    )
    placements = fit_object_placements(
        supports_path=placement_dir / "object_supports.json",
        fit_targets_path=placement_dir / "object_fit_targets.json",
        output_dir=placement_dir,
        object_scale_factor=1.0,
    )

    record = placements["objects"][0]
    assert supports["objects"][0]["support"]["mode"] == "unknown_5dof"
    assert record["degrees_of_freedom"]["model"] == "unknown_support_5dof"
    assert record["render_to_input_optimization"]["method"] == "unknown_support_5dof_discrete_render_proxy_v1"
    assert record["render_to_input_optimization"]["candidate_count"] > 0
    assert record["needs_review"] is True


def test_support_footprint_outside_plane_marks_review(tmp_path: Path) -> None:
    planes_path = tmp_path / "background" / "plane_detections.json"
    objects_dir = tmp_path / "objects"
    object_dir = objects_dir / "01_chair"
    object_geometry = tmp_path / "objects_vggt" / "object_geometry.json"
    placement_dir = tmp_path / "placement"
    write_tiny_floor_report(planes_path)
    write_box_glb(object_dir / "hunyuan3d_textured.glb")
    object_dir.mkdir(parents=True, exist_ok=True)
    (object_dir / "metadata.json").write_text(json.dumps({"id": 1}), encoding="utf-8")
    write_single_object_geometry(object_geometry, label="chair")

    choose_object_supports(
        object_geometry_path=object_geometry,
        planes_path=planes_path,
        detections_path=None,
        objects_dir=objects_dir,
        output_dir=placement_dir,
        object_scale_factor=1.0,
    )
    build_object_fit_targets(
        object_geometry_path=object_geometry,
        supports_path=placement_dir / "object_supports.json",
        objects_dir=objects_dir,
        output_dir=placement_dir,
    )
    placements = fit_object_placements(
        supports_path=placement_dir / "object_supports.json",
        fit_targets_path=placement_dir / "object_fit_targets.json",
        output_dir=placement_dir,
        object_scale_factor=1.0,
    )

    record = placements["objects"][0]
    assert record["support"]["mode"] == "floor_4dof"
    assert record["quality"]["support_footprint"]["status"] == "rejected"
    assert "support_footprint_rejected" in record["quality"]["warnings"]
    assert record["needs_review"] is True


def test_fit_object_placements_reports_vggt_point_loss(tmp_path: Path) -> None:
    planes_path = tmp_path / "background" / "plane_detections.json"
    objects_dir = tmp_path / "objects"
    object_dir = objects_dir / "01_chair"
    object_geometry = tmp_path / "objects_vggt" / "object_geometry.json"
    points_xyz = tmp_path / "objects_vggt" / "regions" / "01_chair" / "points.xyz"
    mask_path = tmp_path / "objects" / "01_chair" / "full_mask.png"
    placement_dir = tmp_path / "placement"
    write_plane_report(planes_path, floor_z=0.0)
    write_box_glb(object_dir / "hunyuan3d_textured.glb")
    object_dir.mkdir(parents=True, exist_ok=True)
    points_xyz.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(points_xyz, np.asarray([[0.0, 1.0, 0.0], [0.1, 1.0, 0.1], [-0.1, 1.0, 0.1]], dtype=np.float32))
    (object_dir / "metadata.json").write_text(json.dumps({"id": 1}), encoding="utf-8")
    object_geometry.parent.mkdir(parents=True, exist_ok=True)
    mask_pixels = np.zeros((100, 100), dtype=np.uint8)
    mask_pixels[30:90, 35:65] = 255
    Image.fromarray(mask_pixels).save(mask_path)
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
                        "detector_label": "chair",
                        "box_type": "aabb",
                        "needs_review": False,
                        "bbox_xyxy": [35.0, 30.0, 65.0, 90.0],
                        "center_xyz": [0.0, 1.0, 0.0],
                        "extent_xyz": [0.5, 0.5, 0.5],
                        "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        "mask_path": str(mask_path),
                        "artifacts": {"points_xyz": str(points_xyz)},
                        "point_count": 3,
                        "valid_point_ratio": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    choose_object_supports(
        object_geometry_path=object_geometry,
        planes_path=planes_path,
        detections_path=None,
        objects_dir=objects_dir,
        output_dir=placement_dir,
        object_scale_factor=1.0,
    )
    targets = build_object_fit_targets(
        object_geometry_path=object_geometry,
        supports_path=placement_dir / "object_supports.json",
        objects_dir=objects_dir,
        output_dir=placement_dir,
    )
    placements = fit_object_placements(
        supports_path=placement_dir / "object_supports.json",
        fit_targets_path=placement_dir / "object_fit_targets.json",
        output_dir=placement_dir,
        object_scale_factor=1.0,
    )

    record = placements["objects"][0]
    optimization = record["render_to_input_optimization"]
    assert Path(targets["objects"][0]["visible_points_scene_path"]).is_file()
    assert optimization["vggt_candidate_fit"]["status"] == "accepted"
    assert optimization["vggt_candidate_fit"]["loss_weight"] > 0.0
    assert optimization["vggt_candidate_fit"]["optimized"]["loss"] is not None
    assert optimization["mask_candidate_fit"]["status"] == "accepted"
    assert optimization["mask_candidate_fit"]["loss_weight"] > 0.0
    assert optimization["mask_candidate_fit"]["optimized"]["iou"] is not None
    assert optimization["optimized_bbox_loss"] is not None
    assert record["losses"]["vggt_points"] is not None
    assert record["quality"]["vggt_points"]["status"] == "accepted"
    assert record["quality"]["silhouette_render"]["status"] == "accepted"
    assert record["quality"]["silhouette_visibility"]["status"] == "accepted"


def test_explicit_placement_cli_commands(tmp_path: Path) -> None:
    planes_path = tmp_path / "background" / "plane_detections.json"
    objects_dir = tmp_path / "objects"
    table_dir = objects_dir / "01_table"
    vase_dir = objects_dir / "02_vase"
    chair_dir = objects_dir / "03_chair"
    object_geometry = tmp_path / "objects_vggt" / "object_geometry.json"
    placement_dir = tmp_path / "placement"
    write_plane_report(planes_path, floor_z=0.0)
    write_box_glb(table_dir / "hunyuan3d_textured.glb")
    write_box_glb(vase_dir / "hunyuan3d_textured.glb")
    write_box_glb(chair_dir / "hunyuan3d_textured.glb")
    for object_id, object_dir in ((1, table_dir), (2, vase_dir), (3, chair_dir)):
        object_dir.mkdir(parents=True, exist_ok=True)
        (object_dir / "metadata.json").write_text(json.dumps({"id": object_id}), encoding="utf-8")
    write_support_object_geometry(object_geometry)

    choose_result = run_cli(
        [
            "choose-object-supports",
            "--object-geometry",
            str(object_geometry),
            "--planes",
            str(planes_path),
            "--detections",
            "",
            "--objects",
            str(objects_dir),
            "--output",
            str(placement_dir),
            "--object-scale-factor",
            "1.0",
        ]
    )
    build_result = run_cli(
        [
            "build-object-fit-targets",
            "--object-geometry",
            str(object_geometry),
            "--supports",
            str(placement_dir / "object_supports.json"),
            "--objects",
            str(objects_dir),
            "--output",
            str(placement_dir),
        ]
    )
    fit_result = run_cli(
        [
            "fit-object-placements",
            "--supports",
            str(placement_dir / "object_supports.json"),
            "--fit-targets",
            str(placement_dir / "object_fit_targets.json"),
            "--output",
            str(placement_dir),
            "--object-scale-factor",
            "1.0",
        ]
    )

    assert choose_result.returncode == 0, choose_result.stderr
    assert build_result.returncode == 0, build_result.stderr
    assert fit_result.returncode == 0, fit_result.stderr
    assert (placement_dir / "object_supports.json").is_file()
    assert (placement_dir / "object_fit_targets.json").is_file()
    assert (placement_dir / "object_placements.json").is_file()


def test_compose_scene_does_not_apply_chair_specific_scale_or_pose(tmp_path: Path) -> None:
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
    assert "label_scale_factors" not in report
    assert "spacing_targets" not in report
    assert "orientation_targets" not in report
    assert "label_scale_factor" not in by_id[2]
    assert "spacing_delta_gltf" not in by_id[2]
    assert "semantic_orientation_kind" not in by_id[2]
    assert "semantic_yaw_radians" not in by_id[2]
    chair_extent_x = by_id[2]["transformed_bounds"][1][0] - by_id[2]["transformed_bounds"][0][0]
    assert np.isclose(chair_extent_x, 1.0)


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
    assert optimization["orientation_search"]["yaw_candidates"]
    assert optimization["orientation_search"]["selected_yaw"] is not None
    assert optimization["orientation_search"]["loss_breakdown"]["candidate_count"] == optimization["candidate_count"]
    assert (output_dir / "input_vs_projection_overlay.png").is_file()


def test_large_image_target_omits_extreme_shrink_scale_candidates(tmp_path: Path) -> None:
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
                        "bbox_xyxy": [20.0, 10.0, 80.0, 90.0],
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

    optimization = report["objects"][0]["render_to_input_optimization"]
    assert optimization["minimum_scale_reason"] == "large_image_target"
    assert optimization["minimum_scale_delta"] == 0.7
    assert min(optimization["scale_candidates"]) == 0.7


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


def test_mesh_facing_prior_uses_vertical_asymmetry_without_label() -> None:
    import trimesh

    seat = trimesh.creation.box(extents=(1.0, 0.18, 1.0))
    back = trimesh.creation.box(extents=(1.0, 0.9, 0.12))
    back.apply_translation((0.0, 0.42, -0.45))
    mesh = trimesh.util.concatenate([seat, back])

    transform = np.eye(4, dtype=np.float64)
    prior = mesh_facing_prior_from_target(
        meshes=[mesh],
        transform=transform,
        facing_target_gltf=[0.0, 0.0, 2.0],
    )
    flipped = transform.copy()
    flipped[:3, :3] = yaw_rotation_gltf(np.pi)

    assert prior["available"] is True
    assert prior["asymmetry"]["offset_ratio"] > 0.12
    assert facing_prior_loss(prior, transform) < 0.05
    assert facing_prior_loss(prior, flipped) > 0.90


def test_visibility_reconciliation_clears_occluded_bottom_review_without_label() -> None:
    objects = [
        {
            "detection_id": 9,
            "status": "accepted",
            "needs_review": True,
            "reason": "occluded_bottom_edge_tolerated",
            "quality": {
                "projection_status": "accepted_occluded_bottom",
                "support_status": "accepted",
                "collision_status": "accepted",
                "silhouette_visibility": {
                    "status": "accepted",
                    "occluder_detection_ids": [3],
                    "visible_area_px": 120,
                    "occluded_area_px": 80,
                    "occluded_area_ratio": 0.4,
                },
            },
        }
    ]

    report = reconcile_visibility_explained_projection_reviews(objects)

    assert report["resolved_detection_ids"] == [9]
    assert objects[0]["needs_review"] is False
    assert objects[0]["reason"] is None
    assert objects[0]["quality"]["projection_review_resolution"]["status"] == "resolved"


def test_physical_size_prior_uses_volume_scale_without_label() -> None:
    source_bounds = np.asarray([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.eye(3, dtype=np.float64) * 0.5

    prior = physical_size_prior_from_target(
        source_bounds=source_bounds,
        transform=transform,
        target_extent_gltf={
            "target_extent_gltf": [1.0, 1.0, 1.0],
            "target_volume_gltf": 1.0,
        },
    )
    corrected = transform.copy()
    corrected[:3, :3] *= prior["scale_candidate"]

    initial_volume = np.prod(transformed_bounds_from_source_bounds(source_bounds, transform)[1] - transformed_bounds_from_source_bounds(source_bounds, transform)[0])
    corrected_volume = np.prod(transformed_bounds_from_source_bounds(source_bounds, corrected)[1] - transformed_bounds_from_source_bounds(source_bounds, corrected)[0])

    assert prior["available"] is True
    assert prior["volume_scale_candidate"] == 2.0
    assert initial_volume < 0.2
    assert corrected_volume == 1.0
    assert physical_size_prior_loss(prior, source_bounds, corrected) == 0.0


def test_support_candidate_transform_preserves_bottom_pivot() -> None:
    source_bounds = np.asarray([[-1.0, 0.0, -1.0], [1.0, 2.0, 1.0]], dtype=np.float64)
    support = {"support_kind": "floor", "support_y": 0.0}
    pivot = support_plane_pivot_local(source_bounds, support)
    assert pivot["method"] == "bottom_center_support_pivot_v1"
    assert np.allclose(pivot["pivot_local"], [0.0, -0.5, 0.0])

    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = [4.0, 5.0, 6.0]
    delta = np.asarray([0.25, 0.0, -0.5], dtype=np.float64)
    candidate = candidate_transform(
        best_transform=transform,
        delta=delta,
        yaw=float(np.pi / 2.0),
        scale=1.35,
        pivot_local=np.asarray(pivot["pivot_local"], dtype=np.float64),
    )

    local = np.asarray(pivot["pivot_local"], dtype=np.float64)
    before = transform[:3, :3] @ local + transform[:3, 3]
    after = candidate[:3, :3] @ local + candidate[:3, 3]
    assert np.allclose(after, before + delta)


def test_projection_residual_adds_planar_translation_candidates() -> None:
    source_bounds = np.asarray([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] *= 0.35
    transform[:3, 3] = [-0.25, 0.0, -2.0]
    contract = {
        "image_width": 800,
        "image_height": 600,
        "fov_degrees": 70.0,
    }
    base = projected_transform_bbox(source_bounds, transform, contract)
    assert base is not None
    target = np.asarray(base, dtype=np.float64) + np.array([20.0, 0.0, 20.0, 0.0], dtype=np.float64)

    dx, dz, report = translation_candidates_for_projection_residual(
        source_bounds=source_bounds,
        projection_vertices=np.asarray(
            [
                [-0.5, -0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, 0.5, 0.5],
                [0.5, -0.5, -0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, -0.5],
                [0.5, 0.5, 0.5],
            ],
            dtype=np.float64,
        ),
        transform=transform,
        target_bbox=target,
        coordinate_contract=contract,
        support_y=-0.175,
        pivot_local=np.asarray([0.0, -0.5, 0.0], dtype=np.float64),
    )

    assert report["status"] == "accepted"
    assert report["method"] == "projected_bbox_residual_planar_translation_candidates_v1"
    assert any(value > 0.0 for value in dx)
    assert report["added_candidate_count"] == len(report["added_candidates"])


def test_floor_occupancy_refit_rejects_projection_regression() -> None:
    initial = {
        "render_to_input_optimization": {
            "optimized_bbox_loss": 0.4,
            "optimized_loss": 1.2,
        }
    }
    candidate = {
        "render_to_input_optimization": {
            "optimized_bbox_loss": 0.8,
            "optimized_loss": 1.0,
        }
    }

    acceptance = floor_occupancy_refit_acceptance(
        initial=initial,
        candidate=candidate,
        initial_overlap=0.02,
        optimized_overlap=0.0,
        avoidance_report={"optimized_loss": 0.0},
    )

    assert acceptance["accepted"] is False
    assert acceptance["reason"] == "projection_loss_degraded"


def test_floor_occupancy_refit_accepts_quality_preserving_overlap_fix() -> None:
    initial = {
        "render_to_input_optimization": {
            "optimized_bbox_loss": 0.4,
            "optimized_loss": 1.2,
        }
    }
    candidate = {
        "render_to_input_optimization": {
            "optimized_bbox_loss": 0.45,
            "optimized_loss": 1.1,
        }
    }

    acceptance = floor_occupancy_refit_acceptance(
        initial=initial,
        candidate=candidate,
        initial_overlap=0.02,
        optimized_overlap=0.0,
        avoidance_report={"optimized_loss": 0.0},
    )

    assert acceptance["accepted"] is True
    assert acceptance["reason"] == "overlap_improved_without_quality_regression"


def test_floor_occupancy_refit_accepts_projection_improvement_with_total_tradeoff() -> None:
    initial = {
        "render_to_input_optimization": {
            "optimized_bbox_loss": 0.65,
            "optimized_loss": 1.0,
        }
    }
    candidate = {
        "render_to_input_optimization": {
            "optimized_bbox_loss": 0.2,
            "optimized_loss": 1.25,
        }
    }

    acceptance = floor_occupancy_refit_acceptance(
        initial=initial,
        candidate=candidate,
        initial_overlap=0.02,
        optimized_overlap=0.0,
        avoidance_report={"optimized_loss": 0.0},
    )

    assert acceptance["accepted"] is True
    assert acceptance["projection_improved"] is True
