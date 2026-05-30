from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from SceneGeometry.VGGT.pipeline import scene_point_to_gltf_vertex
from SceneGeometry.coordinate_contract import DEFAULT_FOV_DEGREES, camera_fusion_contract


SCHEMA_VERSION = 1


def fit_empty_room_planes(
    *,
    background_dir: str | Path,
    output_dir: str | Path | None = None,
    stride: int = 8,
    mesh_name: str = "empty_room_planes.glb",
    align_xyz: bool = True,
    padding_ratio: float = 0.08,
) -> dict[str, Any]:
    if not align_xyz:
        raise ValueError("Only XYZ-aligned V1 plane export is supported.")
    background_dir = Path(background_dir)
    output_dir = Path(output_dir) if output_dir is not None else background_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    points_path = background_dir / "vggt_points.npy"
    image_path = background_dir / "empty_room.png"
    if not points_path.is_file():
        raise FileNotFoundError(f"Missing empty-room VGGT points: {points_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Missing empty-room image: {image_path}")

    points = np.load(points_path).astype(np.float32)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"Expected point map with shape HxWx3, got {points.shape}")
    height, width = points.shape[:2]
    image = Image.open(image_path).convert("RGB")
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    rgb = np.asarray(image, dtype=np.uint8)

    regions = sample_plane_regions(points, stride=stride)
    evidence = {name: fit_plane_evidence(sampled) for name, sampled in regions.items()}
    structural = build_xyz_aligned_room_planes(points, rgb, padding_ratio=padding_ratio)
    mesh_path = output_dir / mesh_name
    mesh_stats = write_planes_glb(structural, mesh_path, image=image)
    report = {
        "schema_version": SCHEMA_VERSION,
        "background_dir": str(background_dir),
        "empty_room_image_path": str(image_path),
        "vggt_points_path": str(points_path),
        "image_width": int(width),
        "image_height": int(height),
        "coordinate_contract": camera_fusion_contract(image_width=width, image_height=height),
        "align_xyz": bool(align_xyz),
        "stride": int(stride),
        "padding_ratio": float(padding_ratio),
        "artifacts": {
            "planes_json": str(output_dir / "plane_detections.json"),
            "planes_glb": str(mesh_path),
        },
        "planes": [
            plane_report(plane, evidence.get(plane["id"]))
            for plane in structural
        ],
        "mesh": mesh_stats,
        "model_info": {
            "backend": "empty_room_vggt_xyz_planes",
            "point_source": "background/vggt_points.npy",
            "regularization": "xyz_aligned_scene_camera_axes",
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "plane_detections.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def sample_plane_regions(points: np.ndarray, *, stride: int) -> dict[str, np.ndarray]:
    height, width = points.shape[:2]
    stride = max(1, int(stride))
    return {
        "floor": valid_points(points[int(0.58 * height) :: stride, ::stride]),
        "back_wall": valid_points(points[: int(0.68 * height) : stride, ::stride]),
        "right_wall": valid_points(points[: int(0.72 * height) : stride, int(0.45 * width) :: stride]),
    }


def valid_points(points: np.ndarray) -> np.ndarray:
    flat = points.reshape(-1, 3)
    return flat[np.isfinite(flat).all(axis=1)]


def fit_plane_evidence(points: np.ndarray) -> dict[str, Any]:
    if len(points) < 3:
        return {
            "support_count": int(len(points)),
            "fitted_center_xyz": None,
            "fitted_normal_xyz": None,
            "fit_residual": None,
        }
    center = points.mean(axis=0)
    centered = points - center
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)
    normal = eigenvectors[:, order[0]]
    residuals = np.abs(centered @ normal)
    return {
        "support_count": int(len(points)),
        "fitted_center_xyz": float_list(center),
        "fitted_normal_xyz": float_list(normal),
        "fit_residual": float(np.mean(residuals)),
    }


def build_xyz_aligned_room_planes(points: np.ndarray, rgb: np.ndarray, *, padding_ratio: float) -> list[dict[str, Any]]:
    height, width = points.shape[:2]
    floor_points = valid_points(points[int(0.58 * height) :, :])
    wall_points = valid_points(points[: int(0.72 * height), :])
    all_points = valid_points(points)
    if len(floor_points) < 8 or len(wall_points) < 8 or len(all_points) < 8:
        raise ValueError("Not enough valid VGGT points to fit empty-room planes.")
    x_min, x_max = quantile_bounds(all_points[:, 0], padding_ratio)
    floor_y_min, floor_y_max = quantile_bounds(floor_points[:, 1], padding_ratio)
    wall_y = float(np.quantile(wall_points[:, 1], 0.985))
    y_front = min(floor_y_min, float(np.quantile(all_points[:, 1], 0.08)))
    y_back = max(floor_y_max, wall_y)
    y_back += max((y_back - y_front) * 0.14, 0.12)
    floor_z = float(np.quantile(floor_points[:, 2], 0.18))
    wall_top_z = float(np.quantile(wall_points[:, 2], 0.97))
    wall_top_z = max(wall_top_z, floor_z + 0.85)
    side_x = x_max + max((x_max - x_min) * 0.22, 0.16)
    floor_color = median_color(rgb[int(0.62 * height) :, :])
    wall_color = median_color(rgb[: int(0.55 * height), :])
    side_color = tuple(max(0, int(value) - 6) for value in wall_color)
    return [
        make_plane(
            plane_id="floor",
            subtype="floor",
            vertices=[
                [x_min, y_front, floor_z],
                [side_x, y_front, floor_z],
                [side_x, y_back, floor_z],
                [x_min, y_back, floor_z],
            ],
            normal=[0.0, 0.0, 1.0],
            color=floor_color,
        ),
        make_plane(
            plane_id="back_wall",
            subtype="wall",
            vertices=[
                [x_min, y_back, floor_z],
                [side_x, y_back, floor_z],
                [side_x, y_back, wall_top_z],
                [x_min, y_back, wall_top_z],
            ],
            normal=[0.0, -1.0, 0.0],
            color=wall_color,
        ),
        make_plane(
            plane_id="right_wall",
            subtype="wall",
            vertices=[
                [side_x, y_front, floor_z],
                [side_x, y_back, floor_z],
                [side_x, y_back, wall_top_z],
                [side_x, y_front, wall_top_z],
            ],
            normal=[-1.0, 0.0, 0.0],
            color=side_color,
        ),
    ]


def quantile_bounds(values: np.ndarray, padding_ratio: float) -> tuple[float, float]:
    low = float(np.quantile(values, 0.04))
    high = float(np.quantile(values, 0.96))
    pad = max((high - low) * float(padding_ratio), 0.05)
    return low - pad, high + pad


def median_color(values: np.ndarray) -> tuple[int, int, int, int]:
    color = np.median(values.reshape(-1, 3), axis=0)
    return (int(color[0]), int(color[1]), int(color[2]), 255)


def make_plane(
    *,
    plane_id: str,
    subtype: str,
    vertices: list[list[float]],
    normal: list[float],
    color: tuple[int, int, int, int],
) -> dict[str, Any]:
    vertex_array = np.asarray(vertices, dtype=np.float64)
    center = vertex_array.mean(axis=0)
    return {
        "id": plane_id,
        "label": subtype,
        "primitive_label": "plane",
        "plane_subtype": subtype,
        "vertices_xyz": vertex_array.tolist(),
        "center_xyz": float_list(center),
        "normal_xyz": normal,
        "color_rgba": list(color),
        "needs_review": False,
        "failure_reason": None,
        "source": "xyz_regularized_empty_room_vggt",
    }


def plane_report(plane: dict[str, Any], evidence: dict[str, Any] | None) -> dict[str, Any]:
    vertices = np.asarray(plane["vertices_xyz"], dtype=np.float64)
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    report = dict(plane)
    report.update(
        plane_extent_xyz=float_list(extent),
        normal_confidence=1.0,
        support_count=int((evidence or {}).get("support_count", 0)),
        fitted_center_xyz=(evidence or {}).get("fitted_center_xyz"),
        fitted_normal_xyz=(evidence or {}).get("fitted_normal_xyz"),
        fit_residual=(evidence or {}).get("fit_residual"),
    )
    return report


def write_planes_glb(
    planes: list[dict[str, Any]],
    output_path: Path,
    *,
    image: Image.Image | None = None,
    grid_steps: int = 36,
) -> dict[str, Any]:
    try:
        import trimesh
        from trimesh.visual.material import PBRMaterial
        from trimesh.visual.texture import TextureVisuals
    except Exception as exc:
        raise RuntimeError("Exporting empty-room planes requires trimesh from requirements.txt.") from exc
    scene = trimesh.Scene()
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8) if image is not None else None
    material = (
        PBRMaterial(
            name="empty_room_projected",
            baseColorTexture=image.copy(),
            baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            emissiveTexture=image.copy(),
            emissiveFactor=[0.45, 0.45, 0.45],
            roughnessFactor=0.85,
            metallicFactor=0.0,
            doubleSided=True,
        )
        if image is not None
        else None
    )
    bounds: list[np.ndarray] = []
    vertex_count = 0
    face_count = 0
    for plane in planes:
        vertices_scene, faces, colors, uvs = textured_plane_geometry(plane, rgb=rgb, grid_steps=grid_steps)
        vertices_gltf = np.asarray([scene_point_to_gltf_vertex(vertex) for vertex in vertices_scene], dtype=np.float32)
        mesh = trimesh.Trimesh(vertices=vertices_gltf, faces=faces, vertex_colors=colors, process=False)
        if material is not None:
            mesh.visual = TextureVisuals(uv=uvs, material=material)
        scene.add_geometry(mesh, geom_name=plane["id"], node_name=plane["id"])
        bounds.append(np.asarray(mesh.bounds, dtype=np.float64))
        vertex_count += int(len(vertices_gltf))
        face_count += int(len(faces))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(output_path)
    merged = merge_bounds(bounds)
    return {
        "mesh_glb": str(output_path),
        "axis_transform": "gltf_x_image_right_y_image_up_z_back_toward_camera",
        "object_count": len(planes),
        "vertex_count": vertex_count,
        "face_count": face_count,
        "grid_steps": int(grid_steps),
        "texture_source": "empty_room_image_uv_projected" if image is not None else None,
        "vertex_colors": "projected_empty_room_image_fallback" if rgb is not None else "plane_median_color",
        "bounds_gltf": merged.tolist(),
    }


def textured_plane_geometry(
    plane: dict[str, Any],
    *,
    rgb: np.ndarray | None,
    grid_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    corners = np.asarray(plane["vertices_xyz"], dtype=np.float64)
    steps = max(1, int(grid_steps))
    vertices: list[np.ndarray] = []
    colors: list[tuple[int, int, int, int]] = []
    uvs: list[tuple[float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for row in range(steps + 1):
        v = row / steps
        left = corners[0] * (1.0 - v) + corners[3] * v
        right = corners[1] * (1.0 - v) + corners[2] * v
        for col in range(steps + 1):
            u = col / steps
            point = left * (1.0 - u) + right * u
            vertices.append(point)
            colors.append(plane_vertex_color(plane, point, rgb))
            uvs.append(plane_vertex_uv(point, rgb))

    row_width = steps + 1
    for row in range(steps):
        for col in range(steps):
            v00 = row * row_width + col
            v10 = v00 + 1
            v01 = (row + 1) * row_width + col
            v11 = v01 + 1
            faces.append((v00, v10, v11))
            faces.append((v00, v11, v01))
    return (
        np.asarray(vertices, dtype=np.float32),
        np.asarray(faces, dtype=np.int64),
        np.asarray(colors, dtype=np.uint8),
        np.asarray(uvs, dtype=np.float32),
    )


def plane_vertex_color(plane: dict[str, Any], point: np.ndarray, rgb: np.ndarray | None) -> tuple[int, int, int, int]:
    base = np.asarray(plane["color_rgba"], dtype=np.float64)
    if rgb is not None:
        projected = project_scene_point_to_pixel(point, image_width=rgb.shape[1], image_height=rgb.shape[0])
        if projected is not None:
            x, y = projected
            base[:3] = np.asarray(rgb[y, x], dtype=np.float64)
    base[:3] = add_subtle_surface_variation(base[:3], point, str(plane.get("plane_subtype") or "plane"))
    return tuple(int(np.clip(value, 0, 255)) for value in base)


def plane_vertex_uv(point: np.ndarray, rgb: np.ndarray | None) -> tuple[float, float]:
    if rgb is None:
        return (0.0, 0.0)
    projected = project_scene_point_to_pixel(point, image_width=rgb.shape[1], image_height=rgb.shape[0])
    if projected is None:
        return (0.0, 0.0)
    x, y = projected
    u = x / max(rgb.shape[1] - 1, 1)
    v = 1.0 - y / max(rgb.shape[0] - 1, 1)
    return (float(np.clip(u, 0.0, 1.0)), float(np.clip(v, 0.0, 1.0)))


def project_scene_point_to_pixel(
    point: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    fov_degrees: float = DEFAULT_FOV_DEGREES,
) -> tuple[int, int] | None:
    x, depth, z = np.asarray(point, dtype=np.float64)
    if not np.isfinite([x, depth, z]).all() or depth <= 1e-6:
        return None
    focal = (image_width / 2.0) / math.tan(math.radians(float(fov_degrees)) / 2.0)
    pixel_x = image_width / 2.0 + (x / depth) * focal
    pixel_y = image_height / 2.0 - (z / depth) * focal
    if not np.isfinite([pixel_x, pixel_y]).all():
        return None
    return (
        int(np.clip(round(pixel_x), 0, image_width - 1)),
        int(np.clip(round(pixel_y), 0, image_height - 1)),
    )


def add_subtle_surface_variation(color: np.ndarray, point: np.ndarray, subtype: str) -> np.ndarray:
    amplitude = 8.0 if subtype == "floor" else 5.0
    wave = (
        math.sin(float(point[0]) * 37.0 + float(point[1]) * 11.0)
        + math.sin(float(point[1]) * 23.0 + float(point[2]) * 29.0)
    ) * 0.5
    return color + wave * amplitude


def merge_bounds(bounds: list[np.ndarray]) -> np.ndarray:
    minimum = np.min([item[0] for item in bounds], axis=0)
    maximum = np.max([item[1] for item in bounds], axis=0)
    return np.stack([minimum, maximum], axis=0)


def float_list(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values).reshape(-1)]
