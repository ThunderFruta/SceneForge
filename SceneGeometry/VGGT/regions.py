from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


SCHEMA_VERSION = 1


@dataclass
class MaskInfo:
    mask: np.ndarray
    path: Path | None
    source: str


@dataclass
class FitResult:
    box_type: str
    center_xyz: list[float] | None
    extent_xyz: list[float] | None
    rotation_matrix: list[list[float]] | None
    needs_review: bool
    failure_reason: str | None


def fit_vggt_boxes(
    *,
    detections_path: str | Path,
    objects_dir: str | Path,
    vggt_dir: str | Path,
    output_dir: str | Path,
    box_mode: str = "auto",
    min_valid_points: int = 64,
) -> dict[str, Any]:
    detections_path = Path(detections_path)
    objects_dir = Path(objects_dir)
    vggt_dir = Path(vggt_dir)
    output_dir = Path(output_dir)
    regions_dir = output_dir / "regions"
    regions_dir.mkdir(parents=True, exist_ok=True)

    detections = load_json(detections_path)
    points_path = vggt_dir / "vggt_points.npy"
    camera_path = vggt_dir / "vggt_camera.json"
    geometry_path = vggt_dir / "vggt_geometry.json"
    if not points_path.is_file():
        raise FileNotFoundError(f"VGGT point map does not exist: {points_path}")
    points = np.load(points_path).astype(np.float32)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"Expected VGGT points with shape HxWx3, got {points.shape}")
    height, width = points.shape[:2]

    object_dirs = index_object_dirs(objects_dir)
    geometry = load_json(geometry_path) if geometry_path.is_file() else {}
    records = []
    for detection in detections.get("objects", []):
        record = process_detection(
            detection=detection,
            object_dirs=object_dirs,
            objects_dir=objects_dir,
            points=points,
            image_size=(width, height),
            regions_dir=regions_dir,
            box_mode=box_mode,
            min_valid_points=min_valid_points,
        )
        records.append(record)

    boxes_obj_path = output_dir / "vggt_boxes.obj"
    overlay_png_path = output_dir / "vggt_regions_overlay.png"
    write_boxes_obj(records, boxes_obj_path)
    write_regions_overlay(records, detections, image_size=(width, height), output_path=overlay_png_path)
    report = {
        "schema_version": SCHEMA_VERSION,
        "detections_path": str(detections_path),
        "objects_dir": str(objects_dir),
        "vggt_dir": str(vggt_dir),
        "vggt_points_path": str(points_path),
        "vggt_camera_path": str(camera_path) if camera_path.is_file() else None,
        "artifacts": {
            "boxes_obj": str(boxes_obj_path),
            "regions_overlay_png": str(overlay_png_path),
        },
        "coordinate_contract": geometry.get("coordinate_contract") or detections.get("model_info", {}).get("fusion_contract"),
        "box_mode": box_mode,
        "min_valid_points": int(min_valid_points),
        "objects": records,
        "summary": {
            "detection_count": len(records),
            "fit_count": sum(1 for item in records if item["box_type"] != "failed"),
            "failed_count": sum(1 for item in records if item["box_type"] == "failed"),
            "needs_review_count": sum(1 for item in records if item["needs_review"]),
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    output_path = output_dir / "object_geometry.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def process_detection(
    *,
    detection: dict[str, Any],
    object_dirs: dict[int, Path],
    objects_dir: Path,
    points: np.ndarray,
    image_size: tuple[int, int],
    regions_dir: Path,
    box_mode: str,
    min_valid_points: int,
) -> dict[str, Any]:
    detection_id = int(detection.get("id", 0))
    label = str(detection.get("detector_label", "object"))
    region_dir = regions_dir / f"{detection_id:02d}_{slugify(label)}"
    region_dir.mkdir(parents=True, exist_ok=True)

    base = base_record(detection, region_dir)
    mask_info = load_detection_mask(detection, object_dirs.get(detection_id), image_size)
    if mask_info is None:
        return failed_record(base, "missing_mask")

    mask = mask_info.mask
    active_pixels = int(mask.sum())
    if active_pixels <= 0:
        base.update(mask_path=str(mask_info.path) if mask_info.path else None, mask_source=mask_info.source)
        return failed_record(base, "empty_mask")

    sampled_points = sample_points_for_mask(points, mask)
    point_count = int(sampled_points.shape[0])
    coverage_ratio = active_pixels / float(mask.size)
    valid_point_ratio = point_count / float(active_pixels)
    points_xyz_path = region_dir / "points.xyz"
    points_obj_path = region_dir / "points.obj"
    mask_png_path = region_dir / "mask.png"
    valid_points_png_path = region_dir / "valid_points.png"
    point_distance_png_path = region_dir / "point_distance.png"
    write_mask_png(mask, mask_png_path)
    write_valid_points_png(points, mask, valid_points_png_path)
    write_point_distance_png(points, mask, point_distance_png_path)
    if point_count:
        write_points_xyz(sampled_points, points_xyz_path)
        write_region_surface_obj(points, mask, points_obj_path)

    base.update(
        mask_path=str(mask_info.path) if mask_info.path else None,
        mask_source=mask_info.source,
        point_count=point_count,
        valid_point_ratio=float(valid_point_ratio),
        coverage_ratio=float(coverage_ratio),
        artifacts={
            "points_xyz": str(points_xyz_path) if point_count else None,
            "points_obj": str(points_obj_path) if point_count else None,
            "mask_png": str(mask_png_path),
            "valid_points_png": str(valid_points_png_path),
            "point_distance_png": str(point_distance_png_path),
        },
    )
    if point_count < min_valid_points:
        return failed_record(base, "too_few_points")

    fit = fit_box(sampled_points, box_mode=box_mode)
    base.update(
        box_type=fit.box_type,
        center_xyz=fit.center_xyz,
        extent_xyz=fit.extent_xyz,
        rotation_matrix=fit.rotation_matrix,
        needs_review=fit.needs_review,
        failure_reason=fit.failure_reason,
    )
    return base


def base_record(detection: dict[str, Any], region_dir: Path) -> dict[str, Any]:
    return {
        "detection_id": int(detection.get("id", 0)),
        "detector_label": detection.get("detector_label"),
        "detector_confidence": detection.get("detector_confidence"),
        "bbox_xyxy": detection.get("bbox_xyxy"),
        "region_dir": str(region_dir),
        "mask_path": None,
        "mask_source": None,
        "point_count": 0,
        "valid_point_ratio": 0.0,
        "coverage_ratio": 0.0,
        "box_type": "failed",
        "center_xyz": None,
        "extent_xyz": None,
        "rotation_matrix": None,
        "needs_review": True,
        "failure_reason": None,
        "artifacts": {
            "points_xyz": None,
            "points_obj": None,
            "mask_png": None,
            "valid_points_png": None,
            "point_distance_png": None,
        },
    }


def failed_record(record: dict[str, Any], reason: str) -> dict[str, Any]:
    record.update(
        box_type="failed",
        center_xyz=None,
        extent_xyz=None,
        rotation_matrix=None,
        needs_review=True,
        failure_reason=reason,
    )
    return record


def fit_box(points: np.ndarray, *, box_mode: str) -> FitResult:
    if box_mode not in {"auto", "aabb", "obb"}:
        raise ValueError(f"Unsupported box mode: {box_mode}")
    if box_mode == "aabb":
        return fit_aabb(points, needs_review=False, failure_reason=None)
    obb = fit_obb(points)
    if obb is not None:
        return obb
    if box_mode == "obb":
        return FitResult(
            box_type="failed",
            center_xyz=None,
            extent_xyz=None,
            rotation_matrix=None,
            needs_review=True,
            failure_reason="degenerate_covariance",
        )
    return fit_aabb(points, needs_review=True, failure_reason="degenerate_covariance")


def fit_aabb(points: np.ndarray, *, needs_review: bool, failure_reason: str | None) -> FitResult:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) / 2.0
    extent = maximum - minimum
    return FitResult(
        box_type="aabb",
        center_xyz=float_list(center),
        extent_xyz=float_list(extent),
        rotation_matrix=identity_matrix(),
        needs_review=needs_review,
        failure_reason=failure_reason,
    )


def fit_obb(points: np.ndarray) -> FitResult | None:
    centered = points - points.mean(axis=0)
    covariance = np.cov(centered, rowvar=False)
    if covariance.shape != (3, 3) or not np.isfinite(covariance).all():
        return None
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    rotation = eigenvectors[:, order]
    if eigenvalues[0] <= 1e-10 or eigenvalues[1] <= 1e-10:
        return None
    if np.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1.0
    local = centered @ rotation
    local_min = local.min(axis=0)
    local_max = local.max(axis=0)
    local_center = (local_min + local_max) / 2.0
    extent = local_max - local_min
    center = points.mean(axis=0) + local_center @ rotation.T
    return FitResult(
        box_type="obb",
        center_xyz=float_list(center),
        extent_xyz=float_list(extent),
        rotation_matrix=matrix_list(rotation),
        needs_review=False,
        failure_reason=None,
    )


def sample_points_for_mask(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    finite = np.isfinite(points).all(axis=2)
    selected = mask.astype(bool) & finite
    return points[selected].astype(np.float32)


def load_detection_mask(
    detection: dict[str, Any],
    object_dir: Path | None,
    image_size: tuple[int, int],
) -> MaskInfo | None:
    if object_dir is not None:
        for mask_path in (object_dir / "full_mask.png", object_dir / "artifacts" / "segmentation" / "full_mask.png"):
            if mask_path.is_file():
                return MaskInfo(mask=load_mask_image(mask_path, image_size), path=mask_path, source="full_mask")
    polygon = detection.get("mask_polygon")
    if polygon:
        return MaskInfo(mask=rasterize_polygon(polygon, image_size), path=None, source="mask_polygon")
    return None


def load_mask_image(mask_path: Path, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    image = Image.open(mask_path).convert("L")
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.uint8) > 0


def rasterize_polygon(polygon: list[list[float]], image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    image = Image.new("L", (width, height), 0)
    points = [(float(x), float(y)) for x, y in polygon]
    ImageDraw.Draw(image).polygon(points, fill=255)
    return np.asarray(image, dtype=np.uint8) > 0


def index_object_dirs(objects_dir: Path) -> dict[int, Path]:
    indexed: dict[int, Path] = {}
    if not objects_dir.is_dir():
        return indexed
    for child in sorted(objects_dir.iterdir()):
        if not child.is_dir():
            continue
        metadata_path = child / "metadata.json"
        if metadata_path.is_file():
            metadata = load_json(metadata_path)
            if "id" in metadata:
                indexed[int(metadata["id"])] = child
                continue
        match = re.match(r"^(\d+)_", child.name)
        if match:
            indexed[int(match.group(1))] = child
    return indexed


def write_points_xyz(points: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# x y z\n")
        for point in points:
            handle.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")


def write_region_surface_obj(points: np.ndarray, mask: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    finite = np.isfinite(points).all(axis=2)
    selected = mask.astype(bool) & finite
    vertex_indices = np.full(selected.shape, -1, dtype=np.int32)
    ys, xs = np.nonzero(selected)
    vertex_indices[ys, xs] = np.arange(1, len(ys) + 1, dtype=np.int32)

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# SceneForge per-SAM-region VGGT visible surface mesh\n")
        for y, x in zip(ys, xs):
            point = points[y, x]
            handle.write(f"v {point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
        for y in range(selected.shape[0] - 1):
            for x in range(selected.shape[1] - 1):
                top_left = vertex_indices[y, x]
                top_right = vertex_indices[y, x + 1]
                bottom_left = vertex_indices[y + 1, x]
                bottom_right = vertex_indices[y + 1, x + 1]
                if min(top_left, top_right, bottom_left, bottom_right) > 0:
                    handle.write(f"f {top_left} {top_right} {bottom_right} {bottom_left}\n")


def write_mask_png(mask: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(output_path)


def write_valid_points_png(points: np.ndarray, mask: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    finite = np.isfinite(points).all(axis=2)
    selected = mask.astype(bool) & finite
    preview = np.zeros((*mask.shape, 3), dtype=np.uint8)
    preview[mask.astype(bool)] = (64, 64, 64)
    preview[selected] = (64, 220, 120)
    Image.fromarray(preview, mode="RGB").save(output_path)


def write_point_distance_png(points: np.ndarray, mask: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    finite = np.isfinite(points).all(axis=2)
    selected = mask.astype(bool) & finite
    preview = np.zeros(mask.shape, dtype=np.uint8)
    if selected.any():
        distances = np.linalg.norm(points[selected], axis=1)
        minimum = float(distances.min())
        maximum = float(distances.max())
        if maximum > minimum:
            normalized = (distances - minimum) / (maximum - minimum)
        else:
            normalized = np.ones_like(distances)
        preview[selected] = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(preview, mode="L").save(output_path)


def write_regions_overlay(
    records: list[dict[str, Any]],
    detections: dict[str, Any],
    *,
    image_size: tuple[int, int],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base = load_overlay_base(detections, image_size)
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    colors = [
        (255, 72, 72, 110),
        (72, 172, 255, 110),
        (90, 220, 120, 110),
        (255, 205, 70, 110),
        (210, 110, 255, 110),
        (255, 140, 70, 110),
        (70, 230, 230, 110),
        (220, 220, 220, 110),
    ]
    for index, record in enumerate(records):
        mask_path = record.get("artifacts", {}).get("mask_png")
        color = colors[index % len(colors)]
        if mask_path:
            mask_image = Image.open(mask_path).convert("L")
            if mask_image.size != base.size:
                mask_image = mask_image.resize(base.size, Image.Resampling.NEAREST)
            color_layer = Image.new("RGBA", base.size, color)
            overlay.alpha_composite(Image.composite(color_layer, Image.new("RGBA", base.size, (0, 0, 0, 0)), mask_image))
        bbox = record.get("bbox_xyxy")
        if bbox and len(bbox) == 4:
            x0, y0, x1, y1 = [float(value) for value in bbox]
            draw.rectangle((x0, y0, x1, y1), outline=color[:3] + (255,), width=3)
    Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB").save(output_path)


def load_overlay_base(detections: dict[str, Any], image_size: tuple[int, int]) -> Image.Image:
    image_path = detections.get("image_path")
    if image_path and Path(image_path).is_file():
        image = Image.open(image_path).convert("RGB")
        if image.size != image_size:
            image = image.resize(image_size, Image.Resampling.BILINEAR)
        return image
    return Image.new("RGB", image_size, (20, 20, 20))


def write_boxes_obj(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    faces = (
        (0, 1, 3, 2),
        (4, 6, 7, 5),
        (0, 4, 5, 1),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 5, 7, 3),
    )
    vertex_offset = 1
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# SceneForge VGGT fitted box face meshes\n")
        for record in records:
            corners = box_corners(record)
            if corners is None:
                continue
            group_name = f"{int(record['detection_id']):02d}_{slugify(str(record.get('detector_label') or 'object'))}"
            handle.write(f"o {group_name}_{record['box_type']}\n")
            for corner in corners:
                handle.write(f"v {corner[0]:.6f} {corner[1]:.6f} {corner[2]:.6f}\n")
            for face in faces:
                handle.write("f " + " ".join(str(vertex_offset + index) for index in face) + "\n")
            vertex_offset += 8


def box_corners(record: dict[str, Any]) -> np.ndarray | None:
    if record.get("box_type") == "failed":
        return None
    center = record.get("center_xyz")
    extent = record.get("extent_xyz")
    rotation = record.get("rotation_matrix")
    if center is None or extent is None or rotation is None:
        return None
    center_array = np.asarray(center, dtype=np.float32)
    half_extent = np.asarray(extent, dtype=np.float32) / 2.0
    rotation_array = np.asarray(rotation, dtype=np.float32)
    if center_array.shape != (3,) or half_extent.shape != (3,) or rotation_array.shape != (3, 3):
        return None
    local_corners = np.array(
        [
            [-half_extent[0], -half_extent[1], -half_extent[2]],
            [half_extent[0], -half_extent[1], -half_extent[2]],
            [-half_extent[0], half_extent[1], -half_extent[2]],
            [half_extent[0], half_extent[1], -half_extent[2]],
            [-half_extent[0], -half_extent[1], half_extent[2]],
            [half_extent[0], -half_extent[1], half_extent[2]],
            [-half_extent[0], half_extent[1], half_extent[2]],
            [half_extent[0], half_extent[1], half_extent[2]],
        ],
        dtype=np.float32,
    )
    return center_array + local_corners @ rotation_array.T


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "object"


def float_list(values: np.ndarray) -> list[float]:
    return [float(value) for value in values.tolist()]


def matrix_list(values: np.ndarray) -> list[list[float]]:
    return [[float(item) for item in row] for row in values.tolist()]


def identity_matrix() -> list[list[float]]:
    return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
