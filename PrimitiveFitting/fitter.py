from __future__ import annotations

from dataclasses import replace
from math import isfinite

import numpy as np

from PrimitiveFitting.camera import PinholeCamera
from PrimitiveFitting.types import PrimitiveFit
from ShapeDetection.report import ObjectShapeDetection


IDENTITY_MATRIX = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
MIN_DIMENSION = 0.02
GEOMETRIC_LABELS = {"sphere", "cylinder", "cone", "box", "plane", "unknown"}
DEPTH_OVERRIDE_CONFIDENCE_LIMIT = 0.9
CONE_MIN_HEIGHT_TO_DIAMETER_RATIO = 0.75
PLANE_THICKNESS = 0.03
MIN_PLANE_EXTENT = 0.08


def fit_primitive(
    detection: ObjectShapeDetection,
    points: np.ndarray,
    camera: PinholeCamera | None = None,
) -> PrimitiveFit:
    label = geometric_label(detection.primitive_label)
    source = str(detection.primitive_label_source or "detector")
    if len(points) < 3 or not np.isfinite(points).all():
        return fallback_fit(detection, label)

    candidates = fit_primitive_candidates(detection, points, camera=camera, label=label)
    scored = [(candidate_score(item, points), item) for item in candidates]
    scored.sort(key=lambda item: (item[0], item[1].fit_quality.get("mode", "")))
    selected_score, selected = select_candidate(scored)

    label_source = source
    label_warning = None
    if label == "box":
        cylinder_candidate = fit_axis_primitive(detection, points, "cylinder")
        cylinder_score = candidate_score(cylinder_candidate, points)
        cylinder_like = likely_cylinder_from_depth(detection, points)
        best_box_score = selected_score
        if (
            detection.detector_confidence < DEPTH_OVERRIDE_CONFIDENCE_LIMIT
            and cylinder_like
            and cylinder_score < best_box_score * 0.85
            and source != "fused"
        ):
            selected = cylinder_candidate
            selected_score = cylinder_score
            label_source = "depth_override"
        elif cylinder_like:
            label_warning = "box_may_be_cylinder"

    quality = dict(selected.fit_quality)
    quality["selected_fit_mode"] = quality.get("mode", "unknown")
    quality["candidate_score"] = round(float(selected_score), 6)
    quality["candidate_scores"] = {
        item.fit_quality.get("mode", "unknown"): round(float(score), 6)
        for score, item in scored
    }
    if label_warning is not None:
        quality["label_warning"] = label_warning

    return replace(
        selected,
        fit_quality=quality,
        primitive_label_source=label_source,
    )


def fit_primitive_candidates(
    detection: ObjectShapeDetection,
    points: np.ndarray,
    camera: PinholeCamera | None = None,
    label: str | None = None,
) -> list[PrimitiveFit]:
    output_label = geometric_label(label or detection.primitive_label)
    candidates: list[PrimitiveFit] = []
    if camera is not None:
        candidates.append(fit_camera_facing(detection, points, camera, output_label))

    if output_label == "sphere":
        if camera is not None:
            candidates.append(fit_sphere_depth(detection, points, camera, output_label))
        else:
            candidates.append(fit_sphere(detection, points, output_label))
    elif output_label in {"cylinder", "cone"}:
        candidates.append(fit_axis_primitive(detection, points, output_label))
    elif output_label == "box":
        candidates.append(fit_box(detection, points, output_label))
    elif output_label == "plane":
        candidates.append(fit_plane(detection, points, output_label))
    else:
        candidates.append(fallback_fit(detection, label))
    return candidates


def candidate_score(fit: PrimitiveFit, points: np.ndarray) -> float:
    mode = str(fit.fit_quality.get("mode", "fallback"))
    residual = fit.fit_quality.get("residual")
    if residual is None:
        residual_value = 1.0
    else:
        residual_value = abs(float(residual))
    scale = max(float(np.linalg.norm(np.asarray(fit.dimensions_xyz))), MIN_DIMENSION)
    score = residual_value / scale
    if mode == "camera_silhouette":
        score += 0.04
    if fit.primitive_label == "cone" and mode == "axis_depth":
        score += 0.25
    if fit.primitive_label == "box" and mode == "depth_pca":
        score += box_surface_residual(points, fit)
    return float(score)


def select_candidate(scored: list[tuple[float, PrimitiveFit]]) -> tuple[float, PrimitiveFit]:
    camera_items = [item for item in scored if item[1].fit_quality.get("mode") == "camera_silhouette"]
    if not camera_items:
        return scored[0]
    camera_score, camera_fit = camera_items[0]
    depth_items = [item for item in scored if item[1].fit_quality.get("mode") != "camera_silhouette"]
    if not depth_items:
        return camera_score, camera_fit

    best_depth_score, best_depth_fit = min(depth_items, key=lambda item: item[0])
    if camera_fit.primitive_label == "plane" and best_depth_fit.primitive_label == "plane":
        if is_valid_plane_depth_fit(best_depth_fit):
            return best_depth_score, best_depth_fit
        quality = dict(camera_fit.fit_quality)
        quality["plane_fallback_reason"] = "invalid_depth_pca"
        return camera_score, replace(camera_fit, fit_quality=quality)
    if camera_fit.primitive_label == "cone" and best_depth_fit.primitive_label == "cone":
        if not is_shape_extent_safe(camera_fit, best_depth_fit):
            return camera_score, camera_fit
        if best_depth_score >= camera_score * 0.20:
            return camera_score, restore_axis_rotation(camera_fit, best_depth_fit)
        return best_depth_score, restore_cone_silhouette_extent(best_depth_fit, camera_fit)
    if (
        camera_fit.primitive_label == "cylinder"
        and best_depth_fit.primitive_label == "cylinder"
        and is_projection_safe(camera_fit, best_depth_fit)
        and is_shape_extent_safe(camera_fit, best_depth_fit)
    ):
        return camera_score, restore_axis_rotation(camera_fit, best_depth_fit)
    if (
        best_depth_score < camera_score * 0.75
        and is_projection_safe(camera_fit, best_depth_fit)
        and is_shape_extent_safe(camera_fit, best_depth_fit)
    ):
        return best_depth_score, best_depth_fit
    return camera_score, camera_fit


def restore_axis_rotation(silhouette_fit: PrimitiveFit, depth_fit: PrimitiveFit) -> PrimitiveFit:
    quality = dict(silhouette_fit.fit_quality)
    quality["rotation_source"] = "depth_axis_silhouette_extent"
    quality["raw_silhouette_rotation_matrix"] = [
        [round(float(value), 6) for value in row]
        for row in silhouette_fit.rotation_matrix
    ]
    quality["depth_rotation_matrix"] = [
        [round(float(value), 6) for value in row]
        for row in depth_fit.rotation_matrix
    ]
    return replace(
        silhouette_fit,
        rotation_matrix=depth_fit.rotation_matrix,
        fit_quality=quality,
    )


def restore_cone_silhouette_extent(depth_fit: PrimitiveFit, camera_fit: PrimitiveFit) -> PrimitiveFit:
    depth_dimensions = tuple(float(value) for value in depth_fit.dimensions_xyz)
    camera_dimensions = tuple(float(value) for value in camera_fit.dimensions_xyz)
    restored_height = max(depth_dimensions[2], camera_dimensions[2] * 0.85)
    restored_diameter = max(depth_dimensions[0], depth_dimensions[1], min(camera_dimensions[0], camera_dimensions[1]) * 0.90)
    quality = dict(depth_fit.fit_quality)
    quality["extent_source"] = "depth_silhouette_blended_axis_length"
    quality["raw_depth_dimensions_xyz"] = [round(float(value), 6) for value in depth_dimensions]
    quality["silhouette_dimensions_xyz"] = [round(float(value), 6) for value in camera_dimensions]
    quality["raw_depth_rotation_matrix"] = [
        [round(float(value), 6) for value in row]
        for row in depth_fit.rotation_matrix
    ]
    rotation = blended_cone_rotation(depth_fit, camera_fit)
    restored_dimensions = enforce_cone_aspect_ratio((restored_diameter, restored_diameter, restored_height))
    return replace(
        depth_fit,
        rotation_matrix=rotation,
        dimensions_xyz=restored_dimensions,
        fit_quality=quality,
    )


def blended_cone_rotation(depth_fit: PrimitiveFit, camera_fit: PrimitiveFit) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    depth_rotation = np.asarray(depth_fit.rotation_matrix, dtype=np.float64)
    camera_rotation = np.asarray(camera_fit.rotation_matrix, dtype=np.float64)
    axis = depth_rotation[:, 2] + camera_rotation[:, 2]
    if np.linalg.norm(axis) < 1e-6:
        axis = depth_rotation[:, 2]
    secondary = depth_rotation[:, 0] + camera_rotation[:, 0]
    if np.linalg.norm(secondary) < 1e-6:
        secondary = depth_rotation[:, 0]
    return matrix_to_tuple(axis_aligned_basis(axis, secondary))


def is_shape_extent_safe(camera_fit: PrimitiveFit, depth_fit: PrimitiveFit) -> bool:
    if camera_fit.primitive_label not in {"cylinder", "cone"}:
        return True
    camera_height = max(float(camera_fit.dimensions_xyz[2]), MIN_DIMENSION)
    depth_height = max(float(depth_fit.dimensions_xyz[2]), MIN_DIMENSION)
    height_ratio = depth_height / camera_height
    return 0.65 <= height_ratio <= 1.35


def is_projection_safe(camera_fit: PrimitiveFit, depth_fit: PrimitiveFit) -> bool:
    camera_dimensions = np.asarray(camera_fit.dimensions_xyz, dtype=np.float64)
    depth_dimensions = np.asarray(depth_fit.dimensions_xyz, dtype=np.float64)
    depth_rotation = np.asarray(depth_fit.rotation_matrix, dtype=np.float64)
    camera_size = max(float(np.linalg.norm(camera_dimensions)), MIN_DIMENSION)
    depth_size = max(float(np.linalg.norm(depth_dimensions)), MIN_DIMENSION)
    if depth_size > camera_size * 1.35:
        return False

    camera_y_extent = max(float(camera_dimensions[1]), MIN_DIMENSION)
    depth_y_extent = float(np.abs(depth_rotation[1, :]) @ depth_dimensions)
    return depth_y_extent <= camera_y_extent * 2.5


def is_valid_plane_depth_fit(fit: PrimitiveFit) -> bool:
    if fit.fit_quality.get("status") != "ok":
        return False
    if fit.fit_quality.get("mode") != "depth_pca":
        return False
    dimensions = tuple(float(value) for value in fit.dimensions_xyz)
    residual = fit.fit_quality.get("residual")
    if residual is None or not isfinite(float(residual)):
        return False
    return min(dimensions[0], dimensions[1]) >= MIN_PLANE_EXTENT and dimensions[2] <= max(PLANE_THICKNESS, MIN_DIMENSION) * 1.5


def fit_camera_facing(
    detection: ObjectShapeDetection,
    points: np.ndarray,
    camera: PinholeCamera,
    label: str,
) -> PrimitiveFit:
    median_depth = float(np.median(points[:, 1]))
    left, top, right, bottom = detection.bbox_xyxy
    bbox_pixels = np.array(
        [
            [left, top],
            [right, top],
            [right, bottom],
            [left, bottom],
            [(left + right) / 2.0, (top + bottom) / 2.0],
        ],
        dtype=np.float64,
    )
    depth_span = max(
        MIN_DIMENSION,
        float(np.percentile(points[:, 1], 85) - np.percentile(points[:, 1], 15)),
    )

    fitted_depth = median_depth
    center, width, height = bbox_geometry_at_depth(camera, bbox_pixels, fitted_depth)
    shift_dimensions = silhouette_dimensions(label, width, height, depth_span, shrink_box=False)
    dimensions = silhouette_dimensions(label, width, height, depth_span)
    fitted_depth = median_depth + visible_front_shift_amount(label, shift_dimensions)
    center, width, height = bbox_geometry_at_depth(camera, bbox_pixels, fitted_depth)
    if label == "sphere":
        dimensions = silhouette_dimensions(label, width, height, depth_span)
    elif label in {"cylinder", "cone"}:
        dimensions = silhouette_dimensions(label, width, height, depth_span)

    rotation = IDENTITY_MATRIX
    if label in {"cylinder", "cone"}:
        rotation = mask_axis_rotation(detection, camera, fitted_depth, label)

    residual = float(np.std(points[:, 1] - median_depth))
    return make_fit(
        detection=detection,
        label=label,
        center=center,
        rotation=rotation,
        dimensions=dimensions,
        points=points,
        residual=residual,
        mode="camera_silhouette",
    )


def fit_sphere(detection: ObjectShapeDetection, points: np.ndarray, label: str) -> PrimitiveFit:
    center = np.median(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    radius = max(MIN_DIMENSION, float(np.percentile(distances, 85)))
    return make_fit(
        detection=detection,
        label=label,
        center=center,
        rotation=IDENTITY_MATRIX,
        dimensions=(radius * 2.0, radius * 2.0, radius * 2.0),
        points=points,
        residual=float(np.std(distances - radius)),
        mode="sphere_depth",
    )


def fit_sphere_depth(
    detection: ObjectShapeDetection,
    points: np.ndarray,
    camera: PinholeCamera,
    label: str,
) -> PrimitiveFit:
    center = np.median(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    depth_radius = max(MIN_DIMENSION, float(np.percentile(distances, 85)))

    silhouette = fit_camera_facing(detection, points, camera, label)
    silhouette_diameter = max(MIN_DIMENSION, min(silhouette.dimensions_xyz[0], silhouette.dimensions_xyz[2]))
    diameter = max(MIN_DIMENSION, (silhouette_diameter + depth_radius * 2.0) * 0.5)
    radius = diameter * 0.5
    center = np.asarray(silhouette.center_xyz, dtype=np.float64)
    return make_fit(
        detection=detection,
        label=label,
        center=center,
        rotation=IDENTITY_MATRIX,
        dimensions=(diameter, diameter, diameter),
        points=points,
        residual=float(np.std(distances - radius)),
        mode="sphere_depth",
    )


def fit_box(detection: ObjectShapeDetection, points: np.ndarray, label: str) -> PrimitiveFit:
    center, axes, dimensions = oriented_bounds(points)
    residual = box_surface_residual_from_components(points, center, axes, dimensions)
    return make_fit(
        detection=detection,
        label=label,
        center=center,
        rotation=matrix_to_tuple(axes),
        dimensions=tuple(float(max(MIN_DIMENSION, value)) for value in dimensions),
        points=points,
        residual=residual,
        mode="depth_pca",
    )


def mask_axis_rotation(
    detection: ObjectShapeDetection,
    camera: PinholeCamera,
    depth: float,
    label: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    major_2d = mask_major_axis_2d(detection, label)
    if major_2d is None:
        return IDENTITY_MATRIX
    center_pixel = np.array(
        [
            [(detection.bbox_xyxy[0] + detection.bbox_xyxy[2]) * 0.5, (detection.bbox_xyxy[1] + detection.bbox_xyxy[3]) * 0.5],
            [
                (detection.bbox_xyxy[0] + detection.bbox_xyxy[2]) * 0.5 + major_2d[0],
                (detection.bbox_xyxy[1] + detection.bbox_xyxy[3]) * 0.5 + major_2d[1],
            ],
        ],
        dtype=np.float64,
    )
    points = camera.unproject_scene_depth_pixels(
        center_pixel,
        np.full(2, depth, dtype=np.float64),
    )
    axis = points[1] - points[0]
    if np.linalg.norm(axis) < 1e-6:
        return IDENTITY_MATRIX
    return matrix_to_tuple(axis_aligned_basis(axis, np.array((1.0, 0.0, 0.0))))


def mask_major_axis_2d(detection: ObjectShapeDetection, label: str) -> np.ndarray | None:
    polygon = np.asarray(detection.mask_polygon, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[0] < 3:
        return None
    centered = polygon - np.mean(polygon, axis=0)
    covariance = np.cov(centered, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    major_2d = vectors[:, int(np.argmax(values))]
    if label == "cone":
        projections = centered @ major_2d
        perpendicular_2d = np.array((-major_2d[1], major_2d[0]), dtype=np.float64)
        low_cutoff = np.percentile(projections, 18)
        high_cutoff = np.percentile(projections, 82)
        low_mask = projections <= low_cutoff
        high_mask = projections >= high_cutoff
        low_points = polygon[low_mask]
        high_points = polygon[high_mask]

        def endpoint_width(points_2d: np.ndarray) -> float:
            if len(points_2d) < 2:
                return float("inf")
            perpendicular_offsets = (points_2d - np.mean(polygon, axis=0)) @ perpendicular_2d
            return float(np.percentile(perpendicular_offsets, 85) - np.percentile(perpendicular_offsets, 15))

        low_width = endpoint_width(low_points)
        high_width = endpoint_width(high_points)
        if high_width < low_width:
            tip_point = np.mean(high_points, axis=0)
            base_point = np.mean(low_points, axis=0)
        else:
            tip_point = np.mean(low_points, axis=0)
            base_point = np.mean(high_points, axis=0)
        major_2d = tip_point - base_point
    elif major_2d[1] > 0.0:
        major_2d *= -1.0
    if np.linalg.norm(major_2d) < 1e-6:
        return None
    major_2d = major_2d / np.linalg.norm(major_2d)
    return major_2d


def bbox_geometry_at_depth(
    camera: PinholeCamera,
    bbox_pixels: np.ndarray,
    depth: float,
) -> tuple[np.ndarray, float, float]:
    bbox_points = camera.unproject_scene_depth_pixels(
        bbox_pixels,
        np.full(len(bbox_pixels), depth, dtype=np.float64),
    )
    center = bbox_points[4]
    width = max(MIN_DIMENSION, float(np.linalg.norm(bbox_points[1] - bbox_points[0])))
    height = max(MIN_DIMENSION, float(np.linalg.norm(bbox_points[3] - bbox_points[0])))
    return center, width, height


def silhouette_dimensions(
    label: str,
    width: float,
    height: float,
    depth_span: float,
    shrink_box: bool = True,
) -> tuple[float, float, float]:
    if label == "sphere":
        diameter = max(MIN_DIMENSION, min(width, height))
        return (diameter, diameter, diameter)
    if label == "cylinder":
        diameter = max(MIN_DIMENSION, width)
        return (diameter, diameter, height)
    if label == "cone":
        diameter = max(MIN_DIMENSION, width)
        return (diameter, diameter, height)
    if label == "plane":
        return (width, max(MIN_DIMENSION, min(depth_span, 0.04)), height)
    if label == "box":
        side_depth = max(MIN_DIMENSION, min(width, height))
        if shrink_box:
            return (width * 0.80, side_depth * 0.91, height * 0.91)
        return (width, side_depth, height)
    thickness = max(MIN_DIMENSION, min(depth_span, max(width, height) * 0.45))
    return (width, thickness, height)


def shift_center_from_visible_front(center: np.ndarray, label: str, dimensions: tuple[float, float, float]) -> np.ndarray:
    shifted = np.asarray(center, dtype=np.float64).copy()
    shifted[1] += visible_front_shift_amount(label, dimensions)
    return shifted


def visible_front_shift_amount(label: str, dimensions: tuple[float, float, float]) -> float:
    if label in {"sphere", "cylinder"}:
        return 0.42 * max(float(dimensions[0]), float(dimensions[1]))
    if label == "cone":
        return 0.24 * float(dimensions[2])
    if label == "box":
        return 0.34 * min(float(dimensions[0]), float(dimensions[1]), float(dimensions[2]))
    return 0.0


def fit_axis_primitive(
    detection: ObjectShapeDetection,
    points: np.ndarray,
    label: str,
) -> PrimitiveFit:
    center = np.median(points, axis=0)
    axes = pca_axes(points)
    axis = axes[:, 0]
    centered = points - center
    along = centered @ axis
    radial_vectors = centered - np.outer(along, axis)
    radial_distances = np.linalg.norm(radial_vectors, axis=1)
    if label == "cone":
        low_mask = along <= np.percentile(along, 18)
        high_mask = along >= np.percentile(along, 82)

        def end_radius(mask: np.ndarray) -> float:
            if not np.any(mask):
                return 0.0
            return float(np.percentile(radial_distances[mask], 85))

        low_radius = end_radius(low_mask)
        high_radius = end_radius(high_mask)
        if high_radius > low_radius:
            axis *= -1.0
    rotation_axes = axis_aligned_basis(axis, axes[:, 1])
    along = centered @ axis
    height = max(MIN_DIMENSION, float(np.percentile(along, 95) - np.percentile(along, 5)))
    radial_vectors = centered - np.outer(along, axis)
    radial_distances = np.linalg.norm(radial_vectors, axis=1)
    percentile = 88 if label == "cylinder" else 94
    radius = max(MIN_DIMENSION, float(np.percentile(radial_distances, percentile)))
    return make_fit(
        detection=detection,
        label=label,
        center=center,
        rotation=matrix_to_tuple(rotation_axes),
        dimensions=(radius * 2.0, radius * 2.0, height),
        points=points,
        residual=float(np.std(radial_distances - np.median(radial_distances))),
        mode="axis_depth",
    )


def fit_plane(detection: ObjectShapeDetection, points: np.ndarray, label: str) -> PrimitiveFit:
    center = np.median(points, axis=0)
    plane_axes = stable_plane_axes(points, center)
    projected = (points - center) @ plane_axes
    ranges = np.percentile(projected, 95, axis=0) - np.percentile(projected, 5, axis=0)
    dimensions = (
        max(MIN_DIMENSION, float(ranges[0])),
        max(MIN_DIMENSION, float(ranges[1])),
        max(MIN_DIMENSION, float(min(PLANE_THICKNESS, max(ranges[2], MIN_DIMENSION)))),
    )
    residual = float(np.std(projected[:, 2]))
    fit = make_fit(
        detection=detection,
        label=label,
        center=center,
        rotation=matrix_to_tuple(plane_axes),
        dimensions=dimensions,
        points=points,
        residual=residual,
        mode="depth_pca",
    )
    quality = dict(fit.fit_quality)
    normal = plane_axes[:, 2]
    quality["plane_normal_xyz"] = [round(float(value), 6) for value in normal]
    quality["plane_extent_source"] = "visible_depth_pca_patch"
    quality["plane_thickness"] = round(float(dimensions[2]), 6)
    return replace(fit, fit_quality=quality)


def stable_plane_axes(points: np.ndarray, center: np.ndarray) -> np.ndarray:
    axes = pca_axes(points)
    x_axis = axes[:, 0]
    normal = axes[:, 2]
    if float(normal @ center) > 0.0:
        normal *= -1.0
    x_axis = x_axis - normal * float(x_axis @ normal)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = axes[:, 1] - normal * float(axes[:, 1] @ normal)
    x_axis = x_axis / max(np.linalg.norm(x_axis), 1e-9)
    dominant = int(np.argmax(np.abs(x_axis)))
    if x_axis[dominant] < 0.0:
        x_axis *= -1.0
    y_axis = np.cross(normal, x_axis)
    y_axis = y_axis / max(np.linalg.norm(y_axis), 1e-9)
    normal = normal / max(np.linalg.norm(normal), 1e-9)
    return np.column_stack((x_axis, y_axis, normal))


def fallback_fit(detection: ObjectShapeDetection, label: str | None = None) -> PrimitiveFit:
    output_label = geometric_label(label or detection.primitive_label)
    left, top, right, bottom = detection.bbox_xyxy
    width = max(MIN_DIMENSION, (right - left) / 100.0)
    height = max(MIN_DIMENSION, (bottom - top) / 100.0)
    center = ((left + right) / 200.0, 3.0, -(top + bottom) / 200.0)
    return PrimitiveFit(
        id=detection.id,
        primitive_label=output_label,
        confidence=detection.primitive_confidence,
        center_xyz=tuple(float(value) for value in center),
        rotation_matrix=IDENTITY_MATRIX,
        dimensions_xyz=(width, width, height),
        fit_quality={
            "status": "fallback",
            "mode": "fallback",
            "selected_fit_mode": "fallback",
            "sample_count": 0,
            "residual": None,
        },
    )


def make_fit(
    detection: ObjectShapeDetection,
    label: str,
    center: np.ndarray,
    rotation: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
    dimensions: tuple[float, float, float],
    points: np.ndarray,
    residual: float,
    mode: str = "pca",
) -> PrimitiveFit:
    finite = all(isfinite(float(value)) for value in center) and all(
        isfinite(float(value)) for value in dimensions
    )
    status = "ok" if finite else "fallback"
    if status != "ok":
        return fallback_fit(detection, label)

    dimensions = enforce_cone_aspect_ratio(dimensions, label=label)
    return PrimitiveFit(
        id=detection.id,
        primitive_label=label,
        confidence=detection.primitive_confidence,
        center_xyz=tuple(float(value) for value in center),
        rotation_matrix=rotation,
        dimensions_xyz=tuple(float(max(MIN_DIMENSION, value)) for value in dimensions),
        fit_quality={
            "status": status,
            "mode": mode,
            "sample_count": int(len(points)),
            "depth_min": round(float(points[:, 1].min()), 6),
            "depth_max": round(float(points[:, 1].max()), 6),
            "residual": round(float(residual), 6),
        },
    )


def oriented_bounds(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axes = pca_axes(points)
    center = np.median(points, axis=0)
    projected = (points - center) @ axes
    mins = np.percentile(projected, 5, axis=0)
    maxs = np.percentile(projected, 95, axis=0)
    local_center = (mins + maxs) / 2.0
    dimensions = maxs - mins
    world_center = center + axes @ local_center
    return world_center, axes, dimensions


def enforce_cone_aspect_ratio(
    dimensions: tuple[float, float, float],
    label: str = "cone",
) -> tuple[float, float, float]:
    if label != "cone":
        return tuple(float(value) for value in dimensions)
    x, y, z = tuple(float(value) for value in dimensions)
    diameter = max(MIN_DIMENSION, x, y)
    height = max(MIN_DIMENSION, z, diameter * CONE_MIN_HEIGHT_TO_DIAMETER_RATIO)
    return diameter, diameter, height


def box_surface_residual(points: np.ndarray, fit: PrimitiveFit) -> float:
    center = np.asarray(fit.center_xyz, dtype=np.float64)
    axes = np.asarray(fit.rotation_matrix, dtype=np.float64)
    dimensions = np.asarray(fit.dimensions_xyz, dtype=np.float64)
    return box_surface_residual_from_components(points, center, axes, dimensions)


def box_surface_residual_from_components(
    points: np.ndarray,
    center: np.ndarray,
    axes: np.ndarray,
    dimensions: np.ndarray,
) -> float:
    local = (points - center) @ axes
    half = np.maximum(dimensions * 0.5, MIN_DIMENSION)
    face_distances = np.abs(np.abs(local) - half)
    nearest_face = np.min(face_distances, axis=1)
    return float(np.mean(nearest_face) / max(float(np.linalg.norm(dimensions)), MIN_DIMENSION))


def likely_cylinder_from_depth(detection: ObjectShapeDetection, points: np.ndarray) -> bool:
    left, top, right, bottom = detection.bbox_xyxy
    width = max(float(right - left), 1.0)
    height = max(float(bottom - top), 1.0)
    aspect = max(width, height) / max(min(width, height), 1.0)
    depth_span = float(np.percentile(points[:, 1], 90) - np.percentile(points[:, 1], 10))
    horizontal_span = float(np.percentile(points[:, 0], 90) - np.percentile(points[:, 0], 10))
    vertical_span = float(np.percentile(points[:, 2], 90) - np.percentile(points[:, 2], 10))
    visible_span = max(horizontal_span, vertical_span, MIN_DIMENSION)
    curvature_ratio = depth_span / visible_span
    return aspect < 2.7 and 0.08 <= curvature_ratio <= 0.75


def pca_axes(points: np.ndarray) -> np.ndarray:
    centered = points - np.mean(points, axis=0)
    covariance = np.cov(centered, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    axes = vectors[:, order]
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1.0
    return axes


def geometric_label(label: str) -> str:
    return label if label in GEOMETRIC_LABELS else "unknown"


def axis_aligned_basis(axis: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    z_axis = axis / max(np.linalg.norm(axis), 1e-9)
    x_axis = secondary - z_axis * float(secondary @ z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        fallback = np.array((1.0, 0.0, 0.0))
        if abs(float(fallback @ z_axis)) > 0.9:
            fallback = np.array((0.0, 1.0, 0.0))
        x_axis = fallback - z_axis * float(fallback @ z_axis)
    x_axis = x_axis / max(np.linalg.norm(x_axis), 1e-9)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / max(np.linalg.norm(y_axis), 1e-9)
    return np.column_stack((x_axis, y_axis, z_axis))


def matrix_to_tuple(matrix: np.ndarray) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    return tuple(tuple(float(matrix[row, col]) for col in range(3)) for row in range(3))
