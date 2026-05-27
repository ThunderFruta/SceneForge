from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from PrimitiveFitting.masks import polygon_to_mask
from PrimitiveFitting.types import PrimitiveFit
from ShapeDetection.report import ObjectShapeDetection


MIN_DIMENSION = 0.02
FOREGROUND_THRESHOLD = 0.01


def refine_fits_against_depth(
    fits: list[PrimitiveFit],
    detections: list[ObjectShapeDetection],
    source_depth_path: str | Path,
    fitted_depth_path: str | Path,
    near_depth: float,
    far_depth: float,
) -> tuple[list[PrimitiveFit], dict[str, Any]]:
    source = _load_depth_image(source_depth_path)
    fitted = _load_depth_image(fitted_depth_path)
    if source.shape != fitted.shape:
        raise ValueError(
            f"Fitted depth size {fitted.shape[1]}x{fitted.shape[0]} does not match "
            f"source depth size {source.shape[1]}x{source.shape[0]}."
        )

    detection_by_id = {item.id: item for item in detections}
    refined: list[PrimitiveFit] = []
    object_diagnostics: list[dict[str, Any]] = []
    changed_count = 0
    for fit in fits:
        detection = detection_by_id.get(fit.id)
        if detection is None:
            refined.append(fit)
            continue
        updated, diagnostics = refine_fit_against_depth(
            fit=fit,
            detection=detection,
            source=source,
            fitted=fitted,
            near_depth=near_depth,
            far_depth=far_depth,
        )
        refined.append(updated)
        object_diagnostics.append(diagnostics)
        if diagnostics["changed"]:
            changed_count += 1

    diagnostics = {
        "schema_version": 1,
        "strategy": "one_pass_depth_residual_refinement",
        "source_depth_path": str(source_depth_path),
        "fitted_depth_path": str(fitted_depth_path),
        "near_depth": round(float(near_depth), 6),
        "far_depth": round(float(far_depth), 6),
        "changed_object_count": changed_count,
        "objects": object_diagnostics,
    }
    return refined, diagnostics


def refine_fit_against_depth(
    fit: PrimitiveFit,
    detection: ObjectShapeDetection,
    source: np.ndarray,
    fitted: np.ndarray,
    near_depth: float,
    far_depth: float,
) -> tuple[PrimitiveFit, dict[str, Any]]:
    height, width = source.shape
    mask = polygon_to_mask(detection.mask_polygon, width, height)
    region = padded_bbox_region(detection.bbox_xyxy, width, height)

    source_present = source > FOREGROUND_THRESHOLD
    fitted_present = fitted > FOREGROUND_THRESHOLD
    visible_overlap = mask & source_present & fitted_present
    coverage_region = region & (source_present | fitted_present)
    overlap_count = int(np.count_nonzero(visible_overlap))
    source_region_count = max(1, int(np.count_nonzero(region & source_present)))
    fitted_region_count = max(1, int(np.count_nonzero(region & fitted_present)))
    union_count = max(1, int(np.count_nonzero(coverage_region)))
    overlap_ratio = float(overlap_count) / union_count

    signed_normalized_error = 0.0
    if np.any(visible_overlap):
        signed_normalized_error = float(np.mean(fitted[visible_overlap] - source[visible_overlap]))

    missing_ratio = float(np.count_nonzero(region & source_present & ~fitted_present)) / source_region_count
    extra_ratio = float(np.count_nonzero(region & fitted_present & ~source_present)) / fitted_region_count
    mismatch_ratio = float(np.count_nonzero(region & (source_present ^ fitted_present))) / union_count

    evidence_reason = depth_evidence_reason(
        overlap_count=overlap_count,
        source_region_count=source_region_count,
        fitted_region_count=fitted_region_count,
        overlap_ratio=overlap_ratio,
    )
    if evidence_reason == "ok":
        size_scale = bounded_size_scale(missing_ratio=missing_ratio, extra_ratio=extra_ratio)
        y_shift = bounded_depth_shift(
            signed_normalized_error=signed_normalized_error,
            fit=fit,
            near_depth=near_depth,
            far_depth=far_depth,
        )
    else:
        size_scale = 1.0
        y_shift = 0.0

    old_center = tuple(float(value) for value in fit.center_xyz)
    old_dimensions = tuple(float(value) for value in fit.dimensions_xyz)
    new_center = (old_center[0], old_center[1] + y_shift, old_center[2])
    new_dimensions = scaled_dimensions(fit.primitive_label, old_dimensions, size_scale)

    changed = (
        abs(y_shift) > 1e-6
        or any(abs(new - old) > 1e-6 for new, old in zip(new_dimensions, old_dimensions))
    )
    quality = dict(fit.fit_quality)
    quality["depth_refinement"] = {
        "strategy": "one_pass_depth_residual_refinement",
        "changed": changed,
        "signed_normalized_error": round(float(signed_normalized_error), 6),
        "missing_ratio": round(float(missing_ratio), 6),
        "extra_ratio": round(float(extra_ratio), 6),
        "mismatch_ratio": round(float(mismatch_ratio), 6),
        "overlap_ratio": round(float(overlap_ratio), 6),
        "evidence_reason": evidence_reason,
        "size_scale": round(float(size_scale), 6),
        "center_y_shift": round(float(y_shift), 6),
    }

    diagnostics = {
        "id": fit.id,
        "primitive_label": fit.primitive_label,
        "changed": changed,
        "signed_normalized_error": round(float(signed_normalized_error), 6),
        "missing_ratio": round(float(missing_ratio), 6),
        "extra_ratio": round(float(extra_ratio), 6),
        "mismatch_ratio": round(float(mismatch_ratio), 6),
        "overlap_ratio": round(float(overlap_ratio), 6),
        "evidence_reason": evidence_reason,
        "size_scale": round(float(size_scale), 6),
        "center_y_shift": round(float(y_shift), 6),
        "old_center_xyz": [round(value, 6) for value in old_center],
        "new_center_xyz": [round(value, 6) for value in new_center],
        "old_dimensions_xyz": [round(value, 6) for value in old_dimensions],
        "new_dimensions_xyz": [round(value, 6) for value in new_dimensions],
    }
    return replace(
        fit,
        center_xyz=new_center,
        dimensions_xyz=new_dimensions,
        fit_quality=quality,
    ), diagnostics


def padded_bbox_region(
    bbox_xyxy: tuple[float, float, float, float],
    width: int,
    height: int,
) -> np.ndarray:
    left, top, right, bottom = bbox_xyxy
    bbox_width = max(1.0, float(right - left))
    bbox_height = max(1.0, float(bottom - top))
    pad_x = max(2, int(round(bbox_width * 0.12)))
    pad_y = max(2, int(round(bbox_height * 0.12)))
    x0 = max(0, int(np.floor(left)) - pad_x)
    y0 = max(0, int(np.floor(top)) - pad_y)
    x1 = min(width, int(np.ceil(right)) + pad_x)
    y1 = min(height, int(np.ceil(bottom)) + pad_y)
    region = np.zeros((height, width), dtype=bool)
    region[y0:y1, x0:x1] = True
    return region


def depth_evidence_reason(
    overlap_count: int,
    source_region_count: int,
    fitted_region_count: int,
    overlap_ratio: float,
) -> str:
    if source_region_count < 64:
        return "too_few_source_pixels"
    if fitted_region_count < 64:
        return "too_few_fitted_pixels"
    if overlap_count < 32:
        return "too_few_overlap_pixels"
    if overlap_ratio < 0.10:
        return "low_overlap_ratio"
    return "ok"


def bounded_size_scale(missing_ratio: float, extra_ratio: float) -> float:
    coverage_balance = float(missing_ratio - extra_ratio)
    if abs(coverage_balance) < 0.015:
        return 1.0
    return float(np.clip(1.0 + coverage_balance * 0.35, 0.82, 1.16))


def bounded_depth_shift(
    signed_normalized_error: float,
    fit: PrimitiveFit,
    near_depth: float,
    far_depth: float,
) -> float:
    if abs(signed_normalized_error) < 0.006:
        return 0.0
    scene_span = max(MIN_DIMENSION, float(far_depth - near_depth))
    dimensions = tuple(max(MIN_DIMENSION, float(value)) for value in fit.dimensions_xyz)
    max_shift = min(0.45, max(dimensions) * 0.28)
    return float(np.clip(signed_normalized_error * scene_span * 0.55, -max_shift, max_shift))


def scaled_dimensions(
    label: str,
    dimensions: tuple[float, float, float],
    size_scale: float,
) -> tuple[float, float, float]:
    x, y, z = (max(MIN_DIMENSION, float(value)) for value in dimensions)
    if abs(size_scale - 1.0) < 1e-9:
        return x, y, z
    if label == "sphere":
        diameter = max(MIN_DIMENSION, max(x, y, z) * size_scale)
        return diameter, diameter, diameter
    if label in {"cylinder", "cone"}:
        diameter = max(MIN_DIMENSION, max(x, y) * size_scale)
        return diameter, diameter, max(MIN_DIMENSION, z * size_scale)
    if label == "box":
        return (
            max(MIN_DIMENSION, x * size_scale),
            max(MIN_DIMENSION, y * (1.0 + (size_scale - 1.0) * 0.55)),
            max(MIN_DIMENSION, z * size_scale),
        )
    return (
        max(MIN_DIMENSION, x * size_scale),
        max(MIN_DIMENSION, y * size_scale),
        max(MIN_DIMENSION, z * size_scale),
    )


def _load_depth_image(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float32) / 255.0
