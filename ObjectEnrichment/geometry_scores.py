from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from ObjectEnrichment.types import GeometryEvidence


BASE_LABELS = ("sphere", "box", "cylinder", "cone", "plane", "unknown")


def _get_cv2():
    try:
        import cv2 as cv2_module
    except ModuleNotFoundError:
        return None
    return cv2_module


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _shape_window_scores(
    *,
    area_ratio: float,
    aspect: float,
    extent: float,
    circularity: float,
    vertex_count: int,
    depth_std: float,
    edge_density: float,
) -> dict[str, float]:
    square = 1.0 - min(1.0, abs(aspect - 1.0))
    elongated = min(1.0, abs(np.log(max(0.05, aspect))))
    four_sided = _clamp_score(1.0 - abs(float(vertex_count) - 4.0) / 8.0)
    low_vertex = _clamp_score(1.0 - max(0.0, float(vertex_count) - 5.0) / 4.0)
    curved = _clamp_score((float(vertex_count) - 6.0) / 18.0)
    cone_extent = _clamp_score(1.0 - abs(extent - 0.64) * 5.5)
    cone_vertices = _clamp_score((float(vertex_count) - 5.0) / 3.0)
    round_extent = _clamp_score(1.0 - abs(extent - 0.78) * 3.0)
    smooth_depth = 1.0 - min(1.0, depth_std * 7.0)
    flat_depth = _clamp_score(1.0 - depth_std * 30.0)
    cone_depth = _clamp_score(1.0 - abs(depth_std - 0.035) * 15.0)
    blocky_depth = _clamp_score((extent - 0.68) * 5.0) * _clamp_score((depth_std - 0.055) / 0.05)
    high_depth_relief = _clamp_score((depth_std - 0.055) / 0.06)
    cylinder_side_extent = _clamp_score(1.0 - abs(extent - 0.58) * 6.0)
    cylinder_side_vertices = _clamp_score(1.0 - max(0.0, float(vertex_count) - 7.0) / 4.0)
    cylinder_high_fill = _clamp_score((extent - 0.78) * 6.0) * smooth_depth * _clamp_score((float(vertex_count) - 4.0) / 3.0)
    faceted = 1.0 - circularity
    del area_ratio

    return {
        "sphere": _clamp_score(0.36 * circularity + 0.14 * square + 0.28 * curved + 0.24 * round_extent - 0.24 * high_depth_relief),
        "box": _clamp_score(0.24 * low_vertex + 0.24 * faceted + 0.20 * square + 0.12 * flat_depth + 0.10 * four_sided + 0.35 * blocky_depth),
        "cylinder": _clamp_score(0.25 * smooth_depth + 0.26 * extent + 0.13 * round_extent + 0.10 * elongated + 0.06 * curved + 0.22 * cylinder_side_extent * cylinder_side_vertices + 0.16 * cylinder_high_fill + 0.02 * edge_density),
        "cone": _clamp_score(0.32 * cone_extent + 0.26 * cone_vertices * (1.0 - blocky_depth) + 0.18 * cone_depth + 0.14 * faceted + 0.10 * square),
        "plane": 0.0,
        "unknown": 0.12,
    }


def _largest_contour(mask: np.ndarray, cv2_module):
    contours, _ = cv2_module.findContours(mask.astype(np.uint8) * 255, cv2_module.RETR_EXTERNAL, cv2_module.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2_module.contourArea)


def _fallback_geometry_scores(mask: np.ndarray, depth: np.ndarray, edge: np.ndarray) -> GeometryEvidence:
    if mask.sum() == 0:
        return GeometryEvidence(
            selected_label="unknown",
            confidence=1.0,
            candidate_scores={label: float(label == "unknown") for label in BASE_LABELS},
        )

    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return GeometryEvidence(
            selected_label="unknown",
            confidence=1.0,
            candidate_scores={label: float(label == "unknown") for label in BASE_LABELS},
        )

    width = max(1.0, float(xs.max() - xs.min() + 1))
    height = max(1.0, float(ys.max() - ys.min() + 1))
    aspect = width / max(1.0, height)
    area = max(1.0, float(mask.sum()))
    extent = area / (width * height)
    square_aspect = 1.0 - min(1.0, abs(aspect - 1.0))
    elongated = min(1.0, abs(np.log(max(0.05, aspect))))
    straightness = _clamp_score(1.0 - min(1.0, abs(aspect - 1.0)))
    masked_depth = depth[mask]
    depth_std = float(np.std(masked_depth)) if masked_depth.size else 0.0
    edge_density = float(np.mean(edge[mask] > 0.15)) if edge[mask].size else 0.0

    scores = _shape_window_scores(
        area_ratio=area / max(1.0, float(mask.shape[0] * mask.shape[1])),
        aspect=aspect,
        extent=extent,
        circularity=square_aspect * 0.8,
        vertex_count=4 if straightness > 0.85 else 8,
        depth_std=depth_std,
        edge_density=edge_density,
    )

    selected_label, confidence = max(scores.items(), key=lambda item: (item[1], item[0]))
    if confidence < 0.20:
        selected_label = "unknown"
        confidence = max(confidence, scores["unknown"])
    return GeometryEvidence(selected_label=selected_label, confidence=confidence, candidate_scores=scores)


def classify_geometry(mask_path: str | Path, depth_path: str | Path, edge_path: str | Path, output_path: str | Path | None = None) -> GeometryEvidence:
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 127
    depth = np.asarray(Image.open(depth_path).convert("L"), dtype=np.float32) / 255.0
    edge = np.asarray(Image.open(edge_path).convert("L"), dtype=np.float32) / 255.0

    cv2 = _get_cv2()
    if cv2 is None:
        evidence = _fallback_geometry_scores(mask, depth, edge)
        _write_optional(evidence, output_path)
        return evidence

    if mask.sum() == 0:
        evidence = GeometryEvidence(selected_label="unknown", confidence=1.0, candidate_scores={label: float(label == "unknown") for label in BASE_LABELS})
        _write_optional(evidence, output_path)
        return evidence

    contour = _largest_contour(mask, cv2)
    if contour is None:
        evidence = GeometryEvidence(selected_label="unknown", confidence=1.0, candidate_scores={label: float(label == "unknown") for label in BASE_LABELS})
        _write_optional(evidence, output_path)
        return evidence

    area = max(1.0, float(cv2.contourArea(contour)))
    perimeter = max(1.0, float(cv2.arcLength(contour, True)))
    x, y, width, height = cv2.boundingRect(contour)
    aspect = width / max(1.0, float(height))
    extent = area / max(1.0, float(width * height))
    circularity = 4.0 * np.pi * area / (perimeter * perimeter)
    approx = cv2.approxPolyDP(contour, max(1.0, 0.018 * perimeter), True)
    vertex_count = len(approx)

    masked_depth = depth[mask]
    depth_std = float(np.std(masked_depth)) if masked_depth.size else 0.0
    edge_density = float(np.mean(edge[mask] > 0.15)) if edge[mask].size else 0.0

    scores = _shape_window_scores(
        area_ratio=area / max(1.0, float(mask.shape[0] * mask.shape[1])),
        aspect=aspect,
        extent=extent,
        circularity=circularity,
        vertex_count=vertex_count,
        depth_std=depth_std,
        edge_density=edge_density,
    )
    selected_label, confidence = max(scores.items(), key=lambda item: (item[1], item[0]))
    if confidence < 0.20:
        selected_label = "unknown"
        confidence = max(confidence, scores["unknown"])
    evidence = GeometryEvidence(selected_label=selected_label, confidence=confidence, candidate_scores=scores)
    _write_optional(evidence, output_path)
    return evidence


def _write_optional(evidence: GeometryEvidence, output_path: str | Path | None) -> None:
    if output_path is None:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evidence.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
