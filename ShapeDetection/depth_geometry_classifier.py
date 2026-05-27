from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from Input.Depth.depth_loader import load_grayscale_depth
from PrimitiveFitting.masks import polygon_to_mask
from Segmentation.depth_edge_segmenter import component_bbox, component_polygon
from Segmentation.types import SegmentDetection
from ShapeDetection.types import PrimitivePrediction


class DepthGeometryPrimitiveClassifier:
    """Weak primitive classifier for geometry-first detection reports.

    This is intentionally lightweight. It gives `detections.json` useful labels
    from object masks plus aligned depth, while enrichment/fusion remains the
    stronger downstream primitive decision.
    """

    source = "depth_geometry_weak"

    def __init__(self, depth_path: str | Path) -> None:
        self.depth_path = Path(depth_path)
        self._depth_cache: dict[tuple[int, int], np.ndarray] = {}

    def classify(self, image: Image.Image, detection: SegmentDetection) -> PrimitivePrediction:
        depth = self._depth_for(image.size)
        mask = polygon_to_mask(detection.mask_polygon, image.width, image.height)
        if int(mask.sum()) <= 0:
            return PrimitivePrediction(label="unknown", confidence=0.0, source=self.source)

        scores = primitive_scores(image, depth, mask)
        label, confidence = max(scores.items(), key=lambda item: (item[1], item[0]))
        if confidence < 0.30:
            label = "unknown"
            confidence = max(confidence, scores.get("unknown", 0.12))
        return PrimitivePrediction(label=label, confidence=float(confidence), source=self.source)

    def _depth_for(self, image_size: tuple[int, int]) -> np.ndarray:
        if image_size not in self._depth_cache:
            self._depth_cache[image_size] = load_grayscale_depth(self.depth_path, expected_size=image_size)
        return self._depth_cache[image_size]


def primitive_scores(image: Image.Image, depth: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    del image
    ys, xs = np.where(mask)
    width = max(1.0, float(xs.max() - xs.min() + 1))
    height = max(1.0, float(ys.max() - ys.min() + 1))
    aspect = width / height
    area = max(1.0, float(mask.sum()))
    extent = area / max(1.0, width * height)
    square = 1.0 - min(1.0, abs(aspect - 1.0))
    elongated = min(1.0, abs(np.log(max(0.05, aspect))))
    depth_values = depth[mask]
    depth_std = float(np.std(depth_values)) if depth_values.size else 0.0
    contour = component_polygon(mask, component_bbox(mask), max_points=192)
    vertex_count = len(simplify_polygon(contour, epsilon=max(width, height) * 0.035))
    circularity = mask_circularity(mask)
    four_sided = clamp(1.0 - abs(vertex_count - 4.0) / 8.0)
    low_vertex = clamp(1.0 - max(0.0, vertex_count - 5.0) / 4.0)
    curved = clamp((vertex_count - 6.0) / 18.0)
    cone_extent = clamp(1.0 - abs(extent - 0.64) * 5.5)
    cone_vertices = clamp((vertex_count - 5.0) / 3.0)
    round_extent = clamp(1.0 - abs(extent - 0.78) * 3.0)
    smooth_depth = 1.0 - min(1.0, depth_std * 7.0)
    flat_depth = clamp(1.0 - depth_std * 30.0)
    cone_depth = clamp(1.0 - abs(depth_std - 0.035) * 15.0)
    blocky_depth = clamp((extent - 0.68) * 5.0) * clamp((depth_std - 0.055) / 0.05)
    high_depth_relief = clamp((depth_std - 0.055) / 0.06)
    cylinder_side_extent = clamp(1.0 - abs(extent - 0.58) * 6.0)
    cylinder_side_vertices = clamp(1.0 - max(0.0, vertex_count - 7.0) / 4.0)
    cylinder_high_fill = clamp((extent - 0.78) * 6.0) * smooth_depth * clamp((vertex_count - 4.0) / 3.0)
    faceted = 1.0 - circularity

    return {
        "sphere": clamp(0.36 * circularity + 0.14 * square + 0.28 * curved + 0.24 * round_extent - 0.24 * high_depth_relief),
        "box": clamp(0.24 * low_vertex + 0.24 * faceted + 0.20 * square + 0.12 * flat_depth + 0.10 * four_sided + 0.35 * blocky_depth),
        "cylinder": clamp(0.25 * smooth_depth + 0.26 * extent + 0.13 * round_extent + 0.10 * elongated + 0.08 * curved + 0.22 * cylinder_side_extent * cylinder_side_vertices + 0.16 * cylinder_high_fill),
        "cone": clamp(0.32 * cone_extent + 0.26 * cone_vertices * (1.0 - blocky_depth) + 0.18 * cone_depth + 0.14 * faceted + 0.10 * square),
        "plane": 0.0,
        "unknown": 0.12,
    }


def mask_circularity(mask: np.ndarray) -> float:
    area = max(1.0, float(mask.sum()))
    boundary = boundary_pixel_count(mask)
    if boundary <= 0.0:
        return 0.0
    return clamp(4.0 * np.pi * area / (boundary * boundary))


def boundary_pixel_count(mask: np.ndarray) -> float:
    interior = mask.copy()
    for dy in (-1, 0, 1):
        y_src, y_dst = shifted_slices(mask.shape[0], dy)
        for dx in (-1, 0, 1):
            x_src, x_dst = shifted_slices(mask.shape[1], dx)
            shifted = np.zeros_like(mask, dtype=bool)
            shifted[y_dst, x_dst] = mask[y_src, x_src]
            interior &= shifted
    return float((mask & ~interior).sum())


def shifted_slices(size: int, offset: int) -> tuple[slice, slice]:
    if offset < 0:
        return slice(-offset, size), slice(0, size + offset)
    if offset > 0:
        return slice(0, size - offset), slice(offset, size)
    return slice(0, size), slice(0, size)


def simplify_polygon(points: list[tuple[float, float]], epsilon: float) -> list[tuple[float, float]]:
    if len(points) <= 2:
        return points
    closed = points + [points[0]]
    simplified = rdp(closed, epsilon)
    if simplified and simplified[-1] == simplified[0]:
        simplified = simplified[:-1]
    return simplified


def rdp(points: list[tuple[float, float]], epsilon: float) -> list[tuple[float, float]]:
    if len(points) <= 2:
        return points
    start = points[0]
    end = points[-1]
    distances = [point_line_distance(point, start, end) for point in points[1:-1]]
    if not distances:
        return points
    max_index = int(np.argmax(distances)) + 1
    if distances[max_index - 1] <= epsilon:
        return [start, end]
    return rdp(points[: max_index + 1], epsilon)[:-1] + rdp(points[max_index:], epsilon)


def point_line_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    px, py = point
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    if dx == 0.0 and dy == 0.0:
        return float(np.hypot(px - sx, py - sy))
    return abs(dy * px - dx * py + ex * sy - ey * sx) / float(np.hypot(dx, dy))


def clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
