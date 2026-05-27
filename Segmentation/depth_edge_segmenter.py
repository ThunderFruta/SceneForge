from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from EdgeDetection.types import EdgeProvider
from Input.Depth.depth_loader import load_grayscale_depth
from Segmentation.backend import LearnedSegmentationModelSpec
from Segmentation.types import SegmentDetection


MIN_COMPONENT_AREA_RATIO = 0.0025
TINY_COMPONENT_AREA_RATIO = 0.0006
MAX_COMPONENTS = 64
EDGE_THRESHOLD = 0.18
DEPTH_GRADIENT_THRESHOLD = 0.035
PLANE_AREA_RATIO = 0.12
PLANE_DEPTH_STD_MAX = 0.16
SCENE_SUPPORT_AREA_RATIO = 0.60
MERGE_GAP_PIXELS = 10.0
MERGE_DEPTH_MEDIAN_MAX_DELTA = 0.075
MERGE_DEPTH_OVERLAP_MIN_RATIO = 0.35
MERGE_CHROMA_MAX_DELTA = 0.12
MERGE_LONG_CONTACT_CHROMA_MAX_DELTA = 0.08
MERGE_SAME_CHROMA_BBOX_OVERLAP_MIN_RATIO = 0.22
MERGE_MIN_BBOX_OVERLAP_PIXELS = 20.0
MERGE_MAX_CONTACT_PIXELS = 32
SUPPORT_CHROMA_DISTANCE_MIN = 0.08
SUPPORT_RGB_DISTANCE_MIN = 0.08


class DepthEdgeSegmenter:
    """Depth/edge instance-proposal scaffold for geometry-first reconstruction.

    This is intentionally not a primitive classifier. It proposes masks from depth
    discontinuities and edge evidence, then labels large flat surfaces as `plane`
    and all other instances as `unknown` so downstream geometric scoring owns the
    primitive decision.
    """

    backend = "depth-edge-instance-scaffold"
    input_channels = ("depth", "edge")
    backend_info = LearnedSegmentationModelSpec(
        architecture="depth_edge_geometry_first",
        input_channels=input_channels,
        output_contract="instance_masks_only",
    ).to_backend_info(
        name=backend,
        model_path=None,
    )

    def __init__(
        self,
        depth_path: str | Path,
        edge_path: str | Path | None = None,
        edge_provider: EdgeProvider | None = None,
        min_component_area_ratio: float = MIN_COMPONENT_AREA_RATIO,
        max_components: int = MAX_COMPONENTS,
    ) -> None:
        self.depth_path = Path(depth_path)
        self.edge_path = Path(edge_path) if edge_path else None
        self.edge_provider = edge_provider
        self.min_component_area_ratio = max(0.0, float(min_component_area_ratio))
        self.max_components = max(1, int(max_components))

    def detect(self, image: Image.Image) -> list[SegmentDetection]:
        depth = load_grayscale_depth(self.depth_path, expected_size=image.size)
        edge = self._load_or_compute_edge(image)
        boundaries = combined_boundary_mask(depth, edge)
        min_area = max(16, int(image.width * image.height * self.min_component_area_ratio))
        tiny_area = max(16, int(image.width * image.height * TINY_COMPONENT_AREA_RATIO))
        components = candidate_components_from_boundaries(
            boundaries,
            tiny_area=tiny_area,
            max_components=self.max_components * 4,
            image=image,
        )
        components = [
            component
            for component in components
            if should_keep_component(component, image.size, min_area)
        ]
        components = merge_object_fragments(components, depth, image)
        components = components[: self.max_components]

        detections: list[SegmentDetection] = []
        for component in components:
            bbox = component_bbox(component)
            label, confidence = component_label_and_confidence(component, depth, image.size)
            polygon = component_polygon(component, bbox)
            detections.append(
                SegmentDetection(
                    bbox_xyxy=bbox,
                    mask_polygon=polygon,
                    detector_label=label,
                    detector_confidence=confidence,
                )
            )
        return sorted(detections, key=lambda item: (item.detector_label != "plane", item.bbox_xyxy[1], item.bbox_xyxy[0]))

    def _load_or_compute_edge(self, image: Image.Image) -> np.ndarray:
        if self.edge_path is not None:
            edge_image = Image.open(self.edge_path).convert("L").resize(image.size)
        elif self.edge_provider is not None:
            edge_image = self.edge_provider.detect_edges(image).image.convert("L").resize(image.size)
        else:
            edge_image = image.convert("L").filter(ImageFilter.FIND_EDGES)
        return np.asarray(edge_image, dtype=np.float32) / 255.0


class EdgeReasonedDepthSegmenter(DepthEdgeSegmenter):
    """Object-level RGB/depth/edge detector backend.

    This is the current deterministic detector path: depth discontinuities and
    edge-enclosed regions produce face fragments, then RGB/depth/edge evidence
    merges likely faces into object-level masks.
    """

    backend = "depth-edge-object-detector"
    input_channels = ("rgb", "depth", "edge")
    backend_info = LearnedSegmentationModelSpec(
        architecture="rgb_depth_edge_object_detector",
        input_channels=input_channels,
        output_contract="instance_masks_only",
    ).to_backend_info(
        name=backend,
        model_path=None,
    )


def candidate_components_from_boundaries(
    boundaries: np.ndarray,
    tiny_area: int,
    max_components: int,
    image: Image.Image | None = None,
) -> list[np.ndarray]:
    raw_components = connected_components(~boundaries, min_area=tiny_area, max_components=max_components)
    sealed_components = connected_components(
        enclosed_regions(dilate_mask(boundaries)),
        min_area=tiny_area,
        max_components=max_components,
    )
    support_components = support_color_residual_components(raw_components, image, tiny_area, max_components)
    return suppress_duplicate_components(support_components + sealed_components + raw_components, max_components=max_components)


def support_color_residual_components(
    raw_components: list[np.ndarray],
    image: Image.Image | None,
    tiny_area: int,
    max_components: int,
) -> list[np.ndarray]:
    if image is None:
        return []
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    components: list[np.ndarray] = []
    for component in raw_components:
        if not is_scene_support_component(component, image.size):
            continue
        support_pixels = rgb[component]
        if support_pixels.size == 0:
            continue
        median_rgb = np.median(support_pixels, axis=0)
        median_chroma = median_rgb / max(float(np.linalg.norm(median_rgb)), 1e-6)
        chroma = rgb / np.maximum(np.linalg.norm(rgb, axis=2, keepdims=True), 1e-6)
        chroma_distance = np.linalg.norm(chroma - median_chroma, axis=2)
        rgb_distance = np.linalg.norm(rgb - median_rgb, axis=2)
        residual = (
            component
            & (chroma_distance >= SUPPORT_CHROMA_DISTANCE_MIN)
            & (rgb_distance >= SUPPORT_RGB_DISTANCE_MIN)
        )
        components.extend(connected_components(residual, min_area=tiny_area, max_components=max_components))
    return components


def enclosed_regions(boundaries: np.ndarray) -> np.ndarray:
    regions = ~boundaries
    height, width = regions.shape
    outside = np.zeros_like(regions, dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for x in range(width):
        if regions[0, x]:
            outside[0, x] = True
            queue.append((0, x))
        if regions[height - 1, x] and not outside[height - 1, x]:
            outside[height - 1, x] = True
            queue.append((height - 1, x))
    for y in range(height):
        if regions[y, 0] and not outside[y, 0]:
            outside[y, 0] = True
            queue.append((y, 0))
        if regions[y, width - 1] and not outside[y, width - 1]:
            outside[y, width - 1] = True
            queue.append((y, width - 1))

    while queue:
        current_y, current_x = queue.popleft()
        for next_y, next_x in neighbors(current_y, current_x, height, width):
            if regions[next_y, next_x] and not outside[next_y, next_x]:
                outside[next_y, next_x] = True
                queue.append((next_y, next_x))

    return regions & ~outside


def suppress_duplicate_components(
    components: list[np.ndarray],
    max_components: int,
    overlap_threshold: float = 0.55,
) -> list[np.ndarray]:
    kept: list[np.ndarray] = []
    for component in sorted(components, key=lambda item: int(item.sum()), reverse=True):
        if any(mask_iou(component, existing) >= overlap_threshold for existing in kept):
            continue
        kept.append(component)
        if len(kept) >= max_components:
            break
    return kept


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int((left & right).sum())
    if intersection == 0:
        return 0.0
    union = int((left | right).sum())
    return intersection / max(1.0, float(union))


def should_keep_component(component: np.ndarray, image_size: tuple[int, int], min_area: int) -> bool:
    if is_scene_support_component(component, image_size):
        return False
    area = int(component.sum())
    if area >= min_area:
        return True
    left, top, right, bottom = component_bbox(component)
    width = max(1.0, right - left)
    height = max(1.0, bottom - top)
    box_fill = float(area) / (width * height)
    aspect = max(width / height, height / width)
    return box_fill >= 0.30 and aspect <= 8.0


def combined_boundary_mask(depth: np.ndarray, edge: np.ndarray) -> np.ndarray:
    gradient_y, gradient_x = np.gradient(depth.astype(np.float32))
    depth_gradient = np.hypot(gradient_x, gradient_y)
    return (edge >= EDGE_THRESHOLD) | (depth_gradient >= DEPTH_GRADIENT_THRESHOLD)


def connected_components(mask: np.ndarray, min_area: int, max_components: int) -> list[np.ndarray]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for y in range(height):
        for x in range(width):
            if visited[y, x] or not mask[y, x]:
                continue
            points: list[tuple[int, int]] = []
            queue: deque[tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            while queue:
                current_y, current_x = queue.popleft()
                points.append((current_y, current_x))
                for next_y, next_x in neighbors(current_y, current_x, height, width):
                    if not visited[next_y, next_x] and mask[next_y, next_x]:
                        visited[next_y, next_x] = True
                        queue.append((next_y, next_x))
            if len(points) >= min_area:
                component = np.zeros_like(mask, dtype=bool)
                ys, xs = zip(*points)
                component[np.asarray(ys), np.asarray(xs)] = True
                components.append(component)

    components.sort(key=lambda item: int(item.sum()), reverse=True)
    return components[:max_components]


def merge_object_fragments(
    components: list[np.ndarray],
    depth: np.ndarray,
    image: Image.Image,
) -> list[np.ndarray]:
    if len(components) < 2:
        return components

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    merged = [component.copy() for component in components]

    changed = True
    while changed:
        changed = False
        stats = [component_stats(component, depth, rgb) for component in merged]
        for left_index in range(len(merged)):
            merge_index = None
            for right_index in range(left_index + 1, len(merged)):
                if should_merge_fragments(stats[left_index], stats[right_index]):
                    merge_index = right_index
                    break
            if merge_index is None:
                continue
            merged[left_index] = merged[left_index] | merged[merge_index]
            del merged[merge_index]
            changed = True
            break

    merged.sort(key=lambda item: int(item.sum()), reverse=True)
    return merged


def component_stats(component: np.ndarray, depth: np.ndarray, rgb: np.ndarray) -> dict:
    values = depth[component]
    colors = rgb[component]
    mean_rgb = colors.mean(axis=0) if colors.size else np.zeros(3, dtype=np.float32)
    chroma = mean_rgb / max(float(np.linalg.norm(mean_rgb)), 1e-6)
    return {
        "bbox": component_bbox(component),
        "depth_p10": float(np.percentile(values, 10)) if values.size else 1.0,
        "depth_p90": float(np.percentile(values, 90)) if values.size else 0.0,
        "depth_median": float(np.median(values)) if values.size else 1.0,
        "chroma": chroma,
        "component": component,
    }


def should_merge_fragments(left: dict, right: dict) -> bool:
    if not inflated_bboxes_intersect(left["bbox"], right["bbox"], MERGE_GAP_PIXELS):
        return False
    overlap_x, overlap_y = bbox_overlap_size(left["bbox"], right["bbox"])
    if min(overlap_x, overlap_y) < MERGE_MIN_BBOX_OVERLAP_PIXELS:
        return False
    chroma_delta = float(np.linalg.norm(left["chroma"] - right["chroma"]))
    if (
        bbox_smaller_overlap_ratio(left["bbox"], right["bbox"]) >= MERGE_SAME_CHROMA_BBOX_OVERLAP_MIN_RATIO
        and chroma_delta <= MERGE_LONG_CONTACT_CHROMA_MAX_DELTA
    ):
        return True
    depth_delta = abs(float(left["depth_median"]) - float(right["depth_median"]))
    if depth_delta > MERGE_DEPTH_MEDIAN_MAX_DELTA:
        return False
    if depth_overlap_ratio(left, right) < MERGE_DEPTH_OVERLAP_MIN_RATIO:
        return False
    if (
        contact_pixels(left["component"], right["component"]) > MERGE_MAX_CONTACT_PIXELS
        and chroma_delta > MERGE_LONG_CONTACT_CHROMA_MAX_DELTA
    ):
        return False
    return chroma_delta <= MERGE_CHROMA_MAX_DELTA


def bbox_smaller_overlap_ratio(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    overlap_x, overlap_y = bbox_overlap_size(left, right)
    overlap_area = overlap_x * overlap_y
    if overlap_area <= 0.0:
        return 0.0
    left_area = max(1.0, (left[2] - left[0]) * (left[3] - left[1]))
    right_area = max(1.0, (right[2] - right[0]) * (right[3] - right[1]))
    return overlap_area / min(left_area, right_area)


def bbox_overlap_size(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float]:
    left_x0, left_y0, left_x1, left_y1 = left
    right_x0, right_y0, right_x1, right_y1 = right
    return (
        max(0.0, min(left_x1, right_x1) - max(left_x0, right_x0)),
        max(0.0, min(left_y1, right_y1) - max(left_y0, right_y0)),
    )


def contact_pixels(left: np.ndarray, right: np.ndarray) -> int:
    dilated = np.zeros_like(left, dtype=bool)
    ys, xs = np.where(left)
    for dy in (-1, 0, 1):
        next_y = ys + dy
        valid_y = (next_y >= 0) & (next_y < left.shape[0])
        for dx in (-1, 0, 1):
            next_x = xs + dx
            valid = valid_y & (next_x >= 0) & (next_x < left.shape[1])
            dilated[next_y[valid], next_x[valid]] = True
    return int((dilated & right).sum())


def inflated_bboxes_intersect(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
    margin: float,
) -> bool:
    left_x0, left_y0, left_x1, left_y1 = left
    right_x0, right_y0, right_x1, right_y1 = right
    return (
        left_x0 - margin <= right_x1
        and left_x1 + margin >= right_x0
        and left_y0 - margin <= right_y1
        and left_y1 + margin >= right_y0
    )


def depth_overlap_ratio(left: dict, right: dict) -> float:
    low = max(float(left["depth_p10"]), float(right["depth_p10"]))
    high = min(float(left["depth_p90"]), float(right["depth_p90"]))
    overlap = max(0.0, high - low)
    span = max(float(left["depth_p90"]), float(right["depth_p90"])) - min(float(left["depth_p10"]), float(right["depth_p10"]))
    if span <= 0.0:
        return 1.0
    return overlap / span


def neighbors(y: int, x: int, height: int, width: int):
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        next_y = y + dy
        next_x = x + dx
        if 0 <= next_y < height and 0 <= next_x < width:
            yield next_y, next_x


def component_bbox(component: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.where(component)
    return (
        float(xs.min()),
        float(ys.min()),
        float(xs.max() + 1),
        float(ys.max() + 1),
    )


def bbox_polygon(bbox: tuple[float, float, float, float]) -> list[tuple[float, float]]:
    left, top, right, bottom = bbox
    return [(left, top), (right, top), (right, bottom), (left, bottom)]


def component_polygon(
    component: np.ndarray,
    bbox: tuple[float, float, float, float],
    max_points: int = 96,
) -> list[tuple[float, float]]:
    mask = close_component_mask(component)
    loops = component_contour_loops(mask)
    if not loops:
        return bbox_polygon(bbox)
    polygon = max(loops, key=lambda item: abs(polygon_area(item)))
    polygon = simplify_collinear_points(polygon)
    polygon = simplify_polygon_rdp(polygon, epsilon=0.75)
    if len(polygon) < 3:
        return bbox_polygon(bbox)
    if len(polygon) > max_points:
        indices = np.linspace(0, len(polygon) - 1, max_points, dtype=np.int64)
        polygon = [polygon[int(index)] for index in indices]
    return [(float(x), float(y)) for x, y in polygon]


def close_component_mask(component: np.ndarray) -> np.ndarray:
    return erode_mask(dilate_mask(component))


def dilate_mask(mask: np.ndarray) -> np.ndarray:
    output = mask.copy()
    for dy in (-1, 0, 1):
        y_src, y_dst = shifted_slices(mask.shape[0], dy)
        for dx in (-1, 0, 1):
            x_src, x_dst = shifted_slices(mask.shape[1], dx)
            output[y_dst, x_dst] |= mask[y_src, x_src]
    return output


def erode_mask(mask: np.ndarray) -> np.ndarray:
    output = np.ones_like(mask, dtype=bool)
    for dy in (-1, 0, 1):
        y_src, y_dst = shifted_slices(mask.shape[0], dy)
        for dx in (-1, 0, 1):
            x_src, x_dst = shifted_slices(mask.shape[1], dx)
            shifted = np.zeros_like(mask, dtype=bool)
            shifted[y_dst, x_dst] = mask[y_src, x_src]
            output &= shifted
    return output


def shifted_slices(size: int, offset: int) -> tuple[slice, slice]:
    if offset < 0:
        return slice(-offset, size), slice(0, size + offset)
    if offset > 0:
        return slice(0, size - offset), slice(offset, size)
    return slice(0, size), slice(0, size)


def component_contour_loops(component: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = component.shape
    edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    ys, xs = np.where(component)
    for y, x in zip(ys.tolist(), xs.tolist(), strict=True):
        if y == 0 or not component[y - 1, x]:
            add_edge(edges, (x, y), (x + 1, y))
        if x == width - 1 or not component[y, x + 1]:
            add_edge(edges, (x + 1, y), (x + 1, y + 1))
        if y == height - 1 or not component[y + 1, x]:
            add_edge(edges, (x + 1, y + 1), (x, y + 1))
        if x == 0 or not component[y, x - 1]:
            add_edge(edges, (x, y + 1), (x, y))

    loops: list[list[tuple[int, int]]] = []
    while edges:
        start = min(edges)
        current = start
        loop = [start]
        while True:
            options = edges.get(current)
            if not options:
                break
            next_point = options.pop(0)
            if not options:
                del edges[current]
            current = next_point
            if current == start:
                break
            loop.append(current)
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def add_edge(
    edges: dict[tuple[int, int], list[tuple[int, int]]],
    start: tuple[int, int],
    end: tuple[int, int],
) -> None:
    edges.setdefault(start, []).append(end)


def simplify_collinear_points(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if len(points) < 3:
        return points
    simplified: list[tuple[int, int]] = []
    for index, point in enumerate(points):
        previous = points[index - 1]
        next_point = points[(index + 1) % len(points)]
        dx1 = point[0] - previous[0]
        dy1 = point[1] - previous[1]
        dx2 = next_point[0] - point[0]
        dy2 = next_point[1] - point[1]
        if dx1 * dy2 == dy1 * dx2:
            continue
        simplified.append(point)
    return simplified


def simplify_polygon_rdp(points: list[tuple[int, int]], epsilon: float) -> list[tuple[int, int]]:
    if len(points) < 4:
        return points
    open_points = points + [points[0]]
    simplified = rdp(open_points, epsilon)
    if simplified and simplified[-1] == simplified[0]:
        simplified = simplified[:-1]
    return simplified


def rdp(points: list[tuple[int, int]], epsilon: float) -> list[tuple[int, int]]:
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
    point: tuple[int, int],
    start: tuple[int, int],
    end: tuple[int, int],
) -> float:
    px, py = point
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    if dx == 0 and dy == 0:
        return float(np.hypot(px - sx, py - sy))
    return abs(dy * px - dx * py + ex * sy - ey * sx) / float(np.hypot(dx, dy))


def polygon_area(points: list[tuple[int, int]]) -> float:
    area = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        area += point[0] * next_point[1] - next_point[0] * point[1]
    return area / 2.0


def is_scene_support_component(component: np.ndarray, image_size: tuple[int, int]) -> bool:
    image_width, image_height = image_size
    area_ratio = float(component.sum()) / max(1.0, float(image_width * image_height))
    if area_ratio < SCENE_SUPPORT_AREA_RATIO:
        return False
    left, top, right, bottom = component_bbox(component)
    border_margin = 2.0
    touches_left = left <= border_margin
    touches_right = right >= float(image_width) - border_margin
    touches_top = top <= border_margin
    touches_bottom = bottom >= float(image_height) - border_margin
    return touches_left and touches_right and touches_top and touches_bottom


def component_label_and_confidence(
    component: np.ndarray,
    depth: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[str, float]:
    image_width, image_height = image_size
    area_ratio = float(component.sum()) / max(1.0, float(image_width * image_height))
    values = depth[component]
    depth_std = float(np.std(values)) if values.size else 1.0
    if area_ratio >= PLANE_AREA_RATIO and depth_std <= PLANE_DEPTH_STD_MAX:
        confidence = min(0.95, 0.55 + area_ratio + max(0.0, PLANE_DEPTH_STD_MAX - depth_std))
        return "plane", confidence
    confidence = min(0.85, 0.35 + min(0.30, area_ratio * 2.0) + max(0.0, 0.20 - depth_std))
    return "unknown", confidence
