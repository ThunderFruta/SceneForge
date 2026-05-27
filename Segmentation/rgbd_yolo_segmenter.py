from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from Input.Depth.depth_loader import load_grayscale_depth
from Segmentation.types import SegmentDetection
from Segmentation.yolo_segmenter import YoloSegmenter, bbox_iou, suppress_overlapping_detections


DEFAULT_RGBD_CHANNEL_WEIGHTS = (0.20, 0.20, 0.20, 0.40)
EQUAL_RGBD_CHANNEL_WEIGHT = 0.25
LOW_CONFIDENCE_PLANE_THRESHOLD = 0.35
LARGE_PLANE_AREA_RATIO = 0.10
LOW_CONFIDENCE_PLANE_MAX_FOREGROUND_OVERLAP_RATIO = 0.35
LOW_CONFIDENCE_PLANE_MAX_DEPTH_STD = 0.18
PLANE_DUPLICATE_IOU_THRESHOLD = 0.45
PLANE_DUPLICATE_CONTAINMENT_THRESHOLD = 0.60


def make_bgrd_array(
    image: Image.Image,
    depth_path: str | Path,
    channel_weights: tuple[float, float, float, float] = DEFAULT_RGBD_CHANNEL_WEIGHTS,
) -> np.ndarray:
    depth = load_grayscale_depth(depth_path, expected_size=image.size)
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    bgr = rgb[..., ::-1]
    depth_u8 = np.clip(depth * 255.0, 0, 255).astype(np.uint8)
    bgrd = np.dstack([bgr, depth_u8]).astype(np.float32)
    return apply_rgbd_channel_weights(bgrd, channel_weights)


def apply_rgbd_channel_weights(
    bgrd: np.ndarray,
    channel_weights: tuple[float, float, float, float] = DEFAULT_RGBD_CHANNEL_WEIGHTS,
) -> np.ndarray:
    if bgrd.ndim != 3 or bgrd.shape[2] != 4:
        return bgrd
    multipliers = np.asarray(normalized_channel_weights(channel_weights), dtype=np.float32) / EQUAL_RGBD_CHANNEL_WEIGHT
    weighted = bgrd * multipliers.reshape((1, 1, 4))
    return np.clip(weighted, 0, 255).astype(np.uint8)


def normalized_channel_weights(values: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    if len(values) != 4:
        raise ValueError("RGBD channel weights must contain exactly four values.")
    weights = tuple(max(0.0, float(value)) for value in values)
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("RGBD channel weights must include at least one positive value.")
    return tuple(value / total for value in weights)


def parse_channel_weights(value: str | tuple[float, float, float, float] | None) -> tuple[float, float, float, float]:
    if value is None:
        return DEFAULT_RGBD_CHANNEL_WEIGHTS
    if isinstance(value, tuple):
        return normalized_channel_weights(value)
    parts = [part.strip() for part in str(value).replace(":", ",").split(",") if part.strip()]
    if len(parts) != 4:
        raise ValueError("--rgbd-channel-weights must contain four comma-separated values, e.g. 0.20,0.20,0.20,0.40.")
    return normalized_channel_weights(tuple(float(part) for part in parts))


def suppress_unreliable_plane_detections(
    detections: list[SegmentDetection],
    image_size: tuple[int, int],
    depth: np.ndarray | None = None,
) -> list[SegmentDetection]:
    if len(detections) <= 1:
        return detections

    image_width, image_height = image_size
    image_area = max(1.0, float(image_width * image_height))
    kept: list[SegmentDetection] = []
    planes_by_confidence = sorted(
        [item for item in detections if _is_plane(item)],
        key=lambda item: (item.detector_confidence, polygon_area(item.mask_polygon)),
        reverse=True,
    )

    kept_planes: list[SegmentDetection] = []
    for plane in planes_by_confidence:
        if _is_duplicate_plane(plane, kept_planes):
            continue
        if plane.detector_confidence < LOW_CONFIDENCE_PLANE_THRESHOLD and not _is_large_clean_plane(
            plane,
            detections,
            image_area,
            image_size,
            depth,
        ):
            continue
        kept_planes.append(plane)

    kept_plane_ids = {id(item) for item in kept_planes}
    for detection in detections:
        if not _is_plane(detection) or id(detection) in kept_plane_ids:
            kept.append(detection)
    return kept


def _is_plane(detection: SegmentDetection) -> bool:
    return str(detection.detector_label).lower() == "plane"


def _is_duplicate_plane(plane: SegmentDetection, kept_planes: list[SegmentDetection]) -> bool:
    for kept in kept_planes:
        if bbox_iou(plane.bbox_xyxy, kept.bbox_xyxy) >= PLANE_DUPLICATE_IOU_THRESHOLD:
            return True
        if bbox_intersection_over_min_area(plane.bbox_xyxy, kept.bbox_xyxy) >= PLANE_DUPLICATE_CONTAINMENT_THRESHOLD:
            return True
    return False


def _is_large_clean_plane(
    plane: SegmentDetection,
    detections: list[SegmentDetection],
    image_area: float,
    image_size: tuple[int, int],
    depth: np.ndarray | None,
) -> bool:
    plane_area = max(polygon_area(plane.mask_polygon), bbox_area(plane.bbox_xyxy))
    if plane_area / image_area < LARGE_PLANE_AREA_RATIO:
        return False
    foreground_overlap = foreground_overlap_ratio(plane, detections)
    if foreground_overlap > LOW_CONFIDENCE_PLANE_MAX_FOREGROUND_OVERLAP_RATIO:
        return False
    depth_std = plane_depth_std(plane, image_size, depth)
    return depth_std is None or depth_std <= LOW_CONFIDENCE_PLANE_MAX_DEPTH_STD


def foreground_overlap_ratio(plane: SegmentDetection, detections: list[SegmentDetection]) -> float:
    plane_area = max(1.0, bbox_area(plane.bbox_xyxy))
    overlap = 0.0
    for detection in detections:
        if detection is plane or _is_plane(detection):
            continue
        overlap += bbox_intersection_area(plane.bbox_xyxy, detection.bbox_xyxy)
    return min(1.0, overlap / plane_area)


def plane_depth_std(
    plane: SegmentDetection,
    image_size: tuple[int, int],
    depth: np.ndarray | None,
) -> float | None:
    if depth is None:
        return None
    mask = polygon_mask(plane.mask_polygon, image_size)
    values = depth[mask]
    if values.size < 32:
        return None
    return float(np.std(values))


def polygon_mask(polygon: list[tuple[float, float]], image_size: tuple[int, int]) -> np.ndarray:
    image_width, image_height = image_size
    mask = Image.new("1", (image_width, image_height), 0)
    if len(polygon) >= 3:
        ImageDraw.Draw(mask).polygon([(float(x), float(y)) for x, y in polygon], fill=1)
    return np.asarray(mask, dtype=bool)


def polygon_area(polygon: list[tuple[float, float]]) -> float:
    if len(polygon) < 3:
        return 0.0
    area = 0.0
    previous_x, previous_y = polygon[-1]
    for x, y in polygon:
        area += previous_x * y - x * previous_y
        previous_x, previous_y = x, y
    return abs(area) * 0.5


def bbox_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def bbox_intersection_area(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])
    return max(0.0, right - left) * max(0.0, bottom - top)


def bbox_intersection_over_min_area(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    denominator = max(MIN_BBOX_AREA, min(bbox_area(box_a), bbox_area(box_b)))
    return bbox_intersection_area(box_a, box_b) / denominator


MIN_BBOX_AREA = 1.0


class RgbdYoloSegmenter(YoloSegmenter):
    def __init__(
        self,
        weights_path: str | Path,
        depth_path: str | Path,
        confidence: float = 0.25,
        device: str | None = None,
        overlap_iou_threshold: float = 0.7,
        retina_masks: bool = True,
        channel_weights: str | tuple[float, float, float, float] | None = None,
    ) -> None:
        super().__init__(
            weights_path=weights_path,
            confidence=confidence,
            device=device,
            overlap_iou_threshold=overlap_iou_threshold,
            retina_masks=retina_masks,
        )
        self.depth_path = Path(depth_path)
        self.channel_weights = parse_channel_weights(channel_weights)

    def detect(self, image: Image.Image) -> list[SegmentDetection]:
        depth = load_grayscale_depth(self.depth_path, expected_size=image.size)
        image_array = make_bgrd_array(image, self.depth_path, self.channel_weights)
        results = self.model.predict(
            source=image_array,
            conf=self.confidence,
            device=self.device,
            retina_masks=self.retina_masks,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        masks = getattr(result, "masks", None)
        mask_polygons = list(getattr(masks, "xy", []) or [])
        names = getattr(result, "names", {}) or {}
        detections: list[SegmentDetection] = []

        for index, box in enumerate(boxes):
            xyxy = box.xyxy[0].detach().cpu().tolist()
            confidence = float(box.conf[0].detach().cpu().item())
            class_id = int(box.cls[0].detach().cpu().item())
            label = str(names.get(class_id, class_id))

            polygon: list[tuple[float, float]] = []
            if index < len(mask_polygons):
                polygon = [
                    (float(point[0]), float(point[1]))
                    for point in mask_polygons[index]
                ]
            if not polygon:
                left, top, right, bottom = [float(value) for value in xyxy]
                polygon = [(left, top), (right, top), (right, bottom), (left, bottom)]

            detections.append(
                SegmentDetection(
                    bbox_xyxy=tuple(float(value) for value in xyxy),
                    mask_polygon=polygon,
                    detector_label=label,
                    detector_confidence=confidence,
                ).normalized(image.width, image.height)
            )

        overlap_suppressed = suppress_overlapping_detections(
            detections,
            iou_threshold=self.overlap_iou_threshold,
        )
        return suppress_unreliable_plane_detections(
            overlap_suppressed,
            image_size=image.size,
            depth=depth,
        )
