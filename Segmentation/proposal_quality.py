from __future__ import annotations

from collections import Counter
from typing import Iterable

from Segmentation.types import SegmentDetection

OPEN_VOCAB_BACKGROUND_LABELS = frozenset(
    {
        "floor",
        "wall",
        "ceiling",
        "ground",
        "background",
        "plane",
    }
)


def filter_open_vocab_segments(
    detections: Iterable[SegmentDetection],
    *,
    image_width: int,
    image_height: int,
    background_labels: frozenset[str] | set[str] = OPEN_VOCAB_BACKGROUND_LABELS,
    max_background_area_ratio: float = 0.20,
    max_bbox_area_ratio: float = 0.96,
    min_bbox_area_ratio: float = 0.0006,
) -> tuple[list[SegmentDetection], dict[str, int]]:
    items = list(detections)
    if not items:
        return [], {"input_count": 0, "output_count": 0}

    filtered: list[SegmentDetection] = []
    filtered_by_label = Counter()
    image_area = max(1.0, float(image_width) * float(image_height))
    normalized_background_labels = {str(label).strip().lower() for label in background_labels}

    for detection in items:
        area_ratio = bbox_area_ratio(detection.bbox_xyxy, image_area)
        label = str(detection.detector_label).strip().lower()

        if area_ratio > max_bbox_area_ratio:
            filtered_by_label["max_bbox_area_ratio"] += 1
            continue
        if area_ratio < min_bbox_area_ratio:
            filtered_by_label["min_bbox_area_ratio"] += 1
            continue
        if label in normalized_background_labels and area_ratio >= max_background_area_ratio:
            filtered_by_label["background_label_large_area"] += 1
            continue
        filtered.append(detection)

    total_filtered = sum(filtered_by_label.values())
    return filtered, {
        "input_count": len(items),
        "output_count": len(filtered),
        "filtered_count": total_filtered,
        **{key: int(value) for key, value in filtered_by_label.items()},
    }


def is_open_vocab_model_info(model_info: dict) -> bool:
    backend = str(model_info.get("detector_backend", ""))
    contract = str(model_info.get("detector_backend_info", {}).get("output_contract", ""))
    return "open-vocabulary" in backend or contract.startswith("open_vocab")


def summarize_open_vocab_proposals(
    detections: Iterable[SegmentDetection],
    *,
    image_width: int,
    image_height: int,
    tiny_mask_pixels: float = 64.0,
    duplicate_iou_threshold: float = 0.75,
) -> dict:
    items = list(detections)
    labels = sorted({item.detector_label for item in items if item.detector_label})
    sources = Counter(str(getattr(item, "proposal_source", "unknown")) for item in items)
    empty_count = sum(1 for item in items if len(item.mask_polygon) < 3)
    tiny_count = sum(1 for item in items if polygon_area(item.mask_polygon) < tiny_mask_pixels)
    rectangle_fallback_count = sum(1 for item in items if str(getattr(item, "proposal_source", "")) in {"groundingdino_box_fallback", "box_fallback"})
    duplicate_count = 0
    for index, left in enumerate(items):
        for right in items[index + 1 :]:
            if bbox_iou(left.bbox_xyxy, right.bbox_xyxy) >= duplicate_iou_threshold:
                duplicate_count += 1
    return {
        "schema_version": 1,
        "object_count": len(items),
        "empty_mask_count": empty_count,
        "rectangle_fallback_count": rectangle_fallback_count,
        "tiny_mask_count": tiny_count,
        "duplicate_overlap_count": duplicate_count,
        "duplicate_iou_threshold": duplicate_iou_threshold,
        "tiny_mask_pixels": tiny_mask_pixels,
        "labels_seen": labels,
        "proposal_sources": dict(sorted(sources.items())),
        "image_width": image_width,
        "image_height": image_height,
    }


def polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    total = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        total += (x1 * y2) - (x2 * y1)
    return abs(total) / 2.0


def bbox_iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1 = max(lx1, rx1)
    iy1 = max(ly1, ry1)
    ix2 = min(lx2, rx2)
    iy2 = min(ly2, ry2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih
    if intersection <= 0.0:
        return 0.0
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def bbox_area_ratio(bbox: tuple[float, float, float, float], image_area: float) -> float:
    left, top, right, bottom = bbox
    width = max(0.0, right - left)
    height = max(0.0, bottom - top)
    return (width * height) / max(1.0, image_area)
