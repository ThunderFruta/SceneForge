from __future__ import annotations

from collections import Counter
from typing import Iterable

from Segmentation.types import SegmentDetection


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
