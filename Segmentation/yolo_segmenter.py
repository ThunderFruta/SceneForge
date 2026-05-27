from __future__ import annotations

from pathlib import Path

from PIL import Image

from Runtime.device import resolve_torch_device
from Segmentation.types import SegmentDetection


def resolve_yolo_device(device: str | None) -> str | None:
    """Legacy compatibility wrapper for older YOLO-facing callers."""
    return resolve_torch_device(device)


class YoloSegmenter:
    def __init__(
        self,
        weights_path: str | Path,
        confidence: float = 0.25,
        device: str | None = None,
        overlap_iou_threshold: float = 0.7,
        retina_masks: bool = True,
    ) -> None:
        self.weights_path = Path(weights_path)
        if not self.weights_path.is_file():
            raise ValueError(f"YOLO weights path does not exist: {self.weights_path}")
        if confidence < 0.0 or confidence > 1.0:
            raise ValueError("--confidence must be between 0 and 1.")
        if overlap_iou_threshold < 0.0 or overlap_iou_threshold > 1.0:
            raise ValueError("--overlap-iou-threshold must be between 0 and 1.")

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for legacy YOLO backends. Install requirements.txt first."
            ) from exc

        self.model = YOLO(str(self.weights_path))
        self.confidence = confidence
        self.requested_device = device
        self.device = resolve_yolo_device(device)
        self.overlap_iou_threshold = overlap_iou_threshold
        self.retina_masks = bool(retina_masks)

    def detect(self, image: Image.Image) -> list[SegmentDetection]:
        import numpy as np

        image_array = np.asarray(image)
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

        return suppress_overlapping_detections(
            detections,
            iou_threshold=self.overlap_iou_threshold,
        )


def suppress_overlapping_detections(
    detections: list[SegmentDetection],
    iou_threshold: float,
) -> list[SegmentDetection]:
    if iou_threshold <= 0.0 or len(detections) <= 1:
        return detections

    kept: list[SegmentDetection] = []
    for detection in sorted(detections, key=lambda item: item.detector_confidence, reverse=True):
        if all(bbox_iou(detection.bbox_xyxy, kept_detection.bbox_xyxy) <= iou_threshold for kept_detection in kept):
            kept.append(detection)
    return sorted(kept, key=lambda item: detections.index(item))


def bbox_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])
    intersection_width = max(0.0, right - left)
    intersection_height = max(0.0, bottom - top)
    intersection_area = intersection_width * intersection_height

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union_area = area_a + area_b - intersection_area
    if union_area <= 0.0:
        return 0.0
    return intersection_area / union_area
