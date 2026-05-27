from Segmentation.types import SegmentDetection
from Segmentation.yolo_segmenter import bbox_iou, suppress_overlapping_detections


def make_detection(
    bbox: tuple[float, float, float, float],
    confidence: float,
    label: str,
) -> SegmentDetection:
    left, top, right, bottom = bbox
    return SegmentDetection(
        bbox_xyxy=bbox,
        mask_polygon=[(left, top), (right, top), (right, bottom), (left, bottom)],
        detector_label=label,
        detector_confidence=confidence,
    )


def test_bbox_iou_is_one_for_identical_boxes() -> None:
    assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_suppress_overlapping_detections_keeps_higher_confidence() -> None:
    high = make_detection((0, 0, 10, 10), 0.9, "box")
    low = make_detection((1, 1, 11, 11), 0.5, "plane")
    far = make_detection((30, 30, 40, 40), 0.4, "sphere")

    detections = suppress_overlapping_detections([low, far, high], iou_threshold=0.5)

    assert detections == [far, high]
