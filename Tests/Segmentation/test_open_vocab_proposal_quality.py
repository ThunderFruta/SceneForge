from __future__ import annotations

from Segmentation.proposal_quality import summarize_open_vocab_proposals
from Segmentation.types import SegmentDetection


def test_open_vocab_quality_counts_fallbacks_duplicates_and_labels() -> None:
    detections = [
        SegmentDetection(
            bbox_xyxy=(0, 0, 10, 10),
            mask_polygon=[(0, 0), (10, 0), (10, 10), (0, 10)],
            detector_label="box",
            detector_confidence=0.9,
            proposal_source="groundingdino_box_fallback",
        ),
        SegmentDetection(
            bbox_xyxy=(0.5, 0.5, 10.5, 10.5),
            mask_polygon=[(1, 1), (3, 1), (3, 3), (1, 3)],
            detector_label="box",
            detector_confidence=0.8,
            proposal_source="sam3_box_prompt",
        ),
        SegmentDetection(
            bbox_xyxy=(40, 40, 42, 42),
            mask_polygon=[],
            detector_label="sphere",
            detector_confidence=0.7,
            proposal_source="sam3_text_prompt",
        ),
    ]

    summary = summarize_open_vocab_proposals(detections, image_width=64, image_height=64)

    assert summary["object_count"] == 3
    assert summary["rectangle_fallback_count"] == 1
    assert summary["duplicate_overlap_count"] == 1
    assert summary["empty_mask_count"] == 1
    assert summary["tiny_mask_count"] == 2
    assert summary["labels_seen"] == ["box", "sphere"]
    assert summary["proposal_sources"]["sam3_box_prompt"] == 1
