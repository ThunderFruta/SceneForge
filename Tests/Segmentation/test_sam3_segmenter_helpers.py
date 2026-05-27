from __future__ import annotations

from Segmentation.sam3_segmenter import xyxy_to_normalized_cxcywh


def test_xyxy_to_normalized_cxcywh_uses_sam3_geometric_prompt_contract() -> None:
    box = xyxy_to_normalized_cxcywh((10.0, 20.0, 50.0, 100.0), {"original_width": 100, "original_height": 200})

    assert box == [0.3, 0.3, 0.4, 0.4]
