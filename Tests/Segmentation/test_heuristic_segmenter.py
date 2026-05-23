from __future__ import annotations

import pytest

from Segmentation.Core.segmentation_labels import SegmentationLabel
from Segmentation.Providers.Heuristic.heuristic_segmenter import build_heuristic_segmentation


def test_heuristic_segmentation_marks_valid_depth_as_detail() -> None:
    mask = build_heuristic_segmentation([[0.2, 0.8], [0.05, 0.01]])

    assert mask.labels == [
        [SegmentationLabel.DETAIL, SegmentationLabel.DETAIL],
        [SegmentationLabel.DETAIL, SegmentationLabel.UNKNOWN],
    ]


def test_heuristic_segmentation_rejects_ragged_depth() -> None:
    with pytest.raises(ValueError, match="Depth map rows"):
        build_heuristic_segmentation([[0.2], [0.3, 0.4]])
