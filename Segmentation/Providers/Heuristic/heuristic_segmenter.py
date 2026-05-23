from __future__ import annotations

from Segmentation.Core.segmentation_labels import SegmentationLabel
from Segmentation.Core.segmentation_mask import SegmentationMask


def build_heuristic_segmentation(
    depth_map: list[list[float]],
    *,
    min_valid_depth: float = 0.04,
) -> SegmentationMask:
    if not depth_map or not depth_map[0]:
        raise ValueError("Depth map must contain at least one pixel.")
    width = len(depth_map[0])
    if any(len(row) != width for row in depth_map):
        raise ValueError("Depth map rows must all have the same width.")

    labels = [
        [
            SegmentationLabel.UNKNOWN
            if depth < min_valid_depth
            else SegmentationLabel.DETAIL
            for depth in row
        ]
        for row in depth_map
    ]
    return SegmentationMask.from_labels(labels)
