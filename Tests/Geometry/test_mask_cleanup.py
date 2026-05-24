from __future__ import annotations

from Geometry.Cleanup.mask_cleanup import cleanup_segmentation_mask
from Segmentation.Core.segmentation_labels import SegmentationLabel
from Segmentation.Core.segmentation_mask import SegmentationMask


def test_cleanup_segmentation_mask_fills_small_unknown_hole() -> None:
    wall = SegmentationLabel.WALL
    unknown = SegmentationLabel.UNKNOWN
    mask = SegmentationMask.from_labels(
        [
            [wall, wall, wall],
            [wall, unknown, wall],
            [wall, wall, wall],
        ]
    )

    result = cleanup_segmentation_mask(mask, max_hole_cells=1)

    assert result.filled_mask_holes == 1
    assert result.mask.labels[1][1] == wall


def test_cleanup_segmentation_mask_preserves_large_unknown_opening() -> None:
    wall = SegmentationLabel.WALL
    unknown = SegmentationLabel.UNKNOWN
    mask = SegmentationMask.from_labels(
        [
            [wall, wall, wall, wall],
            [wall, unknown, unknown, wall],
            [wall, unknown, unknown, wall],
            [wall, wall, wall, wall],
        ]
    )

    result = cleanup_segmentation_mask(mask, max_hole_cells=3)

    assert result.filled_mask_holes == 0
    assert result.mask.labels[1][1] == unknown
    assert result.mask.labels[2][2] == unknown


def test_cleanup_segmentation_mask_removes_tiny_island_without_crossing_boundaries() -> None:
    wall = SegmentationLabel.WALL
    floor = SegmentationLabel.FLOOR
    obj = SegmentationLabel.OBJECT
    mask = SegmentationMask.from_labels(
        [
            [wall, wall, wall, wall],
            [wall, obj, wall, wall],
            [floor, floor, floor, floor],
        ]
    )

    result = cleanup_segmentation_mask(mask, max_island_cells=1)

    assert result.removed_mask_islands == 1
    assert result.mask.labels[1][1] == wall
    assert result.mask.labels[2] == [floor, floor, floor, floor]
