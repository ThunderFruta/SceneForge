from __future__ import annotations

from Segmentation.Core.segmentation_labels import SegmentationLabel
from Segmentation.Core.segmentation_mask import SegmentationMask
from Segmentation.Integration.mask_to_regions import segmentation_mask_to_regions


def test_mask_to_regions_creates_plane_and_detail_regions() -> None:
    mask = SegmentationMask.from_labels(
        [
            [SegmentationLabel.WALL, SegmentationLabel.WALL],
            [SegmentationLabel.OBJECT, SegmentationLabel.OBJECT],
        ]
    )

    regions = segmentation_mask_to_regions(
        mask,
        [[0.4, 0.8], [0.2, 0.9]],
        analysis_columns=2,
        analysis_rows=2,
    )

    assert [(region.name, region.kind, region.cells, region.bounds) for region in regions] == [
        ("plane_000", "plane", [(0, 0), (1, 0)], (0, 0, 2, 1)),
        ("detail_000", "detail", [(0, 1), (1, 1)], (0, 1, 2, 2)),
    ]


def test_mask_to_regions_preserves_l_shape_and_hole_cells() -> None:
    mask = SegmentationMask.from_labels(
        [
            [SegmentationLabel.FLOOR, SegmentationLabel.FLOOR, SegmentationLabel.UNKNOWN],
            [SegmentationLabel.FLOOR, SegmentationLabel.UNKNOWN, SegmentationLabel.UNKNOWN],
            [SegmentationLabel.UNKNOWN, SegmentationLabel.UNKNOWN, SegmentationLabel.FLOOR],
        ]
    )

    regions = segmentation_mask_to_regions(
        mask,
        [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
        analysis_columns=3,
        analysis_rows=3,
    )

    assert [(region.kind, region.cells) for region in regions] == [
        ("plane", [(0, 0), (1, 0), (0, 1)]),
        ("detail", [(2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]),
        ("plane", [(2, 2)]),
    ]


def test_mask_to_regions_maps_unknown_to_detail_and_skips_invalid_depth() -> None:
    mask = SegmentationMask.from_labels(
        [
            [SegmentationLabel.WALL, SegmentationLabel.WALL],
            [SegmentationLabel.UNKNOWN, SegmentationLabel.FLOOR],
        ]
    )

    regions = segmentation_mask_to_regions(
        mask,
        [[0.01, 0.5], [0.5, 0.01]],
        analysis_columns=2,
        analysis_rows=2,
    )

    assert [(region.kind, region.cells) for region in regions] == [
        ("plane", [(1, 0)]),
        ("detail", [(0, 1)]),
    ]
