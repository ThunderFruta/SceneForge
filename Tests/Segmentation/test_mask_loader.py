from __future__ import annotations

from PIL import Image
import pytest

from Segmentation.Core.segmentation_labels import SegmentationLabel
from Segmentation.Providers.Manual.mask_loader import load_segmentation_mask


def test_load_segmentation_mask_maps_known_colors(tmp_path) -> None:
    mask_path = tmp_path / "mask.png"
    image = Image.new("RGB", (3, 2))
    image.putdata(
        [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (0, 255, 255),
            (0, 0, 0),
        ]
    )
    image.save(mask_path)

    mask = load_segmentation_mask(mask_path)

    assert mask.width == 3
    assert mask.height == 2
    assert mask.labels == [
        [SegmentationLabel.WALL, SegmentationLabel.FLOOR, SegmentationLabel.CEILING],
        [SegmentationLabel.OBJECT, SegmentationLabel.DETAIL, SegmentationLabel.UNKNOWN],
    ]


def test_load_segmentation_mask_maps_unknown_colors_to_unknown(tmp_path) -> None:
    mask_path = tmp_path / "mask.png"
    Image.new("RGB", (1, 1), (12, 34, 56)).save(mask_path)

    mask = load_segmentation_mask(mask_path)

    assert mask.labels == [[SegmentationLabel.UNKNOWN]]


def test_segmentation_mask_size_validation_has_clear_error(tmp_path) -> None:
    mask_path = tmp_path / "mask.png"
    Image.new("RGB", (1, 1), (255, 0, 0)).save(mask_path)
    mask = load_segmentation_mask(mask_path)

    with pytest.raises(ValueError, match="dimensions must match"):
        mask.validate_size(width=2, height=1)
