from __future__ import annotations

from enum import StrEnum


class SegmentationLabel(StrEnum):
    WALL = "wall"
    FLOOR = "floor"
    CEILING = "ceiling"
    OBJECT = "object"
    DETAIL = "detail"
    UNKNOWN = "unknown"


LABEL_COLORS: dict[SegmentationLabel, tuple[int, int, int]] = {
    SegmentationLabel.WALL: (255, 0, 0),
    SegmentationLabel.FLOOR: (0, 255, 0),
    SegmentationLabel.CEILING: (0, 0, 255),
    SegmentationLabel.OBJECT: (255, 255, 0),
    SegmentationLabel.DETAIL: (0, 255, 255),
    SegmentationLabel.UNKNOWN: (0, 0, 0),
}

COLOR_LABELS = {color: label for label, color in LABEL_COLORS.items()}
PLANE_LABELS = {
    SegmentationLabel.WALL,
    SegmentationLabel.FLOOR,
    SegmentationLabel.CEILING,
}


def color_to_label(color: tuple[int, int, int]) -> SegmentationLabel:
    return COLOR_LABELS.get(color, SegmentationLabel.UNKNOWN)
