from __future__ import annotations

from dataclasses import dataclass

from Segmentation.Core.segmentation_labels import SegmentationLabel


@dataclass(frozen=True)
class SegmentationMask:
    labels: list[list[SegmentationLabel]]
    width: int
    height: int

    @classmethod
    def from_labels(cls, labels: list[list[SegmentationLabel]]) -> "SegmentationMask":
        if not labels or not labels[0]:
            raise ValueError("Segmentation mask must contain at least one pixel.")
        width = len(labels[0])
        if any(len(row) != width for row in labels):
            raise ValueError("Segmentation mask rows must all have the same width.")
        return cls(labels=labels, width=width, height=len(labels))

    def label_at(self, x: int, y: int) -> SegmentationLabel:
        return self.labels[y][x]

    def validate_size(self, *, width: int, height: int) -> None:
        if (self.width, self.height) != (width, height):
            raise ValueError(
                "Segmentation mask dimensions must match the image and depth map. "
                f"Mask is {(self.width, self.height)}, expected {(width, height)}."
            )
