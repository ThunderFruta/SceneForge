from __future__ import annotations

from dataclasses import dataclass

from ShapeDetection.primitive_labels import validate_primitive_label


@dataclass(frozen=True)
class PrimitivePrediction:
    label: str
    confidence: float
    source: str = "classifier"

    def __post_init__(self) -> None:
        validate_primitive_label(self.label)
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError("Primitive confidence must be between 0 and 1.")
