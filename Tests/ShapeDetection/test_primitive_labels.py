from __future__ import annotations

import pytest

from ShapeDetection.primitive_labels import PRIMITIVE_LABELS, validate_primitive_label


def test_primitive_labels_are_fixed_for_v1() -> None:
    assert PRIMITIVE_LABELS == ("sphere", "box", "cylinder", "cone", "plane", "torus", "tube", "arch", "unknown")


def test_validate_primitive_label_rejects_unknown_label() -> None:
    with pytest.raises(ValueError):
        validate_primitive_label("capsule")
