from __future__ import annotations

PRIMITIVE_LABELS = ("sphere", "box", "cylinder", "cone", "plane", "torus", "tube", "arch", "unknown")

PRIMITIVE_PROMPTS = {
    "sphere": "a photo of an object shaped like a sphere or ball",
    "cylinder": "a photo of an object shaped like a cylinder or tube",
    "cone": "a photo of an object shaped like a cone",
    "box": "a photo of an object shaped like a box or rectangular cuboid",
    "plane": "a photo of a flat planar surface",
    "torus": "a photo of an object shaped like a torus, hoop, or ring",
    "tube": "a photo of an object shaped like a hollow straight tube",
    "arch": "a photo of an object shaped like an arch or partial loop",
    "unknown": "a photo of an object with an unclear geometric shape",
}


def validate_primitive_label(label: str) -> str:
    if label not in PRIMITIVE_LABELS:
        raise ValueError(f"Unsupported primitive label: {label}")
    return label
