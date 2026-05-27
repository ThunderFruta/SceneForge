from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from PIL import Image

from Segmentation.types import SegmentDetection


@dataclass(frozen=True)
class SegmentationBackendInfo:
    name: str
    architecture: str
    input_channels: tuple[str, ...]
    primitive_labels_are_authoritative: bool = False
    legacy: bool = False
    model_path: str | None = None
    proposal_only: bool = True
    output_contract: str = "class_agnostic_instance_masks"
    primitive_label_policy: str = "geometry_fitting_downstream"
    notes: str | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "architecture": self.architecture,
            "input_channels": list(self.input_channels),
            "primitive_labels_are_authoritative": bool(self.primitive_labels_are_authoritative),
            "legacy": bool(self.legacy),
            "model_path": self.model_path,
            "proposal_only": bool(self.proposal_only),
            "output_contract": self.output_contract,
            "primitive_label_policy": self.primitive_label_policy,
            "notes": self.notes,
        }


class SegmentationBackend(Protocol):
    backend: str
    backend_info: SegmentationBackendInfo

    def detect(self, image: Image.Image) -> list[SegmentDetection]:
        raise NotImplementedError


@dataclass(frozen=True)
class LearnedSegmentationModelSpec:
    """Contract for the post-YOLO learned proposal model.

    The intended replacement detector consumes RGB/depth/camera-derived 3D point
    features and emits class-agnostic instance masks only. Primitive type
    selection remains downstream geometry/fusion work, not a detector class
    decision.
    """

    architecture: str = "primitive_3d_point_embedding_v1"
    input_channels: tuple[str, ...] = ("x", "y", "z", "r", "g", "b", "depth", "u", "v", "edge_strength")
    output_contract: str = "class_agnostic_instance_masks"
    primitive_label_policy: str = "geometry_fitting_downstream"

    def to_backend_info(self, name: str, model_path: str | None = None) -> SegmentationBackendInfo:
        return SegmentationBackendInfo(
            name=name,
            architecture=self.architecture,
            input_channels=self.input_channels,
            primitive_labels_are_authoritative=False,
            legacy=False,
            model_path=model_path,
            proposal_only=True,
            output_contract=self.output_contract,
            primitive_label_policy=self.primitive_label_policy,
            notes="Learned YOLO replacement; detector outputs class-agnostic masks, not primitive labels.",
        )
