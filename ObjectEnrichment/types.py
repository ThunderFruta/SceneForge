from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any


@dataclass(frozen=True)
class EdgeEvidence:
    status: str
    boundary_agreement: float
    edge_density: float

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "boundary_agreement": round(float(self.boundary_agreement), 6),
            "edge_density": round(float(self.edge_density), 6),
        }


@dataclass(frozen=True)
class WireframeEvidence:
    status: str = "not_available"
    line_count: int = 0
    junction_count: int = 0
    reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "line_count": int(self.line_count),
            "junction_count": int(self.junction_count),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class MeshEvidence:
    status: str
    path: str | None
    reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "path": self.path,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class GeometryEvidence:
    selected_label: str
    confidence: float
    candidate_scores: dict[str, float]
    schema_version: int = 1

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "selected_label": self.selected_label,
            "confidence": round(float(self.confidence), 6),
            "candidate_scores": {
                key: round(float(value), 6)
                for key, value in sorted(self.candidate_scores.items())
            },
        }


FUSED_LABELS = ("sphere", "cylinder", "cone", "box", "plane", "unknown")
FUSED_MODALITIES = ("detector", "depth", "edge", "wireframe", "mesh", "fusion")


@dataclass(frozen=True)
class FusedState:
    fused_label: str = "unknown"
    fused_confidence: float = 0.0
    fused_contributions: dict[str, dict[str, Any]] = field(default_factory=dict)
    needs_review: bool = False
    needs_review_reason: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "fused_label": self.fused_label,
            "fused_confidence": round(float(self.fused_confidence), 6),
            "fused_contributions": _serialize_nested(self.fused_contributions),
            "needs_review": bool(self.needs_review),
            "needs_review_reason": list(self.needs_review_reason),
        }


@dataclass(frozen=True)
class ObjectEnrichment:
    id: int
    status: str
    error: str | None
    original_detector_label: str
    detector_confidence: float
    paths: dict[str, str | None]
    edge: EdgeEvidence
    wireframe: WireframeEvidence
    mesh: MeshEvidence
    geometry: GeometryEvidence
    fused_state: FusedState = field(default_factory=FusedState)
    fused_label: str = "unknown"
    fused_confidence: float = 0.0
    fused_contributions: dict[str, dict[str, Any]] = field(default_factory=dict)
    needs_review: bool = False
    needs_review_reason: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": int(self.id),
            "status": self.status,
            "error": self.error,
            "original_detector_label": self.original_detector_label,
            "detector_confidence": round(float(self.detector_confidence), 6),
            "paths": dict(self.paths),
            "edge": self.edge.to_dict(),
            "wireframe": self.wireframe.to_dict(),
            "mesh": self.mesh.to_dict(),
            "geometry": self.geometry.to_dict(),
            "fused_state": self.fused_state.to_dict(),
            "fused_label": self.fused_label,
            "fused_confidence": round(float(self.fused_confidence), 6),
            "fused_contributions": _serialize_nested(self.fused_contributions),
            "needs_review": bool(self.needs_review),
            "needs_review_reason": list(self.needs_review_reason),
        }


def _serialize_nested(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _serialize_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_nested(item) for item in value]
    if isinstance(value, float):
        return round(float(value), 6)
    return value


@dataclass(frozen=True)
class ObjectEnrichmentReport:
    image_path: str
    depth_path: str
    detections_path: str
    model_info: dict
    objects: list[ObjectEnrichment]
    schema_version: int = 1

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "image_path": self.image_path,
            "depth_path": self.depth_path,
            "detections_path": self.detections_path,
            "model_info": dict(self.model_info),
            "objects": [item.to_dict() for item in sorted(self.objects, key=lambda item: item.id)],
        }
