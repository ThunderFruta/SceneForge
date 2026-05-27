from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ObjectEnrichment.types import (
    EdgeEvidence,
    GeometryEvidence,
    MeshEvidence,
    FusedState,
    ObjectEnrichment,
    ObjectEnrichmentReport,
    WireframeEvidence,
    FUSED_LABELS,
    FUSED_MODALITIES,
)

from ObjectEnrichment.fusion import FUSION_MODALITY_WEIGHTS


def load_enrichment_report(path: str | Path) -> ObjectEnrichmentReport:
    report_path = Path(path)
    if not report_path.is_file():
        raise ValueError(f"Enrichment path does not exist or is not a file: {report_path}")
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed object_enrichment.json: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ValueError("Malformed object_enrichment.json: top-level value must be an object.")
    objects: list[ObjectEnrichment] = []
    for item in data.get("objects", []):
        edge = dict(item.get("edge", {}))
        wireframe = dict(item.get("wireframe", {}))
        mesh = dict(item.get("mesh", {}))
        geometry = dict(item.get("geometry", {}))
        fused_state_data = dict(item.get("fused_state", {}))
        fused_contributions = item.get("fused_contributions", {})
        if not isinstance(fused_contributions, dict):
            fused_contributions = {}
        fused_state_legacy = FusedState(
            fused_label=str(item.get("fused_label", "unknown")),
            fused_confidence=float(item.get("fused_confidence", 0.0)),
            fused_contributions=_coerce_fused_contributions(fused_contributions),
            needs_review=bool(item.get("needs_review", False)),
            needs_review_reason=_coerce_fused_reason(item.get("needs_review_reason", []), []),
        )
        fused_state = _coerce_fused_state(fused_state_data)
        if fused_state is None:
            fused_state = fused_state_legacy
        objects.append(
            ObjectEnrichment(
                id=int(item["id"]),
                status=str(item.get("status", "ok")),
                error=item.get("error"),
                original_detector_label=str(item.get("original_detector_label", "")),
                detector_confidence=float(item.get("detector_confidence", 0.0)),
                paths=dict(item.get("paths", {})),
                edge=EdgeEvidence(
                    status=str(edge.get("status", "not_available")),
                    boundary_agreement=float(edge.get("boundary_agreement", 0.0)),
                    edge_density=float(edge.get("edge_density", 0.0)),
                ),
                wireframe=WireframeEvidence(
                    status=str(wireframe.get("status", "not_available")),
                    line_count=int(wireframe.get("line_count", 0)),
                    junction_count=int(wireframe.get("junction_count", 0)),
                    reason=wireframe.get("reason"),
                ),
                mesh=MeshEvidence(
                    status=str(mesh.get("status", "missing")),
                    path=mesh.get("path"),
                    reason=mesh.get("reason"),
                ),
                geometry=GeometryEvidence(
                    selected_label=str(geometry.get("selected_label", "unknown")),
                    confidence=float(geometry.get("confidence", 0.0)),
                    candidate_scores={
                        str(key): float(value)
                        for key, value in dict(geometry.get("candidate_scores", {})).items()
                    },
                    schema_version=int(geometry.get("schema_version", 1)),
                ),
                fused_state=fused_state,
                fused_label=str(item.get("fused_label", "unknown")),
                fused_confidence=float(item.get("fused_confidence", 0.0)),
                fused_contributions=_coerce_fused_contributions(fused_contributions),
                needs_review=bool(item.get("needs_review", fused_state.needs_review)),
                needs_review_reason=_coerce_fused_reason(
                    item.get("needs_review_reason"),
                    fused_state.needs_review_reason,
                ),
            )
        )
    return ObjectEnrichmentReport(
        schema_version=int(data.get("schema_version", 1)),
        image_path=str(data.get("image_path", "")),
        depth_path=str(data.get("depth_path", "")),
        detections_path=str(data.get("detections_path", "")),
        model_info=dict(data.get("model_info", {})),
        objects=objects,
    )


def _coerce_fused_contributions(value: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = dict(value)
    contributions: dict[str, dict[str, Any]] = {}
    for modality in FUSED_MODALITIES:
        raw_value = raw.get(modality)
        if isinstance(raw_value, dict):
            contributions[modality] = _coerce_fused_modality_entry(dict(raw_value), modality=modality)
            continue
        contributions[modality] = _fused_state_modality_default(modality)

    for modality in ("detector", "depth", "edge", "wireframe", "mesh", "fusion"):
        if modality not in contributions:
            contributions[modality] = _fused_state_modality_default(modality)
        else:
            contributions[modality] = _coerce_fused_modality_entry(contributions[modality], modality=modality)

    return contributions


def _coerce_fused_modality_entry(value: dict[str, Any], modality: str | None = None) -> dict[str, Any]:
    entry = dict(value)
    entry.setdefault("status", "not_available")
    if "label_scores" not in entry:
        entry["label_scores"] = {label: 0.0 for label in FUSED_LABELS}
    else:
        label_scores_value = entry.get("label_scores")
        if isinstance(label_scores_value, dict):
            label_scores = dict(label_scores_value)
        else:
            label_scores = {}
        normalized_scores = {label: float(label_scores.get(label, 0.0)) for label in FUSED_LABELS}
        entry["label_scores"] = normalized_scores
        if "unknown" not in entry["label_scores"]:
            entry["label_scores"]["unknown"] = 0.0

    if modality == "fusion":
        entry.setdefault("weights", dict(FUSION_MODALITY_WEIGHTS))
        entry.setdefault("active_modalities", [])
    return entry


def _coerce_fused_state(value: dict[str, Any]) -> FusedState | None:
    if not value:
        return None
    payload = dict(value)
    fused_label = _coerce_fused_label(payload.get("fused_label", "unknown"))
    fused_confidence = float(payload.get("fused_confidence", 0.0))
    fused_confidence = max(0.0, min(1.0, fused_confidence))
    return FusedState(
        fused_label=fused_label,
        fused_confidence=fused_confidence,
        fused_contributions=_coerce_fused_contributions(payload.get("fused_contributions", {})),
        needs_review=bool(payload.get("needs_review", False)),
        needs_review_reason=_coerce_fused_reason(payload.get("needs_review_reason", []), []),
    )


def _coerce_fused_reason(value: Any, fallback: list[str]) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [value]
    return list(fallback)


def _coerce_fused_label(value: Any) -> str:
    label = str(value or "unknown").strip().lower()
    return label if label in FUSED_LABELS else "unknown"


def _fused_state_modality_default(modality: str) -> dict[str, Any]:
    if modality == "fusion":
        return {
            "status": "missing",
            "label_scores": {label: 0.0 for label in FUSED_LABELS},
            "weights": dict(FUSION_MODALITY_WEIGHTS),
            "active_modalities": [],
        }
    if modality == "mesh":
        return {
            "status": "missing",
            "selected_label": "unknown",
            "selected_score": 0.0,
            "label_scores": {label: 0.0 for label in FUSED_LABELS},
            "evidence": {"mesh_status": "missing", "has_candidate_path": False, "mesh_path": None, "reason": "missing"},
        }
    return {
        "status": "not_available",
        "selected_label": "unknown",
        "selected_score": 0.0,
        "label_scores": {label: 0.0 for label in FUSED_LABELS},
        "evidence": {},
    }
