from __future__ import annotations

from ShapeDetection.report import ObjectShapeDetection
from ObjectEnrichment.types import (
    EdgeEvidence,
    FUSED_LABELS,
    FusedState,
    GeometryEvidence,
    MeshEvidence,
    WireframeEvidence,
)


FUSION_LABELS = tuple(FUSED_LABELS)
FUSION_MODALITY_WEIGHTS = {
    "detector": 0.36,
    "depth": 0.38,
    "edge": 0.20,
    "wireframe": 0.06,
}
HIGH_CONFIDENCE_DETECTOR_THRESHOLD = 0.97
HIGH_CONFIDENCE_DETECTOR_MARGIN = 0.06
WEAK_DETECTOR_THRESHOLD = 0.75
MEDIUM_CONFIDENCE_DETECTOR_MARGIN = 0.10
WEAK_DETECTOR_DEPTH_OVERRIDE_MARGIN = 0.08
WEAK_DETECTOR_DEPTH_LABEL_MARGIN = 0.05
LOW_CONFIDENCE_PLANE_FUSION_THRESHOLD = 0.35
LOW_MARGIN_REVIEW_THRESHOLD = 0.14
DISAGREEMENT_REVIEW_CONFIDENCE_GAP = 0.05


def fuse_object_state(
    *,
    detection: ObjectShapeDetection,
    geometry: GeometryEvidence,
    edge: EdgeEvidence,
    wireframe: WireframeEvidence,
    mesh: MeshEvidence,
) -> FusedState:
    detector_state = _detector_state(detection)
    depth_state = _depth_state(geometry)
    edge_state = _edge_state(edge)
    wireframe_state = _wireframe_state(wireframe)
    mesh_state = _mesh_state(mesh)

    fused_scores = _weighted_label_scores(
        [
            ("detector", detector_state),
            ("depth", depth_state),
            ("edge", edge_state),
            ("wireframe", wireframe_state),
        ]
    )
    fused_label, fused_confidence = _pick_fused_label(fused_scores, detector_state, depth_state)
    fused_needs_review, fused_needs_review_reason = _needs_review(
        fused_label=fused_label,
        fused_confidence=fused_confidence,
        fused_scores=fused_scores,
        detector=detector_state,
        depth=depth_state,
        edge=edge_state,
        wireframe=wireframe_state,
        mesh=mesh_state,
    )

    contributions = {
        "detector": detector_state,
        "depth": depth_state,
        "edge": edge_state,
        "wireframe": wireframe_state,
        "mesh": mesh_state,
        "fusion": {
            "label_scores": fused_scores,
            "weights": dict(FUSION_MODALITY_WEIGHTS),
            "active_modalities": _active_modalities(
                detector_state,
                depth_state,
                edge_state,
                wireframe_state,
                mesh_state,
            ),
        },
    }
    return FusedState(
        fused_label=fused_label,
        fused_confidence=rounded(fused_confidence),
        fused_contributions=contributions,
        needs_review=bool(fused_needs_review),
        needs_review_reason=fused_needs_review_reason,
    )


def _detector_state(detection: ObjectShapeDetection) -> dict[str, object]:
    primitive_source = str(detection.primitive_label_source or "")
    has_assigned_primitive = (
        primitive_source != "unassigned"
        and _fused_label(detection.primitive_label) != "unknown"
    )
    if has_assigned_primitive:
        label = _fused_label(detection.primitive_label)
        confidence = max(0.0, min(1.0, float(detection.primitive_confidence)))
    else:
        label = _fused_label(detection.detector_label)
        confidence = max(0.0, min(1.0, float(detection.detector_confidence)))
    return {
        "status": "ok",
        "selected_label": label,
        "selected_score": confidence,
        "label_scores": _build_label_scores({label: confidence}),
        "evidence": {
            "detector_label": detection.detector_label,
            "detector_confidence": max(0.0, min(1.0, float(detection.detector_confidence))),
            "primitive_label": detection.primitive_label,
            "primitive_confidence": max(0.0, min(1.0, float(detection.primitive_confidence))),
            "primitive_label_source": primitive_source,
            "selected_evidence": "primitive_label" if has_assigned_primitive else "detector_label",
        },
    }


def _depth_state(geometry: GeometryEvidence) -> dict[str, object]:
    label_scores = {
        label: max(0.0, min(1.0, float(geometry.candidate_scores.get(label, 0.0))))
        for label in FUSION_LABELS
    }
    selected_label = _max_label(label_scores, list(label_scores.keys()))
    return {
        "status": "ok" if geometry.candidate_scores else "not_available",
        "selected_label": selected_label,
        "selected_score": label_scores.get(selected_label, 0.0),
        "label_scores": label_scores,
        "evidence": {
            "geometry_label": geometry.selected_label,
            "geometry_confidence": max(0.0, min(1.0, float(geometry.confidence))),
            "candidate_scores": {
                key: max(0.0, min(1.0, float(value)))
                for key, value in geometry.candidate_scores.items()
            },
        },
    }


def _edge_state(edge: EdgeEvidence) -> dict[str, object]:
    if edge.status != "ok":
        return {
            "status": edge.status,
            "selected_label": "unknown",
            "selected_score": 0.0,
            "label_scores": _build_label_scores({"unknown": 0.0}),
            "evidence": {
                "boundary_agreement": float(edge.boundary_agreement),
                "edge_density": float(edge.edge_density),
                "status": edge.status,
            },
        }

    boundary_agreement = max(0.0, min(1.0, float(edge.boundary_agreement)))
    edge_density = max(0.0, min(1.0, float(edge.edge_density)))
    label_scores = {
        "sphere": clamp(0.55 * (1.0 - edge_density)),
        "cylinder": clamp(0.30 * (1.0 - boundary_agreement) + 0.55 * edge_density),
        "cone": clamp(0.30 * (1.0 - boundary_agreement) + 0.55 * edge_density),
        "box": clamp(0.45 * boundary_agreement + 0.55 * edge_density),
        "plane": clamp(0.75 * boundary_agreement + 0.20 * (1.0 - edge_density)),
        "unknown": 0.08,
    }
    selected_label = _max_label(label_scores, list(label_scores.keys()))
    return {
        "status": "ok",
        "selected_label": selected_label,
        "selected_score": label_scores[selected_label],
        "label_scores": label_scores,
        "evidence": {
            "boundary_agreement": boundary_agreement,
            "edge_density": edge_density,
            "status": edge.status,
        },
    }


def _wireframe_state(wireframe: WireframeEvidence) -> dict[str, object]:
    if wireframe.status != "ok":
        return {
            "status": wireframe.status,
            "selected_label": "unknown",
            "selected_score": 0.0,
            "label_scores": _build_label_scores({"unknown": 0.0}),
            "evidence": {
                "status": wireframe.status,
                "line_count": int(wireframe.line_count),
                "junction_count": int(wireframe.junction_count),
                "reason": wireframe.reason,
            },
        }

    line_factor = min(1.0, max(0.0, float(wireframe.line_count)) / 12.0)
    junction_factor = min(1.0, max(0.0, float(wireframe.junction_count)) / 10.0)
    geometry = 0.2 + 0.75 * min(1.0, line_factor + 0.35 * junction_factor)
    label_scores = {
        "sphere": clamp((1.0 - geometry) * 0.7),
        "cylinder": clamp(0.30 + 0.55 * geometry),
        "cone": clamp(0.20 + 0.45 * geometry),
        "box": clamp(0.35 + 0.55 * geometry),
        "plane": clamp(0.40 + 0.30 * geometry),
        "unknown": 0.06,
    }
    selected_label = _max_label(label_scores, list(label_scores.keys()))
    return {
        "status": "ok",
        "selected_label": selected_label,
        "selected_score": label_scores[selected_label],
        "label_scores": label_scores,
        "evidence": {
            "line_count": int(wireframe.line_count),
            "junction_count": int(wireframe.junction_count),
            "status": wireframe.status,
            "line_factor": line_factor,
            "junction_factor": junction_factor,
        },
    }


def _mesh_state(mesh: MeshEvidence) -> dict[str, object]:
    return {
        "status": mesh.status,
        "selected_label": "unknown",
        "selected_score": 0.0,
        "label_scores": _build_label_scores({}),
        "evidence": {
            "mesh_status": mesh.status,
            "has_candidate_path": mesh.path is not None,
            "mesh_path": mesh.path,
            "reason": mesh.reason,
        },
    }


def _weighted_label_scores(entries: list[tuple[str, dict]]) -> dict[str, float]:
    scores = _build_label_scores({})
    for name, state in entries:
        weight = FUSION_MODALITY_WEIGHTS.get(name, 0.0)
        label_scores = state.get("label_scores", {})
        if not isinstance(label_scores, dict):
            continue
        for label in FUSION_LABELS:
            score = float(label_scores.get(label, 0.0))
            scores[label] = round(scores[label] + weight * score, 9)
    return scores


def _pick_fused_label(scores: dict[str, float], detector: dict[str, object], depth: dict[str, object]) -> tuple[str, float]:
    if not scores:
        return "unknown", 0.0
    best_label, best_score = max(
        scores.items(),
        key=lambda item: (item[1], item[0]),
    )
    detector_label = str(detector.get("selected_label", "unknown"))
    detector_confidence = float(detector.get("selected_score", 0.0))
    detector_score = float(scores.get(detector_label, 0.0))
    depth_label = str(depth.get("selected_label", "unknown"))
    depth_confidence = float(depth.get("selected_score", 0.0))
    depth_scores = depth.get("label_scores", {})
    depth_values = sorted(
        (
            float(depth_scores.get(label, 0.0))
            for label in FUSION_LABELS
            if label != "unknown"
        ),
        reverse=True,
    ) if isinstance(depth_scores, dict) else []
    depth_margin = depth_values[0] - depth_values[1] if len(depth_values) > 1 else depth_confidence
    if (
        detector_label == "plane"
        and detector_confidence < LOW_CONFIDENCE_PLANE_FUSION_THRESHOLD
        and depth_label != "unknown"
        and depth_label != "plane"
        and depth_margin >= WEAK_DETECTOR_DEPTH_LABEL_MARGIN
    ):
        return depth_label, float(scores.get(depth_label, 0.0))
    if (
        detector_label != "unknown"
        and depth_label != "unknown"
        and detector_confidence < WEAK_DETECTOR_THRESHOLD
        and depth_label != detector_label
        and depth_confidence >= detector_confidence + WEAK_DETECTOR_DEPTH_OVERRIDE_MARGIN
        and depth_margin >= WEAK_DETECTOR_DEPTH_LABEL_MARGIN
    ):
        return depth_label, float(scores.get(depth_label, 0.0))
    if (
        detector_label != "unknown"
        and detector_confidence >= 0.20
        and detector_confidence < WEAK_DETECTOR_THRESHOLD
        and (detector_score >= best_score * 0.60 or best_score < 0.25)
    ):
        return detector_label, detector_score
    if (
        detector_label != "unknown"
        and detector_confidence >= WEAK_DETECTOR_THRESHOLD
        and best_score - detector_score <= MEDIUM_CONFIDENCE_DETECTOR_MARGIN
    ):
        return detector_label, detector_score
    if (
        detector_label != "unknown"
        and detector_confidence >= HIGH_CONFIDENCE_DETECTOR_THRESHOLD
        and best_score - detector_score <= HIGH_CONFIDENCE_DETECTOR_MARGIN
    ):
        return detector_label, detector_score
    return best_label, best_score


def _needs_review(
    *,
    fused_label: str,
    fused_confidence: float,
    fused_scores: dict[str, float],
    detector: dict[str, object],
    depth: dict[str, object],
    edge: dict[str, object],
    wireframe: dict[str, object],
    mesh: dict[str, object],
) -> tuple[bool, list[str]]:
    values = sorted(fused_scores.values(), reverse=True)
    top = values[0] if values else 0.0
    second = values[1] if len(values) > 1 else 0.0
    margin = top - second
    active_modalities = [
        "detector",
        "depth",
        "edge",
        "wireframe",
        "mesh",
    ]
    active = 0
    for key, item in zip(active_modalities, (detector, depth, edge, wireframe, mesh)):
        del key
        if item.get("status") == "ok":
            active += 1
    low_support = top < 0.30
    uncertain = margin < 0.10
    low_margin = margin < LOW_MARGIN_REVIEW_THRESHOLD
    mesh_warning = mesh.get("status") == "failed" and active >= 3
    geometry_shift = _max_label(depth.get("label_scores", {})) != _max_label(detector.get("label_scores", {}))
    low_geometry = float(detector.get("selected_score", 0.0)) < 0.15 and active <= 2
    detector_label = str(detector.get("selected_label", "unknown"))
    detector_confidence = float(detector.get("selected_score", 0.0))
    depth_label = str(depth.get("selected_label", "unknown"))
    depth_confidence = float(depth.get("selected_score", 0.0))
    strong_detector_disagreement = (
        detector_label != "unknown"
        and detector_confidence >= HIGH_CONFIDENCE_DETECTOR_THRESHOLD
        and fused_label != detector_label
    )

    reasons: list[str] = []
    if not fused_scores:
        reasons.append("no_fusion_scores")
    if low_support:
        reasons.append("low_support")
    if uncertain:
        reasons.append("uncertain_modality_margin")
    if low_margin:
        reasons.append("low_fusion_margin")
    if mesh_warning:
        reasons.append("mesh_status_failed")
    if geometry_shift and low_support:
        reasons.append("detector_depth_disagreement")
    if strong_detector_disagreement:
        reasons.append("high_confidence_detector_disagreement")
    if low_geometry:
        reasons.append("low_detector_signal")
    if (
        detector_label != "unknown"
        and depth_label != "unknown"
        and detector_label != depth_label
        and abs(detector_confidence - depth_confidence) <= DISAGREEMENT_REVIEW_CONFIDENCE_GAP
    ):
        reasons.append("detector_depth_close_disagreement")
    if (
        detector_label in {"plane", "box", "sphere"}
        and detector_label != fused_label
        and detector_confidence >= 0.35
    ):
        reasons.append("review_label_flip")
    if fused_confidence < 0.30:
        reasons.append("low_fused_confidence")

    return bool(reasons), reasons


def _active_modalities(
    detector: dict[str, object],
    depth: dict[str, object],
    edge: dict[str, object],
    wireframe: dict[str, object],
    mesh: dict[str, object],
) -> list[str]:
    return [
        key
        for key, item in (
            ("detector", detector),
            ("depth", depth),
            ("edge", edge),
            ("wireframe", wireframe),
            ("mesh", mesh),
        )
        if item.get("status") == "ok"
    ]


def _max_label(scores: dict[str, float], candidates: list[str] | None = None) -> str:
    if not scores:
        return "unknown"
    if candidates is None:
        candidates = FUSION_LABELS
    candidate_scores = {
        label: max(0.0, min(1.0, float(scores.get(label, 0.0)))) for label in candidates
    }
    return max(candidate_scores.items(), key=lambda item: (item[1], item[0]))[0]


def _build_label_scores(base: dict[str, float]) -> dict[str, float]:
    scores = {label: 0.0 for label in FUSION_LABELS}
    for label, value in base.items():
        scores[_fused_label(label)] = clamp(float(value))
    return scores


def _fused_label(value: str | None) -> str:
    candidate = str(value or "unknown").lower()
    return candidate if candidate in FUSION_LABELS else "unknown"


def clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def rounded(value: float) -> float:
    return round(clamp(float(value)), 6)
