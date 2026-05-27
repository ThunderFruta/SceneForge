from __future__ import annotations

from ObjectEnrichment.types import (
    EdgeEvidence,
    GeometryEvidence,
    MeshEvidence,
    ObjectEnrichment,
    FusedState,
    WireframeEvidence,
)
from PrimitiveFitting.pipeline import apply_enrichment_fit_metadata
from PrimitiveFitting.types import PrimitiveFit


def test_fit_metadata_records_geometry_and_mesh_audit_fields() -> None:
    fit = PrimitiveFit(
        id=1,
        primitive_label="unknown",
        confidence=0.0,
        center_xyz=(0.0, 1.0, 0.0),
        rotation_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dimensions_xyz=(1.0, 1.0, 1.0),
        fit_quality={"selected_fit_mode": "fallback"},
    )
    enrichment = ObjectEnrichment(
        id=1,
        status="ok",
        error=None,
        original_detector_label="cone",
        detector_confidence=0.8,
        paths={},
        edge=EdgeEvidence(status="ok", boundary_agreement=0.75, edge_density=0.2),
        wireframe=WireframeEvidence(),
        mesh=MeshEvidence(status="ok", path="objects/01/mesh_candidate.obj"),
        geometry=GeometryEvidence(selected_label="cylinder", confidence=0.9, candidate_scores={"cylinder": 0.9}),
        fused_state=FusedState(
            fused_label="sphere",
            fused_confidence=0.87,
            fused_contributions={"detector": {"status": "ok"}},
            needs_review=False,
            needs_review_reason=[],
        ),
        fused_label="sphere",
        fused_confidence=0.87,
        fused_contributions={"detector": {"status": "ok"}},
        needs_review=False,
        needs_review_reason=[],
    )

    updated = apply_enrichment_fit_metadata(fit, enrichment)

    assert updated.primitive_label == "sphere"
    assert updated.primitive_label_source == "fused"
    assert updated.confidence == 0.87
    assert updated.fit_quality["schema_version"] == 2
    assert updated.fit_quality["original_detector_label"] == "cone"
    assert updated.fit_quality["geometry_selected_label"] == "cylinder"
    assert updated.fit_quality["fused_label"] == "sphere"
    assert updated.fit_quality["fused_confidence"] == 0.87
    assert updated.fit_quality["needs_review_reason"] == []
    assert updated.fit_quality["mesh_candidate_path"] == "objects/01/mesh_candidate.obj"
