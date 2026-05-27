from __future__ import annotations

from pathlib import Path

from ObjectEnrichment.geometry_scores import classify_geometry as classify_geometry_from_evidence
from ObjectEnrichment.types import GeometryEvidence


def classify_geometry(
    mask_path: str | Path,
    depth_path: str | Path,
    edge_path: str | Path,
    output_path: str | Path | None = None,
) -> GeometryEvidence:
    """Final primitive label authority for enriched detections."""
    return classify_geometry_from_evidence(mask_path, depth_path, edge_path, output_path)
