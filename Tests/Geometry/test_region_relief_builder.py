from __future__ import annotations

from Geometry.Mesh.region_relief_builder import build_region_relief_part
from Geometry.Regions.region_analyzer import DepthRegion


def _region() -> DepthRegion:
    return DepthRegion(
        name="detail_000",
        kind="detail",
        cells=[(0, 0), (1, 0), (0, 1), (1, 1)],
        bounds=(0, 0, 2, 2),
        average_depth=0.4,
        variance=0.0,
    )


def test_region_relief_keeps_faces_across_small_depth_changes() -> None:
    part = build_region_relief_part(
        _region(),
        [[0.2, 0.25], [0.2, 0.25]],
        analysis_columns=2,
        analysis_rows=2,
        depth_strength=1.0,
        max_steps=2,
        depth_edge_threshold=0.12,
    )

    assert part.faces == [(0, 2, 1), (1, 2, 3)]


def test_region_relief_rejects_faces_across_large_depth_jumps() -> None:
    part = build_region_relief_part(
        _region(),
        [[0.1, 0.9], [0.1, 0.9]],
        analysis_columns=2,
        analysis_rows=2,
        depth_strength=1.0,
        max_steps=2,
        depth_edge_threshold=0.12,
    )

    assert part.faces == []
