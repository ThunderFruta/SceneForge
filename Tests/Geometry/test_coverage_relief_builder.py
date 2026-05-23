from __future__ import annotations

from Geometry.Mesh.coverage_relief_builder import build_coverage_relief_part


def test_coverage_relief_builds_valid_depth_faces_behind_source_depth() -> None:
    part = build_coverage_relief_part(
        [[0.5, 0.5], [0.5, 0.5]],
        analysis_columns=2,
        analysis_rows=2,
        depth_strength=1.0,
        depth_offset=0.02,
    )

    assert part.name == "coverage_000"
    assert part.kind == "detail"
    assert len(part.vertices) == 9
    assert len(part.faces) == 8
    assert all(vertex[1] > 0.5 for vertex in part.vertices)


def test_coverage_relief_skips_invalid_or_large_depth_jump_cells() -> None:
    part = build_coverage_relief_part(
        [[0.5, 0.9], [0.5, 0.9]],
        analysis_columns=1,
        analysis_rows=1,
        depth_strength=1.0,
        depth_edge_threshold=0.12,
    )

    assert part.faces == []
