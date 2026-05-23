from __future__ import annotations

from Geometry.Normals.normal_builder import build_vertex_normals


def test_build_vertex_normals_for_flat_xy_plane() -> None:
    normals = build_vertex_normals(
        vertices=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
        ],
        faces=[(0, 1, 2), (1, 3, 2)],
    )

    assert normals == [
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
    ]


def test_build_vertex_normals_skips_degenerate_faces() -> None:
    normals = build_vertex_normals(
        vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)],
        faces=[(0, 1, 2)],
    )

    assert normals == [
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
    ]
