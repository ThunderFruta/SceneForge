from __future__ import annotations

from Geometry.Mesh.grid_mesh_builder import build_grid_mesh


def test_build_grid_mesh_creates_vertices_faces_and_uvs() -> None:
    mesh = build_grid_mesh(
        [
            [0.0, 0.5],
            [0.75, 1.0],
        ],
        resolution=2,
        depth_strength=2.0,
    )

    assert mesh.columns == 2
    assert mesh.rows == 2
    assert mesh.vertices == [
        (-0.5, 0.5, 0.0),
        (0.5, 0.5, 1.0),
        (-0.5, -0.5, 1.5),
        (0.5, -0.5, 2.0),
    ]
    assert mesh.faces == [(0, 2, 1), (1, 2, 3)]
    assert mesh.uvs == [(0.0, 1.0), (1.0, 1.0), (0.0, 0.0), (1.0, 0.0)]

