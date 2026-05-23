from __future__ import annotations

from Core.Types.scene_data import SceneMeshPart
from Geometry.Normals.normal_builder import with_part_normals
from Geometry.Solidify.scan_solidifier import extract_boundary_edges, solidify_part


def test_extract_boundary_edges_for_rectangle() -> None:
    edges = extract_boundary_edges([(0, 2, 1), (1, 2, 3)])

    assert {tuple(sorted(edge)) for edge in edges} == {
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
    }


def test_extract_boundary_edges_preserves_l_shape_concavity() -> None:
    faces = [
        (0, 3, 1), (1, 3, 4),
        (1, 4, 2), (2, 4, 5),
        (3, 6, 4), (4, 6, 7),
    ]

    edges = extract_boundary_edges(faces)

    assert len(edges) == 8
    assert (4, 5) in {tuple(sorted(edge)) for edge in edges}
    assert (4, 7) in {tuple(sorted(edge)) for edge in edges}


def test_solidify_part_adds_back_vertices_and_side_faces() -> None:
    part = SceneMeshPart(
        name="plane_000",
        kind="plane",
        vertices=[(0.0, 1.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 1.0), (1.0, 1.0, 1.0)],
        faces=[(0, 2, 1), (1, 2, 3)],
        uvs=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
    )

    solid = solidify_part(part, thickness=0.25)

    assert solid.faces[:2] == part.faces
    assert len(solid.vertices) == 8
    assert len(solid.uvs) == 8
    assert len(solid.faces) == 10
    assert solid.vertices[4:] == [
        (0.0, 1.25, 0.0),
        (1.0, 1.25, 0.0),
        (0.0, 1.25, 1.0),
        (1.0, 1.25, 1.0),
    ]


def test_solidified_part_can_recompute_normals_for_all_vertices() -> None:
    part = SceneMeshPart(
        name="plane_000",
        kind="plane",
        vertices=[(0.0, 1.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 1.0), (1.0, 1.0, 1.0)],
        faces=[(0, 2, 1), (1, 2, 3)],
        uvs=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
    )

    solid = with_part_normals(solidify_part(part, thickness=0.25))

    assert solid.normals is not None
    assert len(solid.normals) == len(solid.vertices)
