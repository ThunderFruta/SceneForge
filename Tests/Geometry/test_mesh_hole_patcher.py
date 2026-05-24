from __future__ import annotations

from Core.Types.scene_data import SceneMeshPart
from Geometry.Cleanup.mesh_hole_patcher import patch_small_mesh_holes


def _grid_part_with_center_hole() -> SceneMeshPart:
    vertices = [
        (float(column), 1.0, float(row))
        for row in range(4)
        for column in range(4)
    ]
    uvs = [
        (column / 3.0, row / 3.0)
        for row in range(4)
        for column in range(4)
    ]
    faces = []
    for row in range(3):
        for column in range(3):
            if (column, row) == (1, 1):
                continue
            top_left = row * 4 + column
            top_right = top_left + 1
            bottom_left = top_left + 4
            bottom_right = bottom_left + 1
            faces.extend(
                [
                    (top_left, bottom_left, top_right),
                    (top_right, bottom_left, bottom_right),
                ]
            )
    return SceneMeshPart(
        name="plane_000",
        kind="plane",
        vertices=vertices,
        faces=faces,
        uvs=uvs,
    )


def test_patch_small_mesh_holes_caps_internal_loop() -> None:
    part = _grid_part_with_center_hole()

    patched, patched_count, large_gap_count = patch_small_mesh_holes(
        part,
        max_boundary_edges=4,
    )

    assert patched_count == 1
    assert large_gap_count >= 1
    assert len(patched.vertices) == len(part.vertices) + 1
    assert len(patched.uvs) == len(part.uvs) + 1
    assert len(patched.faces) == len(part.faces) + 4
    assert patched.normals is None


def test_patch_small_mesh_holes_does_not_cap_large_exterior_silhouette() -> None:
    part = SceneMeshPart(
        name="plane_000",
        kind="plane",
        vertices=[
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
        ],
        faces=[(0, 2, 1), (1, 2, 3)],
        uvs=[
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
        ],
    )

    patched, patched_count, large_gap_count = patch_small_mesh_holes(
        part,
        max_boundary_edges=3,
    )

    assert patched_count == 0
    assert large_gap_count == 1
    assert patched.vertices == part.vertices
    assert patched.faces == part.faces
