from __future__ import annotations

from Geometry.Planes.plane_mesh_builder import build_plane_part
from Geometry.Regions.region_analyzer import DepthRegion


def _flat_depth(rows: int, cols: int, value: float = 0.5) -> list[list[float]]:
    return [[value] * cols for _ in range(rows)]


def _region(
    cells: list[tuple[int, int]],
    *,
    average_depth: float = 0.5,
    variance: float = 0.0,
) -> DepthRegion:
    return DepthRegion(
        name="plane_000",
        kind="plane",
        cells=cells,
        bounds=(
            min(column for column, _row in cells),
            min(row for _column, row in cells),
            max(column for column, _row in cells) + 1,
            max(row for _column, row in cells) + 1,
        ),
        average_depth=average_depth,
        variance=variance,
    )


def test_build_plane_part_rectangular_region_uses_cell_mesh() -> None:
    region = _region([(1, 1), (2, 1), (1, 2), (2, 2)])

    part = build_plane_part(
        region,
        _flat_depth(4, 4),
        analysis_columns=4,
        analysis_rows=4,
        depth_strength=2.0,
        aspect_ratio=1.0,
    )

    assert part.name == "plane_000"
    assert part.kind == "plane"
    assert len(part.vertices) == 9
    assert len(part.faces) == 8
    assert (-0.5, 2.0, 0.5) in part.vertices
    assert (0.5, 2.0, -0.5) in part.vertices
    assert (0.25, 0.75) in part.uvs
    assert (0.75, 0.25) in part.uvs


def test_build_plane_part_l_shaped_region_does_not_fill_bounding_box() -> None:
    region = _region([(0, 0), (1, 0), (0, 1)])

    part = build_plane_part(
        region,
        _flat_depth(2, 2),
        analysis_columns=2,
        analysis_rows=2,
        depth_strength=1.0,
        aspect_ratio=1.0,
    )

    assert len(part.vertices) == 8
    assert len(part.faces) == 6


def test_build_plane_part_region_with_hole_does_not_fill_hole_cell() -> None:
    cells = [
        (column, row)
        for row in range(3)
        for column in range(3)
        if (column, row) != (1, 1)
    ]
    region = _region(cells)

    part = build_plane_part(
        region,
        _flat_depth(3, 3),
        analysis_columns=3,
        analysis_rows=3,
        depth_strength=1.0,
        aspect_ratio=1.0,
    )

    assert len(part.faces) == 16
    assert len(part.vertices) == 16


def test_build_plane_part_keeps_region_faces_across_depth_edges() -> None:
    region = _region([(0, 0), (1, 0), (0, 1), (1, 1)])
    depth_map = [[0.1, 0.9], [0.1, 0.9]]

    part = build_plane_part(
        region,
        depth_map,
        analysis_columns=2,
        analysis_rows=2,
        depth_strength=1.0,
        aspect_ratio=1.0,
        depth_edge_threshold=0.12,
    )

    assert len(part.faces) == 8


def test_build_plane_part_uvs_stay_normalized() -> None:
    region = _region([(0, 0), (1, 0), (0, 1)])

    part = build_plane_part(
        region,
        _flat_depth(2, 2),
        analysis_columns=2,
        analysis_rows=2,
        depth_strength=1.0,
        aspect_ratio=1.0,
    )

    assert all(0.0 <= u <= 1.0 and 0.0 <= v <= 1.0 for u, v in part.uvs)
    assert (0.0, 1.0) in part.uvs
    assert (1.0, 0.5) in part.uvs
    assert (0.5, 0.0) in part.uvs


def test_build_plane_part_tilted_floor_vertices_lie_on_fitted_plane() -> None:
    rows, cols = 8, 8
    depth_map = [
        [0.3 + row * 0.05 for _col in range(cols)]
        for row in range(rows)
    ]
    cells = [(column, row) for row in range(2, 6) for column in range(0, 8)]
    region = _region(cells, average_depth=0.5, variance=0.001)

    part = build_plane_part(
        region,
        depth_map,
        analysis_columns=8,
        analysis_rows=8,
        depth_strength=1.0,
        aspect_ratio=1.0,
    )

    a, b, c = _first_non_collinear_triplet(part.vertices)
    normal = _cross(_sub(b, a), _sub(c, a))
    for vertex in part.vertices:
        assert abs(_dot(normal, _sub(vertex, a))) < 1e-8


def test_build_plane_part_falls_back_gracefully_with_no_valid_depth() -> None:
    region = _region([(0, 0), (1, 0), (0, 1), (1, 1)])
    depth_map = [[0.01] * 4 for _ in range(4)]

    part = build_plane_part(
        region,
        depth_map,
        analysis_columns=4,
        analysis_rows=4,
        depth_strength=1.0,
        aspect_ratio=1.0,
    )

    assert part.name == "plane_000"
    assert len(part.vertices) == 9
    assert len(part.faces) == 8


def _first_non_collinear_triplet(
    vertices: list[tuple[float, float, float]],
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    for index_a, a in enumerate(vertices):
        for index_b, b in enumerate(vertices[index_a + 1 :], start=index_a + 1):
            for c in vertices[index_b + 1 :]:
                normal = _cross(_sub(b, a), _sub(c, a))
                if _dot(normal, normal) > 1e-12:
                    return a, b, c
    raise AssertionError("Expected at least one non-collinear vertex triplet.")


def _sub(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
