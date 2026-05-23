from __future__ import annotations

from Core.Types.mesh_data import MeshData
from Geometry.UV.uv_projector import build_grid_uvs


def build_grid_mesh(
    depth_map: list[list[float]],
    *,
    resolution: int | None = None,
    depth_strength: float = 1.0,
) -> MeshData:
    _validate_depth_map(depth_map)
    source_rows = len(depth_map)
    source_columns = len(depth_map[0])
    columns, rows = _target_grid_size(source_columns, source_rows, resolution)

    vertices = []
    for row in range(rows):
        source_y = _nearest_source_index(row, rows, source_rows)
        y = 0.5 - (row / (rows - 1))
        for column in range(columns):
            source_x = _nearest_source_index(column, columns, source_columns)
            x = (column / (columns - 1)) - 0.5
            z = depth_map[source_y][source_x] * depth_strength
            vertices.append((x, y, z))

    faces = []
    for row in range(rows - 1):
        for column in range(columns - 1):
            top_left = row * columns + column
            top_right = top_left + 1
            bottom_left = top_left + columns
            bottom_right = bottom_left + 1
            faces.append((top_left, bottom_left, top_right))
            faces.append((top_right, bottom_left, bottom_right))

    return MeshData(
        vertices=vertices,
        faces=faces,
        uvs=build_grid_uvs(columns, rows),
        columns=columns,
        rows=rows,
    )


def _validate_depth_map(depth_map: list[list[float]]) -> None:
    if not depth_map or not depth_map[0]:
        raise ValueError("Depth map must contain at least one pixel.")

    width = len(depth_map[0])
    for row in depth_map:
        if len(row) != width:
            raise ValueError("Depth map rows must all have the same width.")


def _target_grid_size(
    source_columns: int,
    source_rows: int,
    resolution: int | None,
) -> tuple[int, int]:
    if resolution is None:
        return source_columns, source_rows
    if resolution < 2:
        raise ValueError("Resolution must be at least 2.")

    if source_columns >= source_rows:
        columns = min(resolution, source_columns)
        rows = max(2, round(columns * source_rows / source_columns))
    else:
        rows = min(resolution, source_rows)
        columns = max(2, round(rows * source_columns / source_rows))

    return columns, rows


def _nearest_source_index(index: int, target_count: int, source_count: int) -> int:
    if target_count == 1:
        return 0
    return round(index * (source_count - 1) / (target_count - 1))

