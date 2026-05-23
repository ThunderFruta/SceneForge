from __future__ import annotations

from Core.Types.mesh_data import UV


def build_grid_uvs(columns: int, rows: int) -> list[UV]:
    if columns < 2 or rows < 2:
        raise ValueError("UV grid must be at least 2 by 2.")

    uvs = []
    for row in range(rows):
        v = 1.0 - (row / (rows - 1))
        for column in range(columns):
            u = column / (columns - 1)
            uvs.append((u, v))
    return uvs

