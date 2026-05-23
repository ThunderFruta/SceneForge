from __future__ import annotations

from dataclasses import dataclass


Vertex = tuple[float, float, float]
Normal = tuple[float, float, float]
UV = tuple[float, float]
Face = tuple[int, int, int]


@dataclass(frozen=True)
class MeshData:
    vertices: list[Vertex]
    faces: list[Face]
    uvs: list[UV]
    columns: int
    rows: int
    normals: list[Normal] | None = None
