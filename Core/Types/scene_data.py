from __future__ import annotations

from dataclasses import dataclass

from Core.Types.mesh_data import Face, Normal, UV, Vertex


@dataclass(frozen=True)
class SceneMeshPart:
    name: str
    kind: str
    vertices: list[Vertex]
    faces: list[Face]
    uvs: list[UV]
    normals: list[Normal] | None = None


@dataclass(frozen=True)
class StructuredSceneData:
    plane_parts: list[SceneMeshPart]
    detail_parts: list[SceneMeshPart]

    @property
    def all_parts(self) -> list[SceneMeshPart]:
        return [*self.plane_parts, *self.detail_parts]
