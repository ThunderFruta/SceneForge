from __future__ import annotations

from dataclasses import replace
from math import sqrt

from Core.Types.mesh_data import Face, MeshData, Normal, Vertex
from Core.Types.scene_data import SceneMeshPart, StructuredSceneData


def build_vertex_normals(vertices: list[Vertex], faces: list[Face]) -> list[Normal]:
    accumulated = [(0.0, 0.0, 0.0) for _vertex in vertices]

    for a, b, c in faces:
        face_normal = _face_normal(vertices[a], vertices[b], vertices[c])
        if face_normal is None:
            continue
        accumulated[a] = _add(accumulated[a], face_normal)
        accumulated[b] = _add(accumulated[b], face_normal)
        accumulated[c] = _add(accumulated[c], face_normal)

    return [_normalize(normal) or (0.0, 0.0, 1.0) for normal in accumulated]


def with_mesh_normals(mesh: MeshData) -> MeshData:
    return replace(mesh, normals=build_vertex_normals(mesh.vertices, mesh.faces))


def with_scene_normals(scene: StructuredSceneData) -> StructuredSceneData:
    return StructuredSceneData(
        plane_parts=[with_part_normals(part) for part in scene.plane_parts],
        detail_parts=[with_part_normals(part) for part in scene.detail_parts],
    )


def with_part_normals(part: SceneMeshPart) -> SceneMeshPart:
    return replace(part, normals=build_vertex_normals(part.vertices, part.faces))


def _face_normal(a: Vertex, b: Vertex, c: Vertex) -> Normal | None:
    ab = _sub(b, a)
    ac = _sub(c, a)
    return _normalize(_cross(ab, ac))


def _sub(a: Vertex, b: Vertex) -> Normal:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a: Normal, b: Normal) -> Normal:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _cross(a: Normal, b: Normal) -> Normal:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _normalize(value: Normal) -> Normal | None:
    length = sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2])
    if length <= 1e-12:
        return None
    return (value[0] / length, value[1] / length, value[2] / length)
