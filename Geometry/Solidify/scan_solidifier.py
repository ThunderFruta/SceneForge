from __future__ import annotations

from dataclasses import replace

from Core.Types.mesh_data import Face
from Core.Types.scene_data import SceneMeshPart, StructuredSceneData


def extract_boundary_edges(faces: list[Face]) -> list[tuple[int, int]]:
    edge_counts: dict[tuple[int, int], int] = {}
    edge_directions: dict[tuple[int, int], tuple[int, int]] = {}

    for face in faces:
        for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = (min(start, end), max(start, end))
            edge_counts[key] = edge_counts.get(key, 0) + 1
            edge_directions.setdefault(key, (start, end))

    return [
        edge_directions[key]
        for key in sorted(edge_counts)
        if edge_counts[key] == 1
    ]


def solidify_scene(
    scene: StructuredSceneData,
    *,
    plane_thickness: float,
    detail_thickness: float,
) -> StructuredSceneData:
    return StructuredSceneData(
        plane_parts=[
            solidify_part(part, thickness=plane_thickness)
            for part in scene.plane_parts
        ],
        detail_parts=[
            solidify_part(part, thickness=detail_thickness)
            for part in scene.detail_parts
        ],
    )


def solidify_part(part: SceneMeshPart, *, thickness: float) -> SceneMeshPart:
    if thickness <= 0.0 or not part.vertices or not part.faces:
        return replace(part, normals=None)

    vertex_count = len(part.vertices)
    back_vertices = [
        (x, y + thickness, z)
        for x, y, z in part.vertices
    ]
    vertices = [*part.vertices, *back_vertices]
    uvs = [*part.uvs, *part.uvs]
    faces = list(part.faces)

    for start, end in extract_boundary_edges(part.faces):
        back_start = start + vertex_count
        back_end = end + vertex_count
        faces.append((start, end, back_start))
        faces.append((end, back_end, back_start))

    return SceneMeshPart(
        name=part.name,
        kind=part.kind,
        vertices=vertices,
        faces=faces,
        uvs=uvs,
        normals=None,
    )
