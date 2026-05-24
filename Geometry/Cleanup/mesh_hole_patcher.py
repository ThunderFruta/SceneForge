from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from Core.Types.mesh_data import Face
from Core.Types.scene_data import SceneMeshPart
from Geometry.Solidify.scan_solidifier import extract_boundary_edges


def patch_small_mesh_holes(
    part: SceneMeshPart,
    *,
    max_boundary_edges: int = 12,
) -> tuple[SceneMeshPart, int, int]:
    loops = _boundary_loops(extract_boundary_edges(part.faces))
    faces = list(part.faces)
    patched = 0
    large_gaps = 0

    for loop in loops:
        if len(loop) < 3:
            continue
        if _touches_uv_bounds(part, loop):
            if len(loop) > max_boundary_edges:
                large_gaps += 1
            continue
        if len(loop) > max_boundary_edges:
            large_gaps += 1
            continue
        center_vertex = _average_vertex([part.vertices[index] for index in loop])
        center_uv = _average_uv([part.uvs[index] for index in loop])
        center_index = len(part.vertices)
        for index, next_index in zip(loop, [*loop[1:], loop[0]]):
            faces.append((index, next_index, center_index))
        patched += 1
        part = replace(
            part,
            vertices=[*part.vertices, center_vertex],
            uvs=[*part.uvs, center_uv],
            faces=faces,
            normals=None,
        )

    if patched == 0:
        return replace(part, normals=None), 0, large_gaps
    return replace(part, faces=faces, normals=None), patched, large_gaps


def _boundary_loops(edges: list[tuple[int, int]]) -> list[list[int]]:
    adjacency: dict[int, list[int]] = defaultdict(list)
    for start, end in edges:
        adjacency[start].append(end)
        adjacency[end].append(start)

    used_edges: set[tuple[int, int]] = set()
    loops: list[list[int]] = []

    for start, end in edges:
        key = _edge_key(start, end)
        if key in used_edges:
            continue
        loop = [start]
        previous = start
        current = end
        used_edges.add(key)

        while True:
            loop.append(current)
            candidates = [
                candidate
                for candidate in sorted(adjacency[current])
                if candidate != previous and _edge_key(current, candidate) not in used_edges
            ]
            if not candidates:
                break
            next_vertex = candidates[0]
            if next_vertex == loop[0]:
                used_edges.add(_edge_key(current, next_vertex))
                break
            previous, current = current, next_vertex
            used_edges.add(_edge_key(previous, current))

        if len(loop) >= 3:
            loops.append(loop)

    return loops


def _edge_key(start: int, end: int) -> tuple[int, int]:
    return (min(start, end), max(start, end))


def _average_vertex(vertices: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    count = len(vertices)
    return (
        sum(vertex[0] for vertex in vertices) / count,
        sum(vertex[1] for vertex in vertices) / count,
        sum(vertex[2] for vertex in vertices) / count,
    )


def _average_uv(uvs: list[tuple[float, float]]) -> tuple[float, float]:
    count = len(uvs)
    return (
        sum(uv[0] for uv in uvs) / count,
        sum(uv[1] for uv in uvs) / count,
    )


def _touches_uv_bounds(part: SceneMeshPart, loop: list[int]) -> bool:
    if not part.uvs:
        return True
    min_u = min(u for u, _v in part.uvs)
    max_u = max(u for u, _v in part.uvs)
    min_v = min(v for _u, v in part.uvs)
    max_v = max(v for _u, v in part.uvs)
    eps = 1e-9
    for index in loop:
        u, v = part.uvs[index]
        if (
            abs(u - min_u) <= eps
            or abs(u - max_u) <= eps
            or abs(v - min_v) <= eps
            or abs(v - max_v) <= eps
        ):
            return True
    return False
