from __future__ import annotations

from dataclasses import replace
from math import sqrt

from Core.Types.scene_data import SceneMeshPart


def filter_spike_faces(
    part: SceneMeshPart,
    *,
    threshold: str = "balanced",
) -> tuple[SceneMeshPart, int]:
    max_edge_ratio, max_y_span = _thresholds(threshold)
    kept_faces = []
    rejected = 0

    for face in part.faces:
        vertices = [part.vertices[index] for index in face]
        if _is_spike_face(vertices, max_edge_ratio=max_edge_ratio, max_y_span=max_y_span):
            rejected += 1
            continue
        kept_faces.append(face)

    if rejected == 0:
        return part, 0
    return replace(part, faces=kept_faces, normals=None), rejected


def _thresholds(threshold: str) -> tuple[float, float]:
    if threshold == "conservative":
        return 8.0, 0.30
    if threshold == "balanced":
        return 12.0, 0.45
    if threshold == "permissive":
        return 20.0, 0.70
    raise ValueError(f"Unsupported spike threshold: {threshold}")


def _is_spike_face(
    vertices: list[tuple[float, float, float]],
    *,
    max_edge_ratio: float,
    max_y_span: float,
) -> bool:
    edges = [
        _distance(vertices[0], vertices[1]),
        _distance(vertices[1], vertices[2]),
        _distance(vertices[2], vertices[0]),
    ]
    shortest = max(min(edges), 1e-12)
    if max(edges) / shortest > max_edge_ratio:
        return True

    y_values = [vertex[1] for vertex in vertices]
    return max(y_values) - min(y_values) > max_y_span


def _distance(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> float:
    return sqrt(
        (a[0] - b[0]) ** 2
        + (a[1] - b[1]) ** 2
        + (a[2] - b[2]) ** 2
    )
