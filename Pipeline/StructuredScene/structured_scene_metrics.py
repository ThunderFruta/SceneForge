from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import floor, sqrt
from pathlib import Path
import json

from Core.Types.mesh_data import Face, Vertex
from Core.Types.scene_data import StructuredSceneData
from Geometry.Regions.region_analyzer import DepthRegion
from Geometry.Solidify.scan_solidifier import extract_boundary_edges


@dataclass(frozen=True)
class MeshQualitySummary:
    non_manifold_edge_count: int
    degenerate_face_count: int
    disconnected_component_count: int


@dataclass(frozen=True)
class RegionConfidenceInputs:
    region_name: str
    kind: str
    silhouette_error_proxy: float
    depth_error_proxy: float
    curvature_spike_proxy: float


def collect_region_confidence_inputs(
    regions: list[DepthRegion],
    depth_map: list[list[float]],
    *,
    analysis_columns: int,
    analysis_rows: int,
) -> list[RegionConfidenceInputs]:
    return [
        _build_region_confidence_inputs(
            region=region,
            depth_map=depth_map,
            analysis_columns=analysis_columns,
            analysis_rows=analysis_rows,
        )
        for region in regions
    ]


def _build_region_confidence_inputs(
    *,
    region: DepthRegion,
    depth_map: list[list[float]],
    analysis_columns: int,
    analysis_rows: int,
) -> RegionConfidenceInputs:
    sample_by_cell = _sample_region_cells(
        region,
        depth_map,
        analysis_columns=analysis_columns,
        analysis_rows=analysis_rows,
    )
    samples = list(sample_by_cell.values())
    if not samples:
        return RegionConfidenceInputs(
            region_name=region.name,
            kind=region.kind,
            silhouette_error_proxy=0.0,
            depth_error_proxy=0.0,
            curvature_spike_proxy=0.0,
        )

    cell_set = set(region.cells)
    boundary_edges = 0
    for column, row in cell_set:
        for neighbor in (
            (column - 1, row),
            (column + 1, row),
            (column, row - 1),
            (column, row + 1),
        ):
            if neighbor not in cell_set:
                boundary_edges += 1

    silhouette_error_proxy = boundary_edges / (len(cell_set) * 4)

    mean_depth = sum(samples) / len(samples)
    depth_error_proxy = sqrt(sum((value - mean_depth) ** 2 for value in samples) / len(samples))

    max_spike = 0.0
    for (column, row), value in sample_by_cell.items():
        for neighbor in (
            (column - 1, row),
            (column + 1, row),
            (column, row - 1),
            (column, row + 1),
        ):
            neighbor_depth = sample_by_cell.get(neighbor)
            if neighbor_depth is None:
                continue
            max_spike = max(max_spike, abs(value - neighbor_depth))

    return RegionConfidenceInputs(
        region_name=region.name,
        kind=region.kind,
        silhouette_error_proxy=silhouette_error_proxy,
        depth_error_proxy=depth_error_proxy,
        curvature_spike_proxy=max_spike,
    )


def _sample_region_cells(
    region: DepthRegion,
    depth_map: list[list[float]],
    *,
    analysis_columns: int,
    analysis_rows: int,
) -> dict[tuple[int, int], float]:
    source_rows = len(depth_map)
    source_columns = len(depth_map[0])
    samples: dict[tuple[int, int], float] = {}
    for cell in region.cells:
        column, row = cell
        y0 = floor(row * source_rows / analysis_rows)
        y1 = max(y0 + 1, floor((row + 1) * source_rows / analysis_rows))
        x0 = floor(column * source_columns / analysis_columns)
        x1 = max(x0 + 1, floor((column + 1) * source_columns / analysis_columns))
        values = [
            depth_map[source_y][source_x]
            for source_y in range(y0, min(y1, source_rows))
            for source_x in range(x0, min(x1, source_columns))
        ]
        if not values:
            continue
        samples[(column, row)] = sum(values) / len(values)
    return samples


def collect_scene_mesh_quality(scene: StructuredSceneData) -> MeshQualitySummary:
    vertices, faces = _as_flat_mesh(scene)
    return MeshQualitySummary(
        non_manifold_edge_count=_count_non_manifold_edges(faces),
        degenerate_face_count=_count_degenerate_faces(vertices, faces),
        disconnected_component_count=_count_disconnected_components(vertices, faces),
    )


def collect_boundary_edge_count(scene: StructuredSceneData) -> int:
    _, faces = _as_flat_mesh(scene)
    return len(extract_boundary_edges(faces))


def _as_flat_mesh(scene: StructuredSceneData) -> tuple[list[Vertex], list[Face]]:
    vertices: list[Vertex] = []
    faces: list[Face] = []
    offset = 0
    for part in scene.all_parts:
        vertices.extend(part.vertices)
        for start, middle, end in part.faces:
            faces.append((start + offset, middle + offset, end + offset))
        offset += len(part.vertices)
    return vertices, faces


def _count_non_manifold_edges(faces: list[Face]) -> int:
    edge_counts: dict[tuple[int, int], int] = {}
    for start, middle, end in faces:
        for edge in ((start, middle), (middle, end), (end, start)):
            a, b = sorted(edge)
            edge_counts[(a, b)] = edge_counts.get((a, b), 0) + 1
    return sum(1 for count in edge_counts.values() if count > 2)


def _count_degenerate_faces(vertices: list[Vertex], faces: list[Face]) -> int:
    return sum(1 for face in faces if _is_degenerate_face(vertices, face))


def _is_degenerate_face(vertices: list[Vertex], face: Face) -> bool:
    first, second, third = face
    x1, y1, z1 = vertices[first]
    x2, y2, z2 = vertices[second]
    x3, y3, z3 = vertices[third]
    ab = (x2 - x1, y2 - y1, z2 - z1)
    ac = (x3 - x1, y3 - y1, z3 - z1)
    cross = (
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    )
    area2 = cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]
    return area2 <= 1e-12


def _count_disconnected_components(vertices: list[Vertex], faces: list[Face]) -> int:
    if not faces:
        return 0

    adjacent: dict[int, set[int]] = {index: set() for index in range(len(vertices))}
    for start, middle, end in faces:
        adjacent[start].add(middle)
        adjacent[start].add(end)
        adjacent[middle].add(start)
        adjacent[middle].add(end)
        adjacent[end].add(start)
        adjacent[end].add(middle)

    active = {index for index, neighbors in adjacent.items() if neighbors}
    if not active:
        return 0

    components = 0
    seen: set[int] = set()
    for vertex in sorted(active):
        if vertex in seen:
            continue
        queue = deque([vertex])
        seen.add(vertex)
        while queue:
            current = queue.popleft()
            for next_vertex in adjacent[current]:
                if next_vertex not in seen:
                    seen.add(next_vertex)
                    queue.append(next_vertex)
        components += 1
    return components


def build_structured_scene_metrics_payload(
    *,
    runtime_seconds: dict[str, float],
    peak_memory_bytes: int | None,
    depth_map: list[list[float]],
    regions: list[DepthRegion],
    analysis_columns: int,
    analysis_rows: int,
    fallback_counts: dict[str, int],
    scene_before_cleanup: StructuredSceneData,
    scene_after_cleanup: StructuredSceneData,
) -> dict:
    quality = collect_scene_mesh_quality(scene_after_cleanup)
    return {
        "runtime_breakdown": runtime_seconds,
        "peak_memory_bytes": peak_memory_bytes,
        "region_confidence_inputs": [
            {
                "region_name": item.region_name,
                "kind": item.kind,
                "silhouette_error_proxy": item.silhouette_error_proxy,
                "depth_error_proxy": item.depth_error_proxy,
                "curvature_spike_proxy": item.curvature_spike_proxy,
            }
            for item in collect_region_confidence_inputs(
                regions,
                depth_map,
                analysis_columns=analysis_columns,
                analysis_rows=analysis_rows,
            )
        ],
        "fallback_counts": fallback_counts,
        "mesh_validity": {
            "non_manifold_edge_count": quality.non_manifold_edge_count,
            "degenerate_face_count": quality.degenerate_face_count,
            "disconnected_component_count": quality.disconnected_component_count,
        },
        "seam_diagnostics": {
            "boundary_edge_count_before_cleanup": collect_boundary_edge_count(scene_before_cleanup),
            "boundary_edge_count_after_cleanup": collect_boundary_edge_count(scene_after_cleanup),
        },
    }


def write_structured_scene_metrics(
    output_path: Path,
    payload: dict,
) -> None:
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


@dataclass(frozen=True)
class MemoryTracker:
    _enabled: bool = False

    def start(self) -> "MemoryTracker":
        try:
            import tracemalloc

            if not tracemalloc.is_tracing():
                tracemalloc.start(1)
                return MemoryTracker(_enabled=True)
        except Exception:
            pass
        return self

    def stop(self) -> int | None:
        if not self._enabled:
            return None
        try:
            import tracemalloc

            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return peak
        except Exception:
            return None
