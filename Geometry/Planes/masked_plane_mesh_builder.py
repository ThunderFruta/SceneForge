from __future__ import annotations

from dataclasses import dataclass
from math import floor

from Core.Types.scene_data import SceneMeshPart
from Geometry.DepthValidity.depth_validity import DepthValidityConfig, is_depth_valid
from Geometry.Planes.plane_fitter import fit_plane
from Geometry.Projection.camera_projection import (
    image_uv,
    project_image_depth_to_point,
    ray_plane_intersect,
)
from Geometry.Regions.region_analyzer import DepthRegion


_MIN_POINTS_FOR_FIT = 6


@dataclass(frozen=True)
class MaskedPlaneBuildResult:
    part: SceneMeshPart
    used_plane_fallback: bool


def build_masked_plane_part(
    region: DepthRegion,
    depth_map: list[list[float]],
    *,
    analysis_columns: int,
    analysis_rows: int,
    depth_strength: float,
    aspect_ratio: float = 1.0,
    depth_edge_threshold: float = 0.12,
    min_valid_depth: float = 0.04,
    depth_invalid_mode: str = "black",
) -> SceneMeshPart:
    return build_masked_plane_part_with_fallback(
        region,
        depth_map,
        analysis_columns=analysis_columns,
        analysis_rows=analysis_rows,
        depth_strength=depth_strength,
        aspect_ratio=aspect_ratio,
        depth_edge_threshold=depth_edge_threshold,
        min_valid_depth=min_valid_depth,
        depth_invalid_mode=depth_invalid_mode,
    ).part


def build_masked_plane_part_with_fallback(
    region: DepthRegion,
    depth_map: list[list[float]],
    *,
    analysis_columns: int,
    analysis_rows: int,
    depth_strength: float,
    aspect_ratio: float = 1.0,
    depth_edge_threshold: float = 0.12,
    min_valid_depth: float = 0.04,
    depth_invalid_mode: str = "black",
) -> MaskedPlaneBuildResult:
    del depth_edge_threshold
    validity_config = DepthValidityConfig(
        min_valid_depth=min_valid_depth,
        invalid_mode=depth_invalid_mode,
    )
    points = _unproject_cell_centers(
        region.cells,
        depth_map,
        analysis_columns,
        analysis_rows,
        aspect_ratio,
        depth_strength,
        validity_config,
    )
    fit = fit_plane(points) if len(points) >= _MIN_POINTS_FOR_FIT else None

    vertices = []
    uvs = []
    corner_indices: dict[tuple[int, int], int] = {}
    for cell_column, cell_row in sorted(region.cells, key=lambda cell: (cell[1], cell[0])):
        for corner in (
            (cell_column, cell_row),
            (cell_column + 1, cell_row),
            (cell_column, cell_row + 1),
            (cell_column + 1, cell_row + 1),
        ):
            if corner in corner_indices:
                continue
            u = corner[0] / analysis_columns
            raw_v = corner[1] / analysis_rows
            vertex = _project_corner(
                u=u,
                raw_v=raw_v,
                fit=fit,
                region=region,
                aspect_ratio=aspect_ratio,
                depth_strength=depth_strength,
            )
            corner_indices[corner] = len(vertices)
            vertices.append(vertex)
            uvs.append(image_uv(u, raw_v))

    faces = []
    for cell_column, cell_row in sorted(region.cells, key=lambda cell: (cell[1], cell[0])):
        top_left = corner_indices[(cell_column, cell_row)]
        top_right = corner_indices[(cell_column + 1, cell_row)]
        bottom_left = corner_indices[(cell_column, cell_row + 1)]
        bottom_right = corner_indices[(cell_column + 1, cell_row + 1)]
        faces.append((top_left, bottom_left, top_right))
        faces.append((top_right, bottom_left, bottom_right))

    part = SceneMeshPart(
        name=region.name,
        kind="plane",
        vertices=vertices,
        faces=faces,
        uvs=uvs,
    )
    return MaskedPlaneBuildResult(part=part, used_plane_fallback=(fit is None))


def _unproject_cell_centers(
    cells: list[tuple[int, int]],
    depth_map: list[list[float]],
    analysis_columns: int,
    analysis_rows: int,
    aspect_ratio: float,
    depth_strength: float,
    validity_config: DepthValidityConfig,
) -> list[tuple[float, float, float]]:
    source_rows = len(depth_map)
    source_cols = len(depth_map[0])
    points = []
    for col, row in cells:
        src_x = min(floor((col + 0.5) * source_cols / analysis_columns), source_cols - 1)
        src_y = min(floor((row + 0.5) * source_rows / analysis_rows), source_rows - 1)
        depth = depth_map[src_y][src_x]
        if not is_depth_valid(depth, validity_config):
            continue
        u = (col + 0.5) / analysis_columns
        raw_v = (row + 0.5) / analysis_rows
        points.append(project_image_depth_to_point(u, raw_v, depth, aspect_ratio, depth_strength))
    return points


def _project_corner(
    *,
    u: float,
    raw_v: float,
    fit: tuple[tuple[float, float, float], tuple[float, float, float]] | None,
    region: DepthRegion,
    aspect_ratio: float,
    depth_strength: float,
) -> tuple[float, float, float]:
    if fit is not None:
        centroid, normal = fit
        projected = ray_plane_intersect(u, raw_v, centroid, normal, aspect_ratio)
        if projected is not None:
            return projected
    return project_image_depth_to_point(u, raw_v, region.average_depth, aspect_ratio, depth_strength)
