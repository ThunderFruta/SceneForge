from __future__ import annotations

from Core.Types.scene_data import SceneMeshPart
from Geometry.Projection.camera_projection import image_uv, project_image_depth_to_point
from Geometry.Regions.region_analyzer import DepthRegion


def build_region_relief_part(
    region: DepthRegion,
    depth_map: list[list[float]],
    *,
    analysis_columns: int,
    analysis_rows: int,
    depth_strength: float,
    aspect_ratio: float = 1.0,
    max_steps: int = 18,
    depth_edge_threshold: float = 0.12,
) -> SceneMeshPart:
    min_column, min_row, max_column, max_row = region.bounds
    source_rows = len(depth_map)
    source_columns = len(depth_map[0])
    region_columns = max(2, min(max_column - min_column + 1, max_steps))
    region_rows = max(2, min(max_row - min_row + 1, max_steps))

    vertices = []
    uvs = []
    sampled_depths = []
    for row in range(region_rows):
        row_t = row / (region_rows - 1)
        analysis_y = min_row + (max_row - min_row) * row_t
        raw_v = analysis_y / analysis_rows
        source_y = _source_index(v=raw_v, count=source_rows)
        for column in range(region_columns):
            column_t = column / (region_columns - 1)
            analysis_x = min_column + (max_column - min_column) * column_t
            u = analysis_x / analysis_columns
            source_x = _source_index(v=u, count=source_columns)
            depth = depth_map[source_y][source_x]
            sampled_depths.append(depth)
            vertices.append(
                project_image_depth_to_point(u, raw_v, depth, aspect_ratio, depth_strength)
            )
            uvs.append(image_uv(u, raw_v))

    faces = []
    for row in range(region_rows - 1):
        for column in range(region_columns - 1):
            top_left = row * region_columns + column
            top_right = top_left + 1
            bottom_left = top_left + region_columns
            bottom_right = bottom_left + 1
            depths = [
                sampled_depths[top_left],
                sampled_depths[top_right],
                sampled_depths[bottom_left],
                sampled_depths[bottom_right],
            ]
            if depth_edge_threshold > 0 and max(depths) - min(depths) > depth_edge_threshold:
                continue
            faces.append((top_left, bottom_left, top_right))
            faces.append((top_right, bottom_left, bottom_right))

    return SceneMeshPart(
        name=region.name,
        kind="detail",
        vertices=vertices,
        faces=faces,
        uvs=uvs,
    )


def _source_index(*, v: float, count: int) -> int:
    return max(0, min(count - 1, round(v * (count - 1))))
