from __future__ import annotations

from Core.Types.scene_data import SceneMeshPart
from Geometry.DepthValidity.depth_validity import DepthValidityConfig, is_depth_valid
from Geometry.Projection.camera_projection import image_uv, project_image_depth_to_point


def build_coverage_relief_part(
    depth_map: list[list[float]],
    *,
    analysis_columns: int,
    analysis_rows: int,
    depth_strength: float,
    aspect_ratio: float = 1.0,
    depth_edge_threshold: float = 0.12,
    depth_offset: float = 0.02,
    min_valid_depth: float = 0.04,
    depth_invalid_mode: str = "black",
) -> SceneMeshPart:
    source_rows = len(depth_map)
    source_columns = len(depth_map[0])
    validity_config = DepthValidityConfig(
        min_valid_depth=min_valid_depth,
        invalid_mode=depth_invalid_mode,
    )

    vertices = []
    uvs = []
    sampled_depths = []
    for row in range(analysis_rows + 1):
        raw_v = row / analysis_rows
        source_y = _source_index(v=raw_v, count=source_rows)
        for column in range(analysis_columns + 1):
            u = column / analysis_columns
            source_x = _source_index(v=u, count=source_columns)
            depth = depth_map[source_y][source_x]
            sampled_depths.append(depth)
            x, y, z = project_image_depth_to_point(
                u,
                raw_v,
                depth,
                aspect_ratio,
                depth_strength,
            )
            vertices.append((x, y + depth_offset, z))
            uvs.append(image_uv(u, raw_v))

    faces = []
    stride = analysis_columns + 1
    for row in range(analysis_rows):
        for column in range(analysis_columns):
            top_left = row * stride + column
            top_right = top_left + 1
            bottom_left = top_left + stride
            bottom_right = bottom_left + 1
            depths = [
                sampled_depths[top_left],
                sampled_depths[top_right],
                sampled_depths[bottom_left],
                sampled_depths[bottom_right],
            ]
            if any(not is_depth_valid(depth, validity_config) for depth in depths):
                continue
            if depth_edge_threshold > 0 and max(depths) - min(depths) > depth_edge_threshold:
                continue
            faces.append((top_left, bottom_left, top_right))
            faces.append((top_right, bottom_left, bottom_right))

    return SceneMeshPart(
        name="coverage_000",
        kind="detail",
        vertices=vertices,
        faces=faces,
        uvs=uvs,
    )


def _source_index(*, v: float, count: int) -> int:
    return max(0, min(count - 1, round(v * (count - 1))))
