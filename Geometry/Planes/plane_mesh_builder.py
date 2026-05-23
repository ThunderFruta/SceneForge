from Core.Types.scene_data import SceneMeshPart
from Geometry.Planes.masked_plane_mesh_builder import build_masked_plane_part
from Geometry.Regions.region_analyzer import DepthRegion


def build_plane_part(
    region: DepthRegion,
    depth_map: list[list[float]],
    *,
    analysis_columns: int,
    analysis_rows: int,
    depth_strength: float,
    aspect_ratio: float = 1.0,
    depth_edge_threshold: float = 0.12,
) -> SceneMeshPart:
    return build_masked_plane_part(
        region,
        depth_map,
        analysis_columns=analysis_columns,
        analysis_rows=analysis_rows,
        depth_strength=depth_strength,
        aspect_ratio=aspect_ratio,
        depth_edge_threshold=depth_edge_threshold,
    )
