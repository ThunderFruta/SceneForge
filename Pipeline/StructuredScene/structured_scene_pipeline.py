from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
import warnings

from Core.Types.scene_data import StructuredSceneData
from Export.Blend.blend_exporter import export_blend_from_obj
from Export.OBJ.obj_exporter import ObjExportResult, export_scene_obj
from Geometry.Mesh.region_relief_builder import build_region_relief_part
from Geometry.Normals.normal_builder import with_scene_normals
from Geometry.Planes.plane_mesh_builder import build_plane_part
from Geometry.Regions.region_analyzer import analyze_depth_regions
from Geometry.Solidify.scan_solidifier import solidify_scene
from Input.Depth.depth_loader import derive_depth_from_luminance, load_depth_map
from Input.Image.image_loader import load_rgb_image
from Pipeline.ImageToMesh.image_to_mesh_pipeline import _as_blend_path


@dataclass(frozen=True)
class StructuredSceneResult:
    blend_path: Path
    preview_path: Path
    obj_result: ObjExportResult | None
    scene_data: StructuredSceneData


def run_structured_scene_pipeline(
    *,
    image_path: str | Path,
    output_path: str | Path,
    depth_path: str | Path | None = None,
    resolution: int = 64,
    depth_strength: float = 1.0,
    write_texture: bool = True,
    keep_obj: bool = False,
    blender_executable: str = "blender",
    include_details: bool = False,
    solidify: bool = True,
    solidify_thickness: float = 0.04,
    depth_edge_threshold: float = 0.12,
) -> StructuredSceneResult:
    image = load_rgb_image(image_path)
    if depth_path:
        depth_map = load_depth_map(depth_path)
    else:
        warnings.warn(
            "Structured mode is using luminance fallback depth; plane detection may be weak.",
            stacklevel=2,
        )
        depth_map = derive_depth_from_luminance(image)

    if image.size != (len(depth_map[0]), len(depth_map)):
        raise ValueError(
            "Image and depth map dimensions must match. "
            f"Image is {image.size}, depth is {(len(depth_map[0]), len(depth_map))}."
        )

    scene_data = build_structured_scene_data(
        depth_map,
        resolution=resolution,
        depth_strength=depth_strength,
        include_details=include_details,
        solidify=solidify,
        solidify_thickness=solidify_thickness,
        depth_edge_threshold=depth_edge_threshold,
    )
    blend_path = _as_blend_path(output_path)

    if keep_obj:
        obj_result = export_scene_obj(
            scene_data,
            blend_path.with_suffix(".obj"),
            texture_image=image if write_texture else None,
        )
        blend_result = export_blend_from_obj(
            obj_path=obj_result.obj_path,
            blend_path=blend_path,
            blender_executable=blender_executable,
        )
        return StructuredSceneResult(
            blend_path=blend_result.blend_path,
            preview_path=blend_result.preview_path,
            obj_result=obj_result,
            scene_data=scene_data,
        )

    with tempfile.TemporaryDirectory(prefix="sceneforge_structured_obj_") as temp_dir:
        obj_result = export_scene_obj(
            scene_data,
            Path(temp_dir) / f"{blend_path.stem}.obj",
            texture_image=image if write_texture else None,
        )
        blend_result = export_blend_from_obj(
            obj_path=obj_result.obj_path,
            blend_path=blend_path,
            blender_executable=blender_executable,
        )

    return StructuredSceneResult(
        blend_path=blend_result.blend_path,
        preview_path=blend_result.preview_path,
        obj_result=None,
        scene_data=scene_data,
    )


def build_structured_scene_data(
    depth_map: list[list[float]],
    *,
    resolution: int = 64,
    depth_strength: float = 1.0,
    include_details: bool = False,
    solidify: bool = True,
    solidify_thickness: float = 0.04,
    depth_edge_threshold: float = 0.12,
) -> StructuredSceneData:
    source_rows = len(depth_map)
    source_columns = len(depth_map[0])
    aspect_ratio = source_columns / source_rows
    analysis_columns = max(2, min(resolution, source_columns))
    analysis_rows = max(2, min(round(analysis_columns * source_rows / source_columns), source_rows))
    regions = analyze_depth_regions(
        depth_map,
        analysis_columns=analysis_columns,
        analysis_rows=analysis_rows,
    )

    plane_parts = [
        build_plane_part(
            region,
            depth_map,
            analysis_columns=analysis_columns,
            analysis_rows=analysis_rows,
            depth_strength=depth_strength,
            aspect_ratio=aspect_ratio,
            depth_edge_threshold=depth_edge_threshold,
        )
        for region in regions
        if region.kind == "plane"
    ]
    detail_parts = []
    if include_details:
        detail_parts = [
            build_region_relief_part(
                region,
                depth_map,
                analysis_columns=analysis_columns,
                analysis_rows=analysis_rows,
                depth_strength=depth_strength,
                aspect_ratio=aspect_ratio,
                depth_edge_threshold=depth_edge_threshold,
            )
            for region in regions
            if region.kind == "detail"
        ]
    scene = StructuredSceneData(plane_parts=plane_parts, detail_parts=detail_parts)
    if solidify:
        scene = solidify_scene(
            scene,
            plane_thickness=solidify_thickness,
            detail_thickness=solidify_thickness * 0.5,
        )
    return with_scene_normals(scene)
