from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile

from Export.Blend.blend_exporter import export_blend_from_obj
from Export.OBJ.obj_exporter import ObjExportResult, export_obj
from Geometry.Mesh.grid_mesh_builder import build_grid_mesh
from Geometry.Normals.normal_builder import with_mesh_normals
from Input.Depth.depth_loader import derive_depth_from_luminance, load_depth_map
from Input.Image.image_loader import load_rgb_image


@dataclass(frozen=True)
class ImageToMeshResult:
    blend_path: Path
    preview_path: Path
    obj_result: ObjExportResult | None


def run_image_to_mesh_pipeline(
    *,
    image_path: str | Path,
    output_path: str | Path,
    depth_path: str | Path | None = None,
    resolution: int = 64,
    depth_strength: float = 1.0,
    write_texture: bool = True,
    keep_obj: bool = False,
    blender_executable: str = "blender",
) -> ImageToMeshResult:
    image = load_rgb_image(image_path)
    depth_map = load_depth_map(depth_path) if depth_path else derive_depth_from_luminance(image)

    if image.size != (len(depth_map[0]), len(depth_map)):
        raise ValueError(
            "Image and depth map dimensions must match. "
            f"Image is {image.size}, depth is {(len(depth_map[0]), len(depth_map))}."
        )

    mesh = with_mesh_normals(
        build_grid_mesh(
            depth_map,
            resolution=resolution,
            depth_strength=depth_strength,
        )
    )
    blend_path = _as_blend_path(output_path)

    if keep_obj:
        obj_result = export_obj(
            mesh,
            blend_path.with_suffix(".obj"),
            texture_image=image if write_texture else None,
        )
        blend_result = export_blend_from_obj(
            obj_path=obj_result.obj_path,
            blend_path=blend_path,
            blender_executable=blender_executable,
        )
        return ImageToMeshResult(
            blend_path=blend_result.blend_path,
            preview_path=blend_result.preview_path,
            obj_result=obj_result,
        )

    with tempfile.TemporaryDirectory(prefix="sceneforge_obj_") as temp_dir:
        obj_result = export_obj(
            mesh,
            Path(temp_dir) / f"{blend_path.stem}.obj",
            texture_image=image if write_texture else None,
        )
        blend_result = export_blend_from_obj(
            obj_path=obj_result.obj_path,
            blend_path=blend_path,
            blender_executable=blender_executable,
        )

    return ImageToMeshResult(
        blend_path=blend_result.blend_path,
        preview_path=blend_result.preview_path,
        obj_result=None,
    )


def _as_blend_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    if path.suffix.lower() != ".blend":
        return path.with_suffix(".blend")
    return path
