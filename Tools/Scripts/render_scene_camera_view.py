from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_scene_asset(path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif suffix == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    else:
        raise ValueError(f"Unsupported scene format: {path.suffix}")


def load_coordinate_contract(path: Path | None) -> dict:
    if path is None:
        return {}
    report = json.loads(path.read_text(encoding="utf-8"))
    return report.get("coordinate_contract") or {}


def add_source_camera(*, fov_degrees: float, clip_end: float) -> None:
    camera_data = bpy.data.cameras.new("SceneForgeSourceCamera")
    camera = bpy.data.objects.new("SceneForgeSourceCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = (0.0, 0.0, 0.0)
    # SceneForge GLB depth is negative glTF Z. Blender's glTF importer maps
    # that depth axis into Blender +Y, so the source camera looks down +Y.
    camera.rotation_euler = Vector((0.0, 1.0, 0.0)).to_track_quat("-Z", "Y").to_euler()
    camera_data.type = "PERSP"
    camera_data.sensor_fit = "HORIZONTAL"
    camera_data.angle_x = math.radians(fov_degrees)
    camera_data.clip_start = 0.01
    camera_data.clip_end = clip_end
    bpy.context.scene.camera = camera


def add_fit_preview_camera(*, fov_degrees: float, clip_end: float) -> None:
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH" and obj.name.startswith("object_")]
    if not meshes:
        meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    bounds = object_bounds(meshes)
    center = (bounds[0] + bounds[1]) * 0.5
    extent = np.maximum(bounds[1] - bounds[0], 1e-4)
    camera_data = bpy.data.cameras.new("SceneForgeFitPreviewCamera")
    camera = bpy.data.objects.new("SceneForgeFitPreviewCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    distance = max(float(extent[1]) * 0.95, float(extent[0]) * 0.85, 0.75)
    camera.location = (
        float(center[0]),
        float(bounds[0, 1] - distance),
        float(bounds[0, 2] + extent[2] * 0.56),
    )
    target = Vector(
        (
            float(center[0]),
            float(bounds[0, 1] + extent[1] * 0.55),
            float(bounds[0, 2] + extent[2] * 0.26),
        )
    )
    direction = target - Vector(camera.location)
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    camera_data.type = "PERSP"
    camera_data.sensor_fit = "HORIZONTAL"
    camera_data.angle_x = math.radians(fov_degrees)
    camera_data.clip_start = 0.01
    camera_data.clip_end = clip_end
    bpy.context.scene.camera = camera


def add_camera_light() -> None:
    light_data = bpy.data.lights.new("SceneForgeCameraKey", "AREA")
    light = bpy.data.objects.new("SceneForgeCameraKey", light_data)
    bpy.context.collection.objects.link(light)
    light.location = (0.0, -0.25, 1.2)
    light.rotation_euler = (math.radians(65.0), 0.0, 0.0)
    light_data.energy = 180.0
    light_data.size = 5.0


def assign_debug_background_materials() -> None:
    colors = {
        "background_floor": (0.86, 0.85, 0.82, 1.0),
        "background_back_wall": (0.94, 0.93, 0.90, 1.0),
        "background_side_wall": (0.91, 0.90, 0.88, 1.0),
    }
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        prefix = next((key for key in colors if obj.name.startswith(key)), None)
        if prefix is None:
            continue
        material = bpy.data.materials.new(f"{obj.name}_debug_unlit")
        material.use_nodes = True
        nodes = material.node_tree.nodes
        nodes.clear()
        emission = nodes.new(type="ShaderNodeEmission")
        emission.inputs["Color"].default_value = colors[prefix]
        emission.inputs["Strength"].default_value = 1.0
        output = nodes.new(type="ShaderNodeOutputMaterial")
        material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
        obj.data.materials.clear()
        obj.data.materials.append(material)


def set_white_world() -> None:
    world = bpy.context.scene.world or bpy.data.worlds.new("SceneForgeWorld")
    bpy.context.scene.world = world
    world.color = (1.0, 1.0, 1.0)
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        background.inputs["Strength"].default_value = 1.0


def render_camera_view(
    *,
    input_path: Path,
    output_path: Path,
    alignment_report: Path | None,
    width: int | None,
    height: int | None,
    fov_degrees: float | None,
    camera_mode: str,
) -> None:
    clear_scene()
    import_scene_asset(input_path)
    contract = load_coordinate_contract(alignment_report)
    render_width = int(width or contract.get("image_width") or 1500)
    render_height = int(height or contract.get("image_height") or 1000)
    fov = float(fov_degrees or contract.get("fov_degrees") or 70.0)

    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise ValueError(f"No mesh objects imported from {input_path}")
    clip_end = max(mesh_distance_radius(meshes) * 4.0, 10.0)
    assign_debug_background_materials()
    add_camera_light()
    if camera_mode == "fit-preview":
        add_fit_preview_camera(fov_degrees=fov, clip_end=clip_end)
    else:
        add_source_camera(fov_degrees=fov, clip_end=clip_end)

    scene = bpy.context.scene
    engines = {item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items}
    scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
    scene.render.resolution_x = render_width
    scene.render.resolution_y = render_height
    scene.eevee.taa_render_samples = 32 if hasattr(scene, "eevee") else 16
    scene.render.film_transparent = False
    set_white_world()
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = -0.65
    scene.view_settings.gamma = 1.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)


def mesh_distance_radius(meshes: list[bpy.types.Object]) -> float:
    max_distance = 1.0
    origin = Vector((0.0, 0.0, 0.0))
    for obj in meshes:
        for corner in obj.bound_box:
            point = obj.matrix_world @ Vector(corner)
            max_distance = max(max_distance, (point - origin).length)
    return max_distance


def object_bounds(meshes: list[bpy.types.Object]) -> np.ndarray:
    points: list[Vector] = []
    for obj in meshes:
        points.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
    if not points:
        raise ValueError("Cannot compute bounds for an empty object list")
    array = np.asarray([[point.x, point.y, point.z] for point in points], dtype=np.float64)
    return np.stack([array.min(axis=0), array.max(axis=0)], axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a SceneForge scene GLB from the source camera viewpoint.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--alignment-report", type=Path)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--fov-degrees", type=float)
    parser.add_argument("--camera-mode", choices=("source", "fit-preview"), default="source")
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(argv)
    render_camera_view(
        input_path=args.input,
        output_path=args.output,
        alignment_report=args.alignment_report,
        width=args.width,
        height=args.height,
        fov_degrees=args.fov_degrees,
        camera_mode=args.camera_mode,
    )


if __name__ == "__main__":
    main()
