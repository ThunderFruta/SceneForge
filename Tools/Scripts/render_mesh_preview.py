from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_mesh(path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    elif suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
    else:
        raise ValueError(f"Unsupported mesh preview format: {path.suffix}")


def mesh_bounds(meshes: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    min_v = Vector((1e9, 1e9, 1e9))
    max_v = Vector((-1e9, -1e9, -1e9))
    for obj in meshes:
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            min_v = Vector((min(min_v.x, world.x), min(min_v.y, world.y), min(min_v.z, world.z)))
            max_v = Vector((max(max_v.x, world.x), max(max_v.y, world.y), max(max_v.z, world.z)))
    return min_v, max_v


def frame_camera(center: Vector, radius: float, view: str) -> None:
    camera_data = bpy.data.cameras.new("PreviewCamera")
    camera = bpy.data.objects.new("PreviewCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    if view == "top":
        camera.location = center + Vector((0.0, 0.0, 3.2 * radius))
    else:
        camera.location = center + Vector((2.2 * radius, -3.2 * radius, 1.6 * radius))
    direction = center - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    camera_data.lens = 45
    bpy.context.scene.camera = camera


def add_light(center: Vector, radius: float) -> None:
    light_data = bpy.data.lights.new("PreviewKey", "AREA")
    light = bpy.data.objects.new("PreviewKey", light_data)
    bpy.context.collection.objects.link(light)
    light.location = center + Vector((0, -3 * radius, 4 * radius))
    light_data.energy = 450
    light_data.size = max(radius, 1.0)


def render_preview(input_path: Path, output_path: Path, resolution: int, view: str) -> None:
    clear_scene()
    import_mesh(input_path)
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise ValueError(f"No mesh objects imported from {input_path}")

    min_v, max_v = mesh_bounds(meshes)
    center = (min_v + max_v) / 2
    radius = max((max_v - min_v).length / 2, 0.1)
    add_light(center, radius)
    frame_camera(center, radius, view)

    engines = {item.identifier for item in bpy.context.scene.render.bl_rna.properties["engine"].enum_items}
    bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution
    bpy.context.scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a quick PNG preview for OBJ/GLB meshes.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--resolution", type=int, default=900)
    parser.add_argument("--view", choices=("orbit", "top"), default="orbit")
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(argv)
    render_preview(args.input, args.output, args.resolution, args.view)


if __name__ == "__main__":
    main()
