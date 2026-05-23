from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile


@dataclass(frozen=True)
class BlendExportResult:
    blend_path: Path
    preview_path: Path


def export_blend_from_obj(
    *,
    obj_path: str | Path,
    blend_path: str | Path,
    blender_executable: str = "blender",
    preview_path: str | Path | None = None,
    scene_scale: float = 4.0,
) -> BlendExportResult:
    executable = shutil.which(blender_executable)
    if executable is None:
        raise RuntimeError(
            "Blender executable was not found. Install Blender or pass "
            "`--blender /path/to/blender`."
        )

    source_obj = Path(obj_path).resolve()
    output_blend = Path(blend_path)
    output_preview = Path(preview_path) if preview_path is not None else output_blend.with_name("preview.png")
    output_blend.parent.mkdir(parents=True, exist_ok=True)
    output_preview.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix="_sceneforge_import.py",
        encoding="utf-8",
        delete=False,
    ) as script_file:
        script_path = Path(script_file.name)
        script_file.write(
            _build_blender_import_script(
                source_obj,
                output_blend.resolve(),
                output_preview.resolve(),
                scene_scale,
            )
        )

    try:
        completed = subprocess.run(
            [
                executable,
                "--background",
                "--python",
                str(script_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    finally:
        script_path.unlink(missing_ok=True)

    if completed.returncode != 0:
        raise RuntimeError(
            "Blender failed to create the .blend file.\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    if not output_blend.exists() or not output_preview.exists():
        raise RuntimeError(
            "Blender did not write the expected output files.\n"
            f"Expected blend: {output_blend}\n"
            f"Expected preview: {output_preview}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    return BlendExportResult(blend_path=output_blend, preview_path=output_preview)


def _build_blender_import_script(
    obj_path: Path,
    blend_path: Path,
    preview_path: Path,
    scene_scale: float = 4.0,
) -> str:
    return f"""
from pathlib import Path
import math
import bpy
from mathutils import Vector

obj_path = Path({str(obj_path)!r})
blend_path = Path({str(blend_path)!r})
preview_path = Path({str(preview_path)!r})
scene_scale = {float(scene_scale)!r}

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

if hasattr(bpy.ops.wm, "obj_import"):
    bpy.ops.wm.obj_import(
        filepath=str(obj_path),
        forward_axis="Y",
        up_axis="Z",
        global_scale=scene_scale,
    )
else:
    bpy.ops.import_scene.obj(
        filepath=str(obj_path),
        axis_forward="Y",
        axis_up="Z",
        global_scale=scene_scale,
    )

for obj in bpy.context.scene.objects:
    obj.select_set(True)

for material in bpy.data.materials:
    texture_image = None
    if material.use_nodes and material.node_tree:
        for node in material.node_tree.nodes:
            if getattr(node, "image", None) is not None:
                texture_image = node.image
                break
    material.use_nodes = True
    material.blend_method = "OPAQUE"
    material.use_screen_refraction = False
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1.0
    if texture_image is not None:
        image_node = nodes.new(type="ShaderNodeTexImage")
        image_node.image = texture_image
        material.node_tree.links.new(image_node.outputs["Color"], emission.inputs["Color"])
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])

mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
if mesh_objects:
    min_corner = Vector((float("inf"), float("inf"), float("inf")))
    max_corner = Vector((float("-inf"), float("-inf"), float("-inf")))
    for obj in mesh_objects:
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            min_corner.x = min(min_corner.x, world.x)
            min_corner.y = min(min_corner.y, world.y)
            min_corner.z = min(min_corner.z, world.z)
            max_corner.x = max(max_corner.x, world.x)
            max_corner.y = max(max_corner.y, world.y)
            max_corner.z = max(max_corner.z, world.z)

    center = (min_corner + max_corner) * 0.5
    diagonal = max((max_corner - min_corner).length, 1.0)

    bpy.ops.object.light_add(type="AREA", location=(0.0, 0.25, 0.0))
    light = bpy.context.object
    light.name = "sceneforge_preview_light"
    light.data.energy = 900
    light.data.size = diagonal

    # The primary preview should resemble the input image, not inspect the mesh
    # from an arbitrary orbit angle. Blender imports this OBJ orientation with
    # scene depth along +Y and vertical along Z.
    camera_location = Vector((0.0, 0.0, 0.0))
    bpy.ops.object.camera_add(location=camera_location)
    camera = bpy.context.object
    camera.name = "sceneforge_source_preview_camera"
    direction = Vector((0.0, 1.0, 0.0))
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    camera.data.angle = 2.0 * math.atan(0.5)
    camera.data.clip_start = 0.01
    camera.data.clip_end = diagonal * 20
    bpy.context.scene.camera = camera

    texture_aspect = 4.0 / 3.0
    for image in bpy.data.images:
        if image.size[0] > 0 and image.size[1] > 0 and image.name not in {"Render Result", "Viewer Node"}:
            texture_aspect = image.size[0] / image.size[1]
            break
    if texture_aspect >= 1.0:
        bpy.context.scene.render.resolution_x = 1200
        bpy.context.scene.render.resolution_y = max(1, round(1200 / texture_aspect))
    else:
        bpy.context.scene.render.resolution_y = 1200
        bpy.context.scene.render.resolution_x = max(1, round(1200 * texture_aspect))
    bpy.context.scene.render.film_transparent = False
    try:
        bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        try:
            bpy.context.scene.render.engine = "BLENDER_EEVEE"
        except TypeError:
            pass
    bpy.context.scene.world.color = (0.03, 0.03, 0.03)
    bpy.context.scene.render.filepath = str(preview_path)
    bpy.ops.render.render(write_still=True)

bpy.ops.file.pack_all()
bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
"""
