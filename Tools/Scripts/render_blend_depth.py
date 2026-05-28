from __future__ import annotations

import argparse
import math
import shutil
import sys
import tempfile
from pathlib import Path

try:
    import bpy
except ModuleNotFoundError:
    bpy = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a white-close grayscale depth map from a .blend file.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--near-depth", type=float, default=None)
    parser.add_argument("--far-depth", type=float, default=None)
    parser.add_argument("--fov-degrees", type=float, default=None)
    argv = sys.argv
    script_args = argv[argv.index("--") + 1 :] if "--" in argv else []
    return parser.parse_args(script_args)


def apply_camera_fov(fov_degrees: float | None) -> None:
    if fov_degrees is None:
        return
    if fov_degrees <= 0.0 or fov_degrees >= 179.0:
        raise ValueError("--fov-degrees must be between 0 and 179.")
    scene = bpy.context.scene
    if scene.camera is None:
        raise ValueError("The blend file has no active camera.")
    scene.camera.data.sensor_fit = "HORIZONTAL"
    scene.camera.data.angle = math.radians(fov_degrees)


def visible_scene_depth_range() -> tuple[float, float]:
    scene = bpy.context.scene
    camera = scene.camera
    if camera is None:
        raise ValueError("The blend file has no active camera.")

    camera_inverse = camera.matrix_world.inverted()
    distances: list[float] = []
    for obj in scene.objects:
        if obj.type != "MESH":
            continue
        for vertex in obj.data.vertices:
            local = camera_inverse @ (obj.matrix_world @ vertex.co)
            distance = -float(local.z)
            if math.isfinite(distance) and distance > 0.0:
                distances.append(distance)

    if not distances:
        return 0.1, 10.0

    near = max(0.01, min(distances) * 0.95)
    far = max(near + 0.01, max(distances) * 1.05)
    return near, far


def configure_output_file_node(node, path: Path):
    if hasattr(node, "base_path"):
        node.base_path = str(path.parent)
        node.file_slots[0].path = path.stem
        node.format.file_format = "PNG"
        node.format.color_mode = "BW"
        node.format.color_depth = "8"
        return node.inputs[0]
    node.file_name = str(path.with_suffix(""))
    node.file_output_items.clear()
    item = node.file_output_items.new("FLOAT", "Depth")
    item.override_node_format = True
    item.format.file_format = "PNG"
    item.format.color_mode = "BW"
    item.format.color_depth = "8"
    return node.inputs.get("Depth") or node.inputs[0]


def new_map_range_node(tree):
    try:
        node = tree.nodes.new(type="CompositorNodeMapRange")
        return node, "Value", "Value"
    except RuntimeError:
        node = tree.nodes.new(type="ShaderNodeMapRange")
        return node, "Value", "Result"


def compositor_tree(scene: bpy.types.Scene):
    if hasattr(scene, "node_tree"):
        scene.use_nodes = True
        return scene.node_tree
    tree = getattr(scene, "compositing_node_group", None)
    if tree is None:
        tree = bpy.data.node_groups.new("SceneForgeCompositor", "CompositorNodeTree")
        scene.compositing_node_group = tree
    return tree




def write_raycast_depth(path: Path, near_depth: float, far_depth: float) -> None:
    from PIL import Image
    from mathutils import Vector

    scene = bpy.context.scene
    camera = scene.camera
    if camera is None:
        raise RuntimeError("The blend file has no active camera.")
    width = int(scene.render.resolution_x * scene.render.resolution_percentage / 100)
    height = int(scene.render.resolution_y * scene.render.resolution_percentage / 100)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    camera_matrix = camera.matrix_world
    camera_inverse = camera_matrix.inverted()
    frame = camera.data.view_frame(scene=scene)
    top_right, bottom_right, bottom_left, top_left = frame
    forward = (camera_matrix.to_3x3() @ Vector((0.0, 0.0, -1.0))).normalized()
    pixels = bytearray(width * height)

    for y in range(height):
        v = 1.0 - ((y + 0.5) / height)
        left = bottom_left.lerp(top_left, v)
        right = bottom_right.lerp(top_right, v)
        for x in range(width):
            u = (x + 0.5) / width
            local_point = left.lerp(right, u)
            if camera.data.type == "ORTHO":
                origin = camera_matrix @ local_point
                direction = forward
            else:
                origin = camera_matrix.translation
                direction = (camera_matrix.to_3x3() @ local_point).normalized()
            hit, location, _normal, _index, _obj, _matrix = scene.ray_cast(depsgraph, origin, direction, distance=far_depth)
            if hit:
                camera_space = camera_inverse @ location
                depth = max(near_depth, min(far_depth, -float(camera_space.z)))
            else:
                depth = far_depth
            value = int(round(255.0 * (1.0 - ((depth - near_depth) / (far_depth - near_depth)))))
            pixels[(y * width) + x] = max(0, min(255, value))

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.frombytes("L", (width, height), bytes(pixels)).save(path)

def configure_depth_compositor(output_dir: Path, near_depth: float, far_depth: float) -> None:
    scene = bpy.context.scene
    scene.view_layers[0].use_pass_z = True
    tree = compositor_tree(scene)
    for node in list(tree.nodes):
        tree.nodes.remove(node)

    render_layers = tree.nodes.new(type="CompositorNodeRLayers")
    map_range, map_input_name, map_output_name = new_map_range_node(tree)
    if hasattr(map_range, "use_clamp"):
        map_range.use_clamp = True
    if hasattr(map_range, "clamp"):
        map_range.clamp = True
    map_range.inputs["From Min"].default_value = near_depth
    map_range.inputs["From Max"].default_value = far_depth
    map_range.inputs["To Min"].default_value = 1.0
    map_range.inputs["To Max"].default_value = 0.0

    file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    output_input = configure_output_file_node(file_output, output_dir / "depth_.png")

    depth_output = render_layers.outputs.get("Depth") or render_layers.outputs.get("Z")
    if depth_output is None:
        available = ", ".join(item.name for item in render_layers.outputs)
        raise RuntimeError(f"Render layer depth pass is unavailable; outputs: {available}")
    tree.links.new(depth_output, map_range.inputs[map_input_name])
    tree.links.new(map_range.outputs[map_output_name], output_input)


def render_depth(output_path: Path, near_depth: float | None, far_depth: float | None) -> None:
    auto_near, auto_far = visible_scene_depth_range()
    near = auto_near if near_depth is None else near_depth
    far = auto_far if far_depth is None else far_depth
    if near <= 0.0 or far <= near:
        raise ValueError("Depth range must satisfy 0 < near < far.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        configure_depth_compositor(temp_dir, near, far)
        bpy.ops.render.render(write_still=False)
        rendered = sorted(temp_dir.glob("depth_*.png"))
        if not rendered:
            write_raycast_depth(output_path, near, far)
        else:
            shutil.copyfile(rendered[0], output_path)

    print(f"Rendered depth map: {output_path}")
    print(f"Depth range: near={near:.6f}, far={far:.6f}")


if __name__ == "__main__":
    args = parse_args()
    apply_camera_fov(args.fov_degrees)
    render_depth(Path(args.output), args.near_depth, args.far_depth)
