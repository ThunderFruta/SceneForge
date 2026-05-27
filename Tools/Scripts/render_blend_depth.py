from __future__ import annotations

import argparse
import math
import shutil
import sys
import tempfile
from pathlib import Path

import bpy


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


def configure_depth_compositor(output_dir: Path, near_depth: float, far_depth: float) -> None:
    scene = bpy.context.scene
    scene.view_layers[0].use_pass_z = True
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    render_layers = tree.nodes.new(type="CompositorNodeRLayers")
    map_range = tree.nodes.new(type="CompositorNodeMapRange")
    map_range.use_clamp = True
    map_range.inputs["From Min"].default_value = near_depth
    map_range.inputs["From Max"].default_value = far_depth
    map_range.inputs["To Min"].default_value = 1.0
    map_range.inputs["To Max"].default_value = 0.0

    file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    file_output.base_path = str(output_dir)
    file_output.file_slots[0].path = "depth_"
    file_output.format.file_format = "PNG"
    file_output.format.color_mode = "BW"
    file_output.format.color_depth = "8"

    tree.links.new(render_layers.outputs["Depth"], map_range.inputs["Value"])
    tree.links.new(map_range.outputs["Value"], file_output.inputs[0])


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
            raise RuntimeError("Depth compositor did not write an output PNG.")
        shutil.copyfile(rendered[0], output_path)

    print(f"Rendered depth map: {output_path}")
    print(f"Depth range: near={near:.6f}, far={far:.6f}")


if __name__ == "__main__":
    args = parse_args()
    apply_camera_fov(args.fov_degrees)
    render_depth(Path(args.output), args.near_depth, args.far_depth)
