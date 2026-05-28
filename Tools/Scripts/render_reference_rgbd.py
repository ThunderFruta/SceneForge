from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from math import degrees
from pathlib import Path

try:
    import bpy
except ModuleNotFoundError:
    bpy = None


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    script_args = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser(description="Render RGB and normalized depth from the active Blender camera.")
    parser.add_argument("--image-output", required=True)
    parser.add_argument("--depth-output", required=True)
    parser.add_argument("--camera-output", required=True)
    parser.add_argument("--camera-name")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--render-samples", type=int, default=16)
    parser.add_argument("--near-depth", type=float, default=1.0)
    parser.add_argument("--far-depth", type=float, default=8.0)
    return parser.parse_args(script_args)


def configure_scene(args: argparse.Namespace) -> bpy.types.Object:
    scene = bpy.context.scene
    if args.camera_name:
        camera = bpy.data.objects.get(args.camera_name)
        if camera is None or camera.type != "CAMERA":
            raise ValueError(f"Camera does not exist: {args.camera_name}")
        scene.camera = camera
    camera = scene.camera
    if camera is None:
        raise ValueError("The blend file has no active camera.")

    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = args.render_samples
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.view_layers[0].use_pass_z = True
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    camera.data.sensor_fit = "HORIZONTAL"
    return camera


def render_rgb(path: Path) -> None:
    scene = bpy.context.scene
    path.parent.mkdir(parents=True, exist_ok=True)
    scene.use_nodes = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def compositor_tree(scene: bpy.types.Scene):
    if hasattr(scene, "node_tree"):
        scene.use_nodes = True
        return scene.node_tree
    tree = getattr(scene, "compositing_node_group", None)
    if tree is None:
        tree = bpy.data.node_groups.new("SceneForgeCompositor", "CompositorNodeTree")
        scene.compositing_node_group = tree
    return tree


def render_depth(path: Path, near_depth: float, far_depth: float) -> None:
    scene = bpy.context.scene
    path.parent.mkdir(parents=True, exist_ok=True)
    tree = compositor_tree(scene)
    for node in list(tree.nodes):
        tree.nodes.remove(node)

    layers = tree.nodes.new(type="CompositorNodeRLayers")
    mapper = tree.nodes.new(type="CompositorNodeMapRange")
    mapper.inputs[1].default_value = near_depth
    mapper.inputs[2].default_value = far_depth
    mapper.inputs[3].default_value = 1.0
    mapper.inputs[4].default_value = 0.0
    mapper.use_clamp = True
    output = tree.nodes.new(type="CompositorNodeOutputFile")
    output.base_path = str(path.parent)
    output.file_slots[0].path = path.stem
    output.format.file_format = "PNG"
    output.format.color_mode = "BW"
    output.format.color_depth = "8"
    depth_output = layers.outputs.get("Depth") or layers.outputs.get("Z")
    if depth_output is None:
        available = ", ".join(item.name for item in layers.outputs)
        raise RuntimeError(f"Render layer depth pass is unavailable; outputs: {available}")
    tree.links.new(depth_output, mapper.inputs[0])
    tree.links.new(mapper.outputs[0], output.inputs[0])

    scene.frame_set(1)
    bpy.ops.render.render(write_still=False)
    candidates = sorted(path.parent.glob(f"{path.stem}*.png"))
    if not candidates:
        raise RuntimeError(f"Depth compositor did not write {path}")
    latest = candidates[-1]
    if latest.resolve() != path.resolve():
        path.unlink(missing_ok=True)
        shutil.move(str(latest), str(path))


def write_camera_metadata(path: Path, camera: bpy.types.Object, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "camera_name": camera.name,
        "image_width": args.width,
        "image_height": args.height,
        "sensor_fit": camera.data.sensor_fit.lower(),
        "fov_degrees": degrees(float(camera.data.angle)),
        "shift_x": float(camera.data.shift_x),
        "shift_y": float(camera.data.shift_y),
        "near_depth": args.near_depth,
        "far_depth": args.far_depth,
        "depth_convention": "white_close_black_far",
        "coordinate_contract": "x_right_y_depth_away_z_up_camera_space",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.near_depth <= 0 or args.far_depth <= args.near_depth:
        raise ValueError("--near-depth and --far-depth must satisfy 0 < near < far")
    camera = configure_scene(args)
    render_rgb(Path(args.image_output))
    render_depth(Path(args.depth_output), args.near_depth, args.far_depth)
    write_camera_metadata(Path(args.camera_output), camera, args)
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 1:
        from Runtime.guided_cli import guided_blender_tool_main

        raise SystemExit(
            guided_blender_tool_main(
                Path(__file__),
                "Render RGB/depth/camera outputs from a .blend file.",
                [
                    "--image-output",
                    "Output/Latest/render/image.png",
                    "--depth-output",
                    "Output/Latest/render/depth.png",
                    "--camera-output",
                    "Output/Latest/render/camera.json",
                ],
                blend_path="Assets/Samples/shapes.blend",
            )
        )
    raise SystemExit(main())
