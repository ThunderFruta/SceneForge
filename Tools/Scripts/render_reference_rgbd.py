from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from math import degrees
from pathlib import Path

try:
    from render_progress import BlenderRenderProgressBar
except ModuleNotFoundError:
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from render_progress import BlenderRenderProgressBar

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
    parser.add_argument("--render-quality", choices=("fast", "balanced", "quality"), default="balanced")
    parser.add_argument("--render-engine", default="CYCLES", choices=("auto", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "CYCLES"))
    parser.add_argument("--cycles-device-filter", default="4080")
    parser.add_argument("--near-depth", type=float, default=1.0)
    parser.add_argument("--far-depth", type=float, default=8.0)
    parser.add_argument("--exposure", default="auto")
    parser.add_argument("--gamma", type=float, default=1.0)
    return parser.parse_args(script_args)


def _configure_cycles_device(preferred_device_filter: str = "4080") -> None:
    cycles_addon = bpy.context.preferences.addons.get("cycles") if bpy else None
    if cycles_addon is None:
        return
    cycles_preferences = cycles_addon.preferences
    scene = bpy.context.scene
    preferred_device = str(preferred_device_filter or "").strip().lower()

    # Choose a GPU backend first, with CUDA prioritized.
    for compute_type in ("CUDA", "OPTIX", "HIP", "ONEAPI", "METAL"):
        try:
            if compute_type in [item[0] for item in cycles_preferences.get_device_types(bpy.context)]:
                cycles_preferences.compute_device_type = compute_type
                break
        except Exception:
            continue

    if hasattr(cycles_preferences, "refresh_devices"):
        cycles_preferences.refresh_devices()

    gpu_devices = [
        device
        for device in getattr(cycles_preferences, "devices", [])
        if str(device.type).upper() != "CPU"
    ]
    if not gpu_devices:
        if hasattr(scene.cycles, "device"):
            scene.cycles.device = "CPU"
        return

    preferred_matches = []
    if preferred_device:
        preferred_matches = [
            device
            for device in gpu_devices
            if preferred_device in str(device.name).lower()
        ]

    if preferred_matches:
        for device in gpu_devices:
            device.use = device in preferred_matches
        if not any(device.use for device in gpu_devices):
            for device in gpu_devices:
                device.use = True
    else:
        for device in gpu_devices:
            device.use = True

    if hasattr(scene.cycles, "device"):
        scene.cycles.device = "GPU"

def configure_render_quality(scene: bpy.types.Scene, sample_budget: int, render_quality: str = "balanced") -> None:
    quality = str(render_quality or "balanced").strip().lower()
    if quality == "fast":
        taa_samples = max(4, int(sample_budget))
    elif quality == "quality":
        taa_samples = max(64, int(sample_budget))
    else:
        taa_samples = max(16, int(sample_budget))

    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = int(taa_samples)
        for attr, value in ((
            ("use_bloom", False),
            ("use_ssr", False),
            ("use_ssao", False),
            ("use_gtao", False),
            ("use_bokeh_jitter", False),
            ("use_volumetric_shadows", False),
        )):
            if hasattr(scene.eevee, attr):
                setattr(scene.eevee, attr, value)

    if hasattr(scene.render, "use_motion_blur"):
        scene.render.use_motion_blur = False

    if scene.render.engine == "CYCLES" and hasattr(scene.cycles, "samples"):
        scene.cycles.samples = max(1, int(taa_samples))
        if hasattr(scene.cycles, "use_denoising"):
            scene.cycles.use_denoising = False

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

    if args.render_engine != "auto":
        scene.render.engine = args.render_engine
    if scene.render.engine == "CYCLES":
        _configure_cycles_device(args.cycles_device_filter)
    configure_render_quality(scene, args.render_samples, args.render_quality)
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.view_layers[0].use_pass_z = True    
    exposure = str(args.exposure).strip().lower()
    if exposure != "auto":
        scene.view_settings.exposure = float(args.exposure)
    if hasattr(scene.view_settings, "gamma"):
        scene.view_settings.gamma = float(args.gamma)
    camera.data.sensor_fit = "HORIZONTAL"
    return camera


def _is_cuda_misaligned_error(message: str) -> bool:
    lowered = str(message).lower()
    return "misaligned address in cuda queue" in lowered or "misaligned address" in lowered


def _fallback_cycles_to_cpu() -> bool:
    cycles_addon = bpy.context.preferences.addons.get("cycles") if bpy else None
    if cycles_addon is None:
        return False
    cycles_preferences = cycles_addon.preferences
    scene = bpy.context.scene

    if hasattr(cycles_preferences, "compute_device_type"):
        try:
            cycles_preferences.compute_device_type = "NONE"
        except Exception:
            pass

    devices = list(getattr(cycles_preferences, "devices", []))
    if not devices:
        if hasattr(scene.cycles, "device"):
            scene.cycles.device = "CPU"
        return True

    for device in devices:
        if str(device.type).upper() == "CPU":
            device.use = True
        else:
            device.use = False

    if hasattr(scene.cycles, "device"):
        scene.cycles.device = "CPU"
    return True


def render_rgb(path: Path) -> None:
    scene = bpy.context.scene
    path.parent.mkdir(parents=True, exist_ok=True)
    scene.use_nodes = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.filepath = str(path)
    try:
        bpy.ops.render.render(write_still=True)
        return
    except RuntimeError as exc:
        if _is_cuda_misaligned_error(str(exc)) and _fallback_cycles_to_cpu():
            print("CUDA render failed (misaligned address); retrying on CPU.")
            bpy.ops.render.render(write_still=True)
            return
        raise


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


def render_depth(path: Path, near_depth: float, far_depth: float) -> None:
    scene = bpy.context.scene
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    target_width = int(scene.render.resolution_x * scene.render.resolution_percentage / 100)
    target_height = int(scene.render.resolution_y * scene.render.resolution_percentage / 100)
    depth_width, depth_height = depth_render_size(target_width, target_height)
    original_resolution = (scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage)
    try:
        scene.render.resolution_x = depth_width
        scene.render.resolution_y = depth_height
        scene.render.resolution_percentage = 100
        if hasattr(scene, "eevee"):
            scene.eevee.taa_render_samples = 1
        with tempfile.TemporaryDirectory(prefix="sceneforge_depth_") as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            exr_path = temp_dir / "depth.exr"
            configure_depth_exr_compositor(exr_path)
            scene.frame_set(1)
            try:
                bpy.ops.render.render(write_still=False)
            except RuntimeError as exc:
                if _is_cuda_misaligned_error(str(exc)) and _fallback_cycles_to_cpu():
                    print("CUDA depth render failed (misaligned address); retrying on CPU.")
                    bpy.ops.render.render(write_still=False)
                else:
                    raise
            candidates = sorted(temp_dir.glob("depth*.exr"))
            if candidates and convert_depth_exr_to_png(candidates[-1], path, near_depth, far_depth, target_width, target_height):
                return
    finally:
        scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = original_resolution
    raise RuntimeError(
        "Blender Z-pass depth render did not produce a convertible EXR. "
        "Refusing to fall back to CPU raycast depth for this render."
    )


def depth_render_size(width: int, height: int, max_edge: int = 384) -> tuple[int, int]:
    scale = min(1.0, float(max_edge) / float(max(width, height)))
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def configure_depth_exr_compositor(path: Path) -> None:
    scene = bpy.context.scene
    scene.view_layers[0].use_pass_z = True
    tree = compositor_tree(scene)
    for node in list(tree.nodes):
        tree.nodes.remove(node)

    layers = tree.nodes.new(type="CompositorNodeRLayers")
    output = tree.nodes.new(type="CompositorNodeOutputFile")
    if hasattr(output, "directory"):
        output.directory = str(path.parent)
        output.file_name = path.stem
        output.file_output_items.clear()
        output.file_output_items.new("FLOAT", "Depth")
        output_input = output.inputs.get("Depth") or output.inputs[0]
    else:
        output.base_path = str(path.parent)
        output.file_slots[0].path = path.stem
        output.format.file_format = "OPEN_EXR"
        output_input = output.inputs[0]

    depth_output = layers.outputs.get("Depth") or layers.outputs.get("Z")
    if depth_output is None:
        available = ", ".join(item.name for item in layers.outputs)
        raise RuntimeError(f"Render layer depth pass is unavailable; outputs: {available}")
    tree.links.new(depth_output, output_input)


def convert_depth_exr_to_png(exr_path: Path, output_path: Path, near_depth: float, far_depth: float, output_width: int, output_height: int) -> bool:
    python_path = Path.cwd() / ".venv" / "bin" / "python"
    if not python_path.is_file():
        python_path = Path(sys.executable)
    code = """
import sys
import numpy as np
from PIL import Image
import OpenEXR

exr_path, output_path, near_depth, far_depth = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
exr = OpenEXR.File(exr_path)
channels = exr.parts[0].channels
channel = channels.get('Depth.V') or channels.get('Depth') or next(iter(channels.values()))
depth = np.asarray(channel.pixels, dtype=np.float32)
depth = np.nan_to_num(depth, nan=far_depth, posinf=far_depth, neginf=far_depth)
normalized = 1.0 - ((depth - near_depth) / (far_depth - near_depth))
image = (np.clip(normalized, 0.0, 1.0) * 255.0).round().astype(np.uint8)
Image.fromarray(image, mode='L').resize((int(sys.argv[5]), int(sys.argv[6])), Image.Resampling.BILINEAR).save(output_path)
"""
    try:
        subprocess.run(
            [str(python_path), "-c", code, str(exr_path), str(output_path), str(near_depth), str(far_depth), str(output_width), str(output_height)],
            check=True,
        )
        return output_path.is_file()
    except (OSError, subprocess.CalledProcessError):
        return False


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
    with BlenderRenderProgressBar("Rendering image", total_samples=args.render_samples):
        render_rgb(Path(args.image_output))
    with BlenderRenderProgressBar("Rendering depth", total_samples=args.render_samples):
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
                blend_path="Assets/Samples/roomScene.blend",
            )
        )
    raise SystemExit(main())
