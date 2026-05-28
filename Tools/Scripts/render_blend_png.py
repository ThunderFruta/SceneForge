from __future__ import annotations

import argparse
import sys
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
    parser = argparse.ArgumentParser(description="Render a PNG from the active Blender camera.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--camera-name")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--render-samples", type=int, default=2048)
    parser.add_argument("--render-quality", choices=("fast", "balanced", "quality"), default="balanced")
    parser.add_argument("--render-engine", default="CYCLES", choices=("auto", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "CYCLES"))
    parser.add_argument("--cycles-device-filter", default="4080")
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

    if scene.render.engine == "CYCLES" and hasattr(scene.cycles, "samples"):
        scene.cycles.samples = max(1, int(taa_samples))
        if hasattr(scene.cycles, "use_denoising"):
            scene.cycles.use_denoising = False

    if hasattr(scene.render, "use_motion_blur"):
        scene.render.use_motion_blur = False

def configure_scene(args: argparse.Namespace):
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
    exposure = str(args.exposure).strip().lower()
    if exposure != "auto":
        scene.view_settings.exposure = float(args.exposure)
    if hasattr(scene.view_settings, "gamma"):
        scene.view_settings.gamma = float(args.gamma)
    camera.data.sensor_fit = "HORIZONTAL"
    return scene


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


def render_png(path: Path) -> None:
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


def main() -> int:
    args = parse_args()
    with BlenderRenderProgressBar("Rendering", total_samples=args.render_samples):
        configure_scene(args)
        render_png(Path(args.output))
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 1:
        root = Path(__file__).resolve().parents[2]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from Runtime.guided_cli import guided_blender_tool_main

        raise SystemExit(
            guided_blender_tool_main(
                Path(__file__),
                "Render a PNG from a .blend file.",
                [
                    "--output",
                    "Output/Latest/render/image.png",
                "--width",
                "1280",
                "--height",
                "720",
                "--render-samples",
                "2048",
                "--render-quality",
                "balanced",
                "--exposure",
                    "auto",
                ],
                blend_path="Assets/Samples/roomScene.blend",
            )
        )
    raise SystemExit(main())
