from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    parser.add_argument("--render-samples", type=int, default=8)
    parser.add_argument("--exposure", default="auto")
    parser.add_argument("--gamma", type=float, default=1.0)
    return parser.parse_args(script_args)


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

    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = args.render_samples
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    exposure = str(args.exposure).strip().lower()
    if exposure != "auto":
        scene.view_settings.exposure = float(args.exposure)
    if hasattr(scene.view_settings, "gamma"):
        scene.view_settings.gamma = float(args.gamma)
    camera.data.sensor_fit = "HORIZONTAL"
    return scene


def render_png(path: Path) -> None:
    scene = bpy.context.scene
    path.parent.mkdir(parents=True, exist_ok=True)
    scene.use_nodes = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def main() -> int:
    args = parse_args()
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
                    "8",
                    "--exposure",
                    "auto",
                ],
                blend_path="Assets/Samples/shapes.blend",
            )
        )
    raise SystemExit(main())
