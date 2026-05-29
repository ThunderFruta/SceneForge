from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_blender_view_script(
    blend_path: Path,
    preview_dir: Path,
    preview_stem: str,
    glb_path: Path,
    render_size: int,
    views: str,
    orbit_steps: int,
    orbit_elevation_deg: float,
    export_gltf: bool,
    report_path: Path,
    include_report: bool,
) -> str:
    script = '''from pathlib import Path
import json
import math

import bpy
from mathutils import Vector

blend_path = Path("{{blend_path}}")
preview_dir = Path("{{preview_dir}}")
preview_stem = "{{preview_stem}}"
glb_path = Path("{{glb_path}}")
render_size = {{render_size}}
views_arg = "{{views}}"
orbit_steps = {{orbit_steps}}
orbit_elevation_deg = {{orbit_elevation_deg}}
export_gltf = {{export_gltf}}
report_path = Path("{{report_path}}")
include_report = {{include_report}}


def parse_views() -> list[tuple[str, float, float]]:
    entries = []
    values = [v.strip().lower() for v in views_arg.split(",") if v.strip()]
    if not values:
        values = ["front"]

    for value in values:
        if value == "front":
            entries.append(("front", math.radians(90.0), math.radians(12.0)))
        elif value == "left":
            entries.append(("left", math.radians(0.0), math.radians(12.0)))
        elif value == "right":
            entries.append(("right", math.radians(180.0), math.radians(12.0)))
        elif value == "back":
            entries.append(("back", math.radians(270.0), math.radians(12.0)))
        elif value == "top":
            entries.append(("top", math.radians(90.0), math.radians(75.0)))
        elif value == "bottom":
            entries.append(("bottom", math.radians(90.0), math.radians(-75.0)))
        elif value == "iso":
            entries.append(("iso", math.radians(45.0), math.radians(35.0)))
        elif value == "orbit":
            if orbit_steps > 1:
                for i in range(orbit_steps):
                    azimuth = 2.0 * math.pi * i / orbit_steps
                    entries.append((f"orbit_{i:03d}", azimuth, math.radians(orbit_elevation_deg)))
        else:
            raise RuntimeError(f"Unsupported view preset: {value}")

    if not entries:
        entries.append(("front", math.radians(90.0), math.radians(12.0)))
    return entries


def scene_bounds_world() -> tuple[Vector, Vector]:
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not mesh_objects:
        return Vector((-1.0, -1.0, -1.0)), Vector((1.0, 1.0, 1.0))

    min_corner = None
    max_corner = None
    for obj in mesh_objects:
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            if min_corner is None:
                min_corner = world.copy()
                max_corner = world.copy()
            else:
                min_corner.x = min(min_corner.x, world.x)
                min_corner.y = min(min_corner.y, world.y)
                min_corner.z = min(min_corner.z, world.z)
                max_corner.x = max(max_corner.x, world.x)
                max_corner.y = max(max_corner.y, world.y)
                max_corner.z = max(max_corner.z, world.z)

    assert min_corner is not None and max_corner is not None
    return min_corner, max_corner


def set_camera_view(camera_obj, center: Vector, azimuth: float, elevation: float, distance: float) -> None:
    offset = Vector(
        (
            distance * math.cos(elevation) * math.cos(azimuth),
            distance * math.cos(elevation) * math.sin(azimuth),
            distance * math.sin(elevation),
        )
    )
    camera_obj.location = center + offset
    forward = center - camera_obj.location
    camera_obj.rotation_euler = forward.to_track_quat("-Z", "Y").to_euler()


def ensure_lighting(distance: float) -> None:
    if not any(obj.type == "LIGHT" for obj in bpy.context.scene.objects):
        bpy.ops.object.light_add(type="AREA", location=(0.0, 0.0, 0.0))
        light = bpy.context.object
        light.name = "sceneforge_preview_fill"
        light.data.energy = 1200.0
        light.data.size = max(1.0, distance)


def render_preview(path: Path, camera_obj) -> None:
    bpy.context.scene.camera = camera_obj
    bpy.context.scene.render.filepath = str(path)
    try:
        bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        bpy.context.scene.render.engine = "BLENDER_EEVEE"
    bpy.context.scene.render.film_transparent = False
    bpy.context.scene.render.resolution_x = render_size
    bpy.context.scene.render.resolution_y = render_size
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


bpy.ops.wm.open_mainfile(filepath=str(blend_path))
mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
min_corner, max_corner = scene_bounds_world()
center = (min_corner + max_corner) * 0.5
radius = max((max_corner - min_corner).length, 0.5)

camera = bpy.context.scene.camera
if camera is None:
    for candidate_name in ("sceneforge_source_preview_camera", "Camera"):
        candidate = bpy.data.objects.get(candidate_name)
        if candidate is not None and candidate.type == "CAMERA":
            camera = candidate
            break

if camera is None:
    bpy.ops.object.camera_add(location=(0.0, 0.0, 0.0))
    camera = bpy.context.object
    camera.name = "sceneforge_source_preview_camera"

camera.data.clip_start = 0.001
camera.data.clip_end = max(radius * 20.0, 1000.0)
camera.data.lens = 50
ensure_lighting(radius * 2.4)

camera_records = []
for name, azimuth, elevation in parse_views():
    set_camera_view(camera, center, azimuth, elevation, radius * 2.4)
    output = preview_dir / f"{preview_stem}_{name}.png"
    render_preview(output, camera)
    camera_records.append(
        {
            "name": name,
            "azimuth_rad": azimuth,
            "elevation_rad": elevation,
            "camera_location": [float(v) for v in camera.location],
            "camera_rotation_euler": [float(v) for v in camera.rotation_euler],
            "output": str(output),
        }
    )

if export_gltf:
    if hasattr(bpy.ops.export_scene, "gltf"):
        bpy.ops.export_scene.gltf(filepath=str(glb_path), export_format="GLB")
    else:
        raise RuntimeError("Blender build does not expose the glTF exporter.")

if include_report:
    mesh_statistics = {
        "mesh_count": len(mesh_objects),
        "total_vertices": sum(len(obj.data.vertices) for obj in mesh_objects),
        "total_edges": sum(len(obj.data.edges) for obj in mesh_objects),
        "total_faces": sum(len(obj.data.polygons) for obj in mesh_objects),
    }
    payload = {
        "blend_path": str(blend_path),
        "mesh": mesh_statistics,
        "bounds": {
            "min": [float(min_corner.x), float(min_corner.y), float(min_corner.z)],
            "max": [float(max_corner.x), float(max_corner.y), float(max_corner.z)],
            "center": [float(center.x), float(center.y), float(center.z)],
            "radius": float(radius),
        },
        "render_size": render_size,
        "views": camera_records,
        "outputs": {
            "gltf": str(glb_path) if export_gltf else None,
        },
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
'''

    return (
        script
        .replace("{{blend_path}}", str(blend_path))
        .replace("{{preview_dir}}", str(preview_dir))
        .replace("{{preview_stem}}", preview_stem)
        .replace("{{glb_path}}", str(glb_path))
        .replace("{{render_size}}", str(int(render_size)))
        .replace("{{views}}", views)
        .replace("{{orbit_steps}}", str(int(orbit_steps)))
        .replace("{{orbit_elevation_deg}}", str(float(orbit_elevation_deg)))
        .replace("{{export_gltf}}", "True" if export_gltf else "False")
        .replace("{{report_path}}", str(report_path))
        .replace("{{include_report}}", "True" if include_report else "False")
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render one or more preview views and export a viewer-friendly glTF from "
            "a SceneForge .blend output."
        )
    )
    parser.add_argument("--blend", required=True, type=Path, help="Path to a .blend file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for outputs. Defaults to the blend parent.",
    )
    parser.add_argument(
        "--preview",
        type=Path,
        help=(
            "Path for first preview PNG (for single-view output). "
            "For multiple views this becomes a stem prefix."
        ),
    )
    parser.add_argument(
        "--gltf",
        type=Path,
        help="Custom path for output .glb. Defaults to <blend>.glb in output-dir.",
    )
    parser.add_argument(
        "--no-gltf",
        action="store_true",
        help="Skip .glb export and only render preview images.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Write a JSON report. Defaults to <blend>_view_report.json in output-dir.",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip JSON report generation.",
    )
    parser.add_argument(
        "--blender",
        default="blender",
        help="Blender executable name or full path.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1200,
        help="Square render size in pixels.",
    )
    parser.add_argument(
        "--views",
        default="front",
        help="Comma list: front,left,right,back,top,bottom,iso,orbit",
    )
    parser.add_argument(
        "--orbit-steps",
        type=int,
        default=0,
        help="Number of orbit steps if views includes `orbit`.",
    )
    parser.add_argument(
        "--orbit-elevation",
        type=float,
        default=25.0,
        help="Orbit camera elevation in degrees.",
    )
    return parser.parse_args(argv)


def _derive_previews(args: argparse.Namespace, stem: str, preview_dir: Path) -> list[Path]:
    requested = [value.strip().lower() for value in args.views.split(",") if value.strip()]
    if not requested:
        requested = ["front"]

    outputs: list[Path] = []
    for value in requested:
        if value == "orbit":
            count = max(0, args.orbit_steps)
            if count <= 1:
                count = 8
            for i in range(count):
                outputs.append(preview_dir / f"{stem}_orbit_{i:03d}.png")
        else:
            outputs.append(preview_dir / f"{stem}_{value}.png")

    if not outputs:
        outputs.append(preview_dir / f"{stem}_front.png")
    return outputs


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    blend_path = args.blend.resolve()
    if not blend_path.exists():
        raise SystemExit(f"Blend file not found: {blend_path}")

    executable = shutil.which(args.blender)
    if executable is None:
        raise SystemExit(
            "Blender executable was not found. Install Blender or pass --blender /path/to/blender."
        )

    output_dir = args.output_dir or blend_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    preview_dir = output_dir
    preview_stem = f"{blend_path.stem}_view"
    if args.preview:
        preview_dir = args.preview.parent if args.preview.parent != Path(".") else output_dir
        preview_stem = args.preview.stem
    preview_dir.mkdir(parents=True, exist_ok=True)

    glb_path = args.gltf or output_dir / f"{blend_path.stem}.glb"
    report_path = args.report or output_dir / f"{blend_path.stem}_view_report.json"

    expected_previews = _derive_previews(args, preview_stem, preview_dir)
    script = build_blender_view_script(
        blend_path=blend_path,
        preview_dir=preview_dir,
        preview_stem=preview_stem,
        glb_path=glb_path,
        render_size=args.size,
        views=args.views,
        orbit_steps=max(0, args.orbit_steps),
        orbit_elevation_deg=args.orbit_elevation,
        export_gltf=not args.no_gltf,
        report_path=report_path,
        include_report=not args.no_report,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix="_sceneforge_view.py",
        encoding="utf-8",
        delete=False,
    ) as script_file:
        script_file.write(script)
        script_path = Path(script_file.name)

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
        raise SystemExit(
            "Blender failed while preparing .blend view artifacts.\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    missing = [path for path in expected_previews if not path.exists()]
    if missing:
        raise SystemExit("Expected preview output was not produced: " + ", ".join(str(path) for path in missing))

    for path in expected_previews:
        print(f"Wrote preview: {path}")

    if not args.no_gltf and not glb_path.exists():
        raise SystemExit(f"Expected glTF output was not produced at: {glb_path}")
    if not args.no_gltf:
        print(f"Wrote glTF: {glb_path}")

    if not args.no_report and not report_path.exists():
        raise SystemExit(f"Expected report output was not produced at: {report_path}")
    if not args.no_report:
        print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        from Runtime.guided_cli import guided_tool_main

        def _run(argv: list[str]) -> int:
            main(argv)
            return 0

        raise SystemExit(
            guided_tool_main(
                Path(__file__),
                "Inspect a SceneForge .blend output with preview renders.",
                ["--blend", "path/to/output.blend", "--views", "front,iso", "--no-gltf"],
                _run,
            )
        )
    main()
