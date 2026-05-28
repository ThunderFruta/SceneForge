from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

try:
    import bpy
except ModuleNotFoundError:
    bpy = None
from mathutils import Matrix, Vector


PRIMITIVE_COLORS = {
    "sphere": (0.165, 0.616, 0.561, 1.0),
    "cylinder": (0.149, 0.275, 0.325, 1.0),
    "cone": (0.902, 0.224, 0.275, 1.0),
    "box": (0.957, 0.635, 0.38, 1.0),
    "plane": (0.271, 0.482, 0.616, 1.0),
    "unknown": (0.424, 0.459, 0.49, 1.0),
}
CONE_TIP_RADIUS_RATIO = 0.06


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export fitted primitive JSON to a Blender scene.")
    parser.add_argument("--fits", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--layout",
        choices=("camera", "ground", "original-camera"),
        default="camera",
        help="camera preserves camera-space fitting; ground creates an upright inspection layout; original-camera maps fitted objects into the reference blend camera frame.",
    )
    parser.add_argument(
        "--reference-blend",
        help="Optional .blend whose active camera pose and framing should be mapped onto the ground layout.",
    )
    argv = sys.argv
    script_args = argv[argv.index("--") + 1 :] if "--" in argv else []
    return parser.parse_args(script_args)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def make_material(label: str) -> bpy.types.Material:
    material = bpy.data.materials.new(f"{label}_material")
    material.diffuse_color = PRIMITIVE_COLORS.get(label, PRIMITIVE_COLORS["unknown"])
    return material


def primitive_dimensions(label: str, dimensions: list[float]) -> tuple[float, float, float]:
    x, y, z = (max(0.02, float(value)) for value in dimensions)
    if label == "sphere":
        diameter = max(x, y, z)
        return diameter, diameter, diameter
    if label in {"cylinder", "cone"}:
        diameter = max(x, y)
        return diameter, diameter, z
    if label == "plane":
        return x, y, max(0.02, z)
    return x, y, z


def add_primitive(label: str) -> bpy.types.Object:
    if label == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=0.5)
    elif label == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=0.5, depth=1.0)
    elif label == "cone":
        tip_radius = 0.5 * CONE_TIP_RADIUS_RATIO
        bpy.ops.mesh.primitive_cone_add(vertices=48, radius1=0.5, radius2=tip_radius, depth=1.0)
    elif label == "plane":
        bpy.ops.mesh.primitive_cube_add(size=1.0)
    else:
        bpy.ops.mesh.primitive_cube_add(size=1.0)
    return bpy.context.object


def make_ground_layout_context(objects: list[dict]) -> dict:
    if not objects:
        return {
            "max_depth": 0.0,
            "layout_center": Vector((0.0, 0.0, 0.0)),
            "layout_extent": 4.0,
            "layout_radius": 2.0,
        }

    centers = [Vector(item["center_xyz"]) for item in objects]
    dimensions = [
        primitive_dimensions(item["primitive_label"], item["dimensions_xyz"])
        for item in objects
    ]
    max_depth = max(center.y for center in centers)

    layout_points: list[Vector] = []
    for center, dims in zip(centers, dimensions):
        layout_center = Vector((center.x, center.y - max_depth, dims[2] * 0.5))
        radius = max(dims) * 0.5
        layout_points.append(layout_center + Vector((radius, radius, radius)))
        layout_points.append(layout_center - Vector((radius, radius, radius)))

    minimum = Vector((
        min(point.x for point in layout_points),
        min(point.y for point in layout_points),
        min(point.z for point in layout_points),
    ))
    maximum = Vector((
        max(point.x for point in layout_points),
        max(point.y for point in layout_points),
        max(point.z for point in layout_points),
    ))
    layout_center = (minimum + maximum) * 0.5
    layout_extent = max(4.0, max(maximum.x - minimum.x, maximum.y - minimum.y, maximum.z - minimum.z))
    layout_radius = max(0.5, float((maximum - minimum).length) * 0.5)
    return {
        "max_depth": max_depth,
        "layout_center": layout_center,
        "layout_extent": layout_extent,
        "layout_radius": layout_radius,
    }


def mesh_scene_bounds() -> tuple[Vector, Vector]:
    points: list[Vector] = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        points.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
    if not points:
        return Vector((-1.0, -1.0, -1.0)), Vector((1.0, 1.0, 1.0))
    minimum = Vector((min(point[index] for point in points) for index in range(3)))
    maximum = Vector((max(point[index] for point in points) for index in range(3)))
    return minimum, maximum


def load_reference_camera(reference_blend_path: Path | None) -> dict | None:
    if reference_blend_path is None:
        return None
    if not reference_blend_path.is_file():
        raise FileNotFoundError(f"Reference blend does not exist: {reference_blend_path}")

    bpy.ops.wm.open_mainfile(filepath=str(reference_blend_path))
    scene = bpy.context.scene
    camera = bpy.context.scene.camera
    if camera is None:
        raise ValueError(f"Reference blend has no active camera: {reference_blend_path}")
    bounds_min, bounds_max = mesh_scene_bounds()
    center = (bounds_min + bounds_max) * 0.5
    radius = max(0.5, float((bounds_max - bounds_min).length) * 0.5)
    return {
        "center": center,
        "radius": radius,
        "camera_location": camera.location.copy(),
        "camera_rotation": camera.rotation_euler.copy(),
        "camera_matrix_world": camera.matrix_world.copy(),
        "angle": float(camera.data.angle),
        "sensor_fit": camera.data.sensor_fit,
        "shift_x": float(camera.data.shift_x),
        "shift_y": float(camera.data.shift_y),
        "clip_start": float(camera.data.clip_start),
        "clip_end": float(camera.data.clip_end),
        "resolution_x": int(scene.render.resolution_x),
        "resolution_y": int(scene.render.resolution_y),
        "resolution_percentage": int(scene.render.resolution_percentage),
        "pixel_aspect_x": float(scene.render.pixel_aspect_x),
        "pixel_aspect_y": float(scene.render.pixel_aspect_y),
    }


def apply_camera_transform(obj: bpy.types.Object, item: dict) -> None:
    center = Vector(item["center_xyz"])
    dimensions = primitive_dimensions(item["primitive_label"], item["dimensions_xyz"])
    rotation = Matrix(item["rotation_matrix"]).to_4x4()
    scale = Matrix.Diagonal((dimensions[0], dimensions[1], dimensions[2], 1.0))
    translation = Matrix.Translation(center)
    obj.matrix_world = translation @ rotation @ scale
    obj["camera_space_center_xyz"] = [float(value) for value in item["center_xyz"]]
    obj["camera_space_rotation_matrix"] = item["rotation_matrix"]


def apply_ground_transform(obj: bpy.types.Object, item: dict, layout_context: dict) -> None:
    center = Vector(item["center_xyz"])
    dimensions = primitive_dimensions(item["primitive_label"], item["dimensions_xyz"])
    layout_center = Vector((
        center.x,
        center.y - float(layout_context["max_depth"]),
        dimensions[2] * 0.5,
    ))
    scale = Matrix.Diagonal((dimensions[0], dimensions[1], dimensions[2], 1.0))
    obj.matrix_world = Matrix.Translation(layout_center) @ scale
    obj["camera_space_center_xyz"] = [float(value) for value in item["center_xyz"]]
    obj["camera_space_rotation_matrix"] = item["rotation_matrix"]


def sceneforge_to_blender_camera_matrix() -> Matrix:
    return Matrix(
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )


def apply_original_camera_transform(obj: bpy.types.Object, item: dict, reference_camera: dict) -> None:
    center = Vector(item["center_xyz"])
    dimensions = primitive_dimensions(item["primitive_label"], item["dimensions_xyz"])
    scale = Matrix.Diagonal((dimensions[0], dimensions[1], dimensions[2], 1.0))
    source_to_world = reference_camera["camera_matrix_world"] @ sceneforge_to_blender_camera_matrix()
    source_rotation = Matrix(item["rotation_matrix"]).to_4x4()
    source_transform = Matrix.Translation(center) @ source_rotation @ scale
    obj.matrix_world = source_to_world @ source_transform
    obj["camera_space_center_xyz"] = [float(value) for value in item["center_xyz"]]
    obj["camera_space_rotation_matrix"] = item["rotation_matrix"]
    obj["source_camera_frame"] = "reference_blend_active_camera_world"


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def setup_camera(report: dict, layout: str, layout_context: dict, reference_camera: dict | None) -> None:
    if layout == "original-camera":
        if reference_camera is None:
            raise ValueError("--reference-blend is required for --layout original-camera")
        bpy.ops.object.camera_add()
        camera = bpy.context.object
        camera.matrix_world = reference_camera["camera_matrix_world"]
        camera.data.sensor_fit = reference_camera["sensor_fit"]
        camera.data.angle = reference_camera["angle"]
        camera.data.shift_x = reference_camera["shift_x"]
        camera.data.shift_y = reference_camera["shift_y"]
        camera.data.clip_start = reference_camera["clip_start"]
        camera.data.clip_end = reference_camera["clip_end"]
        bpy.context.scene.camera = camera
        bpy.context.scene.render.resolution_x = int(reference_camera["resolution_x"])
        bpy.context.scene.render.resolution_y = int(reference_camera["resolution_y"])
        bpy.context.scene.render.resolution_percentage = int(reference_camera["resolution_percentage"])
        bpy.context.scene.render.pixel_aspect_x = float(reference_camera["pixel_aspect_x"])
        bpy.context.scene.render.pixel_aspect_y = float(reference_camera["pixel_aspect_y"])
        return

    if layout == "ground":
        center = layout_context["layout_center"]
        extent = float(layout_context["layout_extent"])
        if reference_camera is not None:
            reference_offset = reference_camera["camera_location"] - reference_camera["center"]
            scale = float(layout_context["layout_radius"]) / float(reference_camera["radius"])
            bpy.ops.object.camera_add(location=center + reference_offset * scale)
            camera = bpy.context.object
            camera.rotation_euler = reference_camera["camera_rotation"]
            camera.data.sensor_fit = reference_camera["sensor_fit"]
            camera.data.angle = reference_camera["angle"]
            camera.data.shift_x = reference_camera["shift_x"]
            camera.data.shift_y = reference_camera["shift_y"]
            camera.data.clip_start = reference_camera["clip_start"]
            camera.data.clip_end = max(reference_camera["clip_end"] * max(1.0, scale), extent * 8.0)
            bpy.context.scene.camera = camera
            return
        distance = extent * 1.05
        height = max(2.5, extent * 0.38)
        bpy.ops.object.camera_add(location=(center.x + extent * 0.58, center.y - distance, center.z + height))
        camera = bpy.context.object
        camera.data.sensor_fit = "HORIZONTAL"
        camera.data.angle = math.radians(70.0)
        camera.data.clip_end = max(100.0, extent * 8.0)
        look_at(camera, center)
        bpy.context.scene.camera = camera
        return

    camera_info = report["camera"]
    fov_degrees = float(camera_info.get("fov_degrees", 70.0))
    sensor_fit = str(camera_info.get("sensor_fit", "horizontal")).upper()
    far_depth = float(camera_info["far_depth"])
    bpy.ops.object.camera_add(location=(0.0, 0.0, 0.0), rotation=(math.radians(90.0), 0.0, 0.0))
    camera = bpy.context.object
    camera.data.sensor_fit = sensor_fit if sensor_fit in {"HORIZONTAL", "VERTICAL"} else "HORIZONTAL"
    camera.data.angle = math.radians(fov_degrees)
    camera.data.shift_x = float(camera_info.get("shift_x", 0.0))
    camera.data.shift_y = float(camera_info.get("shift_y", 0.0))
    camera.data.clip_end = far_depth * 4.0
    bpy.context.scene.camera = camera


def export_scene(fits_path: Path, output_path: Path, layout: str, reference_blend_path: Path | None) -> None:
    report = json.loads(fits_path.read_text(encoding="utf-8"))
    if layout == "original-camera" and reference_blend_path is None:
        raise ValueError("--reference-blend is required for --layout original-camera")
    reference_camera = load_reference_camera(reference_blend_path)
    clear_scene()
    bpy.context.scene["sceneforge_export_layout"] = layout
    bpy.context.scene["sceneforge_coordinate_contract"] = json.dumps(
        report.get("camera", {}).get("fusion_contract", {}),
        sort_keys=True,
    )
    materials: dict[str, bpy.types.Material] = {}
    layout_context = make_ground_layout_context(report["objects"])

    for item in report["objects"]:
        label = item["primitive_label"]
        obj = add_primitive(label)
        obj.name = f"{int(item['id']):02d}_{label}_{float(item['confidence']):.2f}"
        if label not in materials:
            materials[label] = make_material(label)
        obj.data.materials.append(materials[label])
        if layout == "original-camera":
            apply_original_camera_transform(obj, item, reference_camera)
        elif layout == "ground":
            apply_ground_transform(obj, item, layout_context)
        else:
            apply_camera_transform(obj, item)

    setup_camera(report, layout, layout_context, reference_camera)
    if layout != "original-camera":
        bpy.context.scene.render.resolution_x = int(report["image_width"])
        bpy.context.scene.render.resolution_y = int(report["image_height"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path))


if __name__ == "__main__":
    if len(sys.argv) == 1:
        from Runtime.guided_cli import guided_blender_tool_main

        raise SystemExit(
            guided_blender_tool_main(
                Path(__file__),
                "Export fitted primitive JSON to a Blender scene.",
                ["--fits", "Output/Latest/fit/primitive_fits.json", "--output", "Output/Latest/fitted_scene.blend"],
            )
        )
    args = parse_args()
    export_scene(
        Path(args.fits),
        Path(args.output),
        args.layout,
        Path(args.reference_blend) if args.reference_blend else None,
    )
