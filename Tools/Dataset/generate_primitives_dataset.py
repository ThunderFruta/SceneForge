from __future__ import annotations

import argparse
import colorsys
import json
import math
import random
import shutil
import sys
import tempfile
from math import radians
from pathlib import Path

try:
    import bpy
except ModuleNotFoundError:
    bpy = None
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Tools.Dataset.rgbd_curriculum import (  # noqa: E402
    BASE_CLASSES,
    DATASET_SPLITS,
    EXTENDED_CLASSES,
    split_name_for_index,
    split_path,
    stage_for_id,
    write_rgbd_data_yaml,
    write_stage_manifest,
)


CLASSES = BASE_CLASSES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic YOLO segmentation dataset of simple primitives."
    )
    parser.add_argument("--output", default="Datasets/PrimitiveShapes")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--fov-degrees", type=float, default=70.0)
    parser.add_argument("--render-samples", type=int, default=16)
    parser.add_argument("--train-split", type=float, default=0.70)
    parser.add_argument("--val-split", type=float, default=0.20)
    parser.add_argument(
        "--dark-background-ratio",
        type=float,
        default=0.30,
        help="Fraction of renders that use a dark world background.",
    )
    parser.add_argument("--min-shapes", type=int, default=1)
    parser.add_argument("--max-shapes", type=int, default=3)
    parser.add_argument(
        "--curriculum-stage",
        type=int,
        choices=tuple(range(1, 11)),
        help="Use one of the RGBD curriculum presets. Writes a stage-specific RGBD dataset layout.",
    )
    parser.add_argument(
        "--images-per-class",
        type=int,
        help="For curriculum stage 1, derive --count from class count times this value.",
    )
    parser.add_argument(
        "--write-rgbd",
        action="store_true",
        help="Write RGB, depth, RGBA RGBD, masks, and RGBD data YAML outputs.",
    )
    parser.add_argument(
        "--depth-near",
        type=float,
        default=1.0,
        help="Compositor depth value mapped to white when --write-rgbd is used.",
    )
    parser.add_argument(
        "--depth-far",
        type=float,
        default=8.0,
        help="Compositor depth value mapped to black when --write-rgbd is used.",
    )
    parser.add_argument(
        "--shape-scale-min",
        type=float,
        default=1.05,
        help="Minimum primitive scale multiplier.",
    )
    parser.add_argument(
        "--shape-scale-max",
        type=float,
        default=1.55,
        help="Maximum primitive scale multiplier.",
    )
    parser.add_argument(
        "--min-screen-area-ratio",
        type=float,
        default=0.018,
        help="Minimum projected object area as a fraction of image area for labels.",
    )
    parser.add_argument(
        "--max-screen-overlap-ratio",
        type=float,
        default=0.08,
        help="Maximum allowed projected bbox overlap for non-occluded placement.",
    )
    parser.add_argument(
        "--write-instance-masks",
        action="store_true",
        help="Also render one visible silhouette mask per object for higher-quality YOLO labels.",
    )
    parser.add_argument(
        "--mask-subdir",
        default="masks",
        help="Dataset subdirectory for instance masks when --write-instance-masks is used.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard index for parallel generation.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Total shard count for parallel generation.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First global sample index to consider, inclusive.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        help="Last global sample index to consider, exclusive. Defaults to --count.",
    )
    parser.add_argument(
        "--finish",
        action="store_true",
        help="Skip already-complete samples and only render missing indices.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Log progress every N generated samples within each shard (0 disables per-sample logs).",
    )
    argv = sys.argv
    script_args = argv[argv.index("--") + 1 :] if "--" in argv else []
    args = parser.parse_args(script_args)
    if args.count < 1:
        raise ValueError("--count must be at least 1")
    if args.min_shapes < 1 or args.max_shapes < args.min_shapes:
        raise ValueError("--min-shapes and --max-shapes must describe a valid range")
    if args.train_split <= 0.0 or args.train_split >= 1.0:
        raise ValueError("--train-split must be between 0 and 1")
    if args.val_split <= 0.0 or args.val_split >= 1.0:
        raise ValueError("--val-split must be between 0 and 1")
    if args.train_split + args.val_split >= 1.0:
        raise ValueError("--train-split plus --val-split must leave room for a test split")
    if args.fov_degrees <= 0.0 or args.fov_degrees >= 179.0:
        raise ValueError("--fov-degrees must be between 0 and 179")
    if args.render_samples < 1:
        raise ValueError("--render-samples must be at least 1")
    if args.dark_background_ratio < 0.0 or args.dark_background_ratio > 1.0:
        raise ValueError("--dark-background-ratio must be between 0 and 1")
    if args.images_per_class is not None and args.images_per_class < 1:
        raise ValueError("--images-per-class must be at least 1")
    if args.depth_near <= 0.0 or args.depth_far <= args.depth_near:
        raise ValueError("--depth-near and --depth-far must satisfy 0 < near < far")
    if args.shape_scale_min <= 0.0 or args.shape_scale_max < args.shape_scale_min:
        raise ValueError("--shape-scale-min and --shape-scale-max must describe a valid positive range")
    if args.min_screen_area_ratio < 0.0 or args.min_screen_area_ratio >= 1.0:
        raise ValueError("--min-screen-area-ratio must be between 0 and 1")
    if args.max_screen_overlap_ratio < 0.0 or args.max_screen_overlap_ratio > 1.0:
        raise ValueError("--max-screen-overlap-ratio must be between 0 and 1")
    if args.log_every < 0:
        raise ValueError("--log-every must be greater than or equal to 0")
    if args.shard_count < 1:
        raise ValueError("--shard-count must be at least 1")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < shard-count")
    if args.curriculum_stage:
        stage = stage_for_id(args.curriculum_stage)
        args.min_shapes = stage.min_objects
        args.max_shapes = stage.max_objects
        args.shape_scale_min = stage.scale_min
        args.shape_scale_max = stage.scale_max
        args.min_screen_area_ratio = stage.min_screen_area_ratio
        args.max_screen_overlap_ratio = stage.max_screen_overlap_ratio
        args.write_rgbd = True
        args.write_instance_masks = True
        if args.images_per_class is not None and stage.id == 1:
            args.count = args.images_per_class * len(stage.classes)
    if args.start_index < 0 or args.start_index >= args.count:
        raise ValueError("--start-index must satisfy 0 <= start-index < count")
    if args.end_index is None:
        args.end_index = args.count
    if args.end_index <= args.start_index or args.end_index > args.count:
        raise ValueError("--end-index must satisfy start-index < end-index <= count")
    return args


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def make_material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = 0.55
    return material


def make_emission_material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    for node in list(nodes):
        nodes.remove(node)
    emission = nodes.new(type="ShaderNodeEmission")
    output = nodes.new(type="ShaderNodeOutputMaterial")
    emission.inputs["Color"].default_value = color
    emission.inputs["Strength"].default_value = 1.0
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def random_material_color(rng: random.Random) -> tuple[float, float, float, float]:
    hue = rng.random()
    saturation = rng.uniform(0.35, 0.95)
    if rng.random() < 0.70:
        value = rng.uniform(0.12, 0.55)
    else:
        value = rng.uniform(0.55, 0.82)
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return (red, green, blue, 1.0)


def set_origin_camera(camera: bpy.types.Object, target: Vector) -> None:
    direction = target - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def setup_scene(
    width: int,
    height: int,
    fov_degrees: float,
    render_samples: int,
    dark_background_ratio: float,
    rng: random.Random,
) -> bpy.types.Object:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = render_samples
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.world = bpy.data.worlds.new("World")
    if rng.random() < dark_background_ratio:
        scene.world.color = (
            rng.uniform(0.015, 0.16),
            rng.uniform(0.015, 0.16),
            rng.uniform(0.015, 0.16),
        )
    else:
        scene.world.color = (
            rng.uniform(0.45, 0.75),
            rng.uniform(0.45, 0.75),
            rng.uniform(0.45, 0.75),
        )

    bpy.ops.object.light_add(type="AREA", location=(rng.uniform(-3, 3), -4, 6))
    key = bpy.context.object
    key.name = "KeyLight"
    if rng.random() < 0.45:
        key.data.energy = rng.uniform(180, 480)
    else:
        key.data.energy = rng.uniform(480, 820)
    key.data.size = rng.uniform(4, 7)

    bpy.ops.object.camera_add(
        location=(
            rng.uniform(-0.6, 0.6),
            rng.uniform(-7.0, -6.2),
            rng.uniform(2.4, 3.2),
        )
    )
    camera = bpy.context.object
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.angle = radians(fov_degrees)
    set_origin_camera(camera, Vector((0, 0, 0.75)))
    scene.camera = camera
    return camera


def add_primitive(
    kind: str,
    location: tuple[float, float, float],
    rng: random.Random,
    scale_min: float,
    scale_max: float,
) -> bpy.types.Object:
    color = random_material_color(rng)
    material = make_material(f"{kind}_material", color)
    scale = rng.uniform(scale_min, scale_max)

    if kind == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=0.75 * scale, location=location)
    elif kind == "box":
        bpy.ops.mesh.primitive_cube_add(size=1.25 * scale, location=location)
    elif kind == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.55 * scale, depth=1.5 * scale, location=location)
    elif kind == "cone":
        bpy.ops.mesh.primitive_cone_add(vertices=32, radius1=0.75 * scale, radius2=0.0, depth=1.7 * scale, location=location)
    elif kind == "plane":
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
        obj = bpy.context.object
        obj.dimensions = (1.65 * scale, 1.2 * scale, 0.055 * scale)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    elif kind == "torus":
        bpy.ops.mesh.primitive_torus_add(
            major_segments=48,
            minor_segments=16,
            major_radius=0.48 * scale,
            minor_radius=0.15 * scale,
            location=location,
        )
    elif kind == "tube":
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=48,
            radius=0.62 * scale,
            depth=1.45 * scale,
            end_fill_type="NOTHING",
            location=location,
        )
        obj = bpy.context.object
        solidify = obj.modifiers.new("tube_wall", "SOLIDIFY")
        solidify.thickness = 0.12 * scale
        bpy.ops.object.modifier_apply(modifier=solidify.name)
    elif kind == "arch":
        add_arch_mesh(scale, location)
    else:
        raise ValueError(f"Unknown primitive kind: {kind}")

    obj = bpy.context.object
    obj.name = kind
    obj.data.materials.append(material)
    if kind in {"sphere", "cylinder", "cone", "torus", "tube", "arch"}:
        bpy.ops.object.shade_smooth()

    if kind == "plane":
        obj.rotation_euler = (
            rng.uniform(-0.22, 0.22),
            rng.uniform(-0.18, 0.18),
            rng.uniform(-0.7, 0.7),
        )
    elif kind in {"torus", "tube", "arch"}:
        obj.rotation_euler = (
            rng.uniform(-0.35, 0.35),
            rng.uniform(-0.85, 0.85),
            rng.uniform(-0.75, 0.75),
        )
    else:
        obj.rotation_euler = (
            rng.uniform(-0.12, 0.12),
            rng.uniform(-0.35, 0.35),
            rng.uniform(-0.28, 0.28),
        )
    return obj


def add_arch_mesh(scale: float, location: tuple[float, float, float]) -> None:
    major_radius = 0.62 * scale
    minor_radius = 0.14 * scale
    arc_steps = 32
    tube_steps = 12
    vertices = []
    faces = []
    for arc_index in range(arc_steps + 1):
        theta = 3.141592653589793 * arc_index / arc_steps
        center = Vector((major_radius * (theta - 1.5707963267948966) / 1.5707963267948966, 0.0, major_radius * abs(math.sin(theta))))
        tangent_angle = theta
        for tube_index in range(tube_steps):
            phi = 2.0 * 3.141592653589793 * tube_index / tube_steps
            x = center.x + minor_radius * math.cos(phi) * math.cos(tangent_angle)
            y = center.y + minor_radius * math.sin(phi)
            z = center.z + minor_radius * math.cos(phi) * math.sin(tangent_angle)
            vertices.append((x, y, z))
    for arc_index in range(arc_steps):
        row = arc_index * tube_steps
        next_row = (arc_index + 1) * tube_steps
        for tube_index in range(tube_steps):
            faces.append(
                (
                    row + tube_index,
                    row + ((tube_index + 1) % tube_steps),
                    next_row + ((tube_index + 1) % tube_steps),
                    next_row + tube_index,
                )
            )
    mesh = bpy.data.meshes.new("arch_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("arch", mesh)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)


def choose_shape_kinds(index: int, count: int, rng: random.Random, classes: tuple[str, ...]) -> list[str]:
    first_kind = classes[index % len(classes)]
    remaining = [rng.choice(classes) for _ in range(max(0, count - 1))]
    kinds = [first_kind, *remaining]
    rng.shuffle(kinds)
    return kinds


def object_location(shape_index: int, shape_count: int, allow_occlusion: bool, rng: random.Random) -> tuple[float, float, float]:
    if allow_occlusion:
        x = rng.uniform(-2.4, 2.4)
        y = rng.uniform(-1.0, 1.0)
        z = rng.uniform(-0.35, 0.65)
        return x, y, z

    columns = max(1, min(shape_count, 4))
    rows = max(1, math.ceil(shape_count / columns))
    column = shape_index % columns
    row = shape_index // columns
    x_extent = 8.4 if shape_count <= 4 else 9.6
    z_extent = 3.0 if rows <= 2 else 3.8
    x_spacing = x_extent / max(1, columns - 1) if columns > 1 else 0.0
    z_spacing = z_extent / max(1, rows - 1) if rows > 1 else 0.0
    x = -x_extent * 0.5 + column * x_spacing if columns > 1 else 0.0
    z = -0.75 + row * z_spacing if rows > 1 else 0.15
    return (
        x + rng.uniform(-0.20, 0.20),
        rng.uniform(-0.18, 0.18),
        z + rng.uniform(-0.12, 0.12),
    )


def hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = sorted(set(points))
    if len(unique) <= 2:
        return unique

    def cross(origin, point_a, point_b) -> float:
        return (point_a[0] - origin[0]) * (point_b[1] - origin[1]) - (
            point_a[1] - origin[1]
        ) * (point_b[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def projected_polygon(obj: bpy.types.Object, camera: bpy.types.Object) -> list[tuple[float, float]]:
    scene = bpy.context.scene
    coords: list[tuple[float, float]] = []
    for vertex in obj.data.vertices:
        world = obj.matrix_world @ vertex.co
        projected = world_to_camera_view(scene, camera, world)
        if projected.z < 0:
            continue
        x = max(0.0, min(1.0, projected.x))
        y = max(0.0, min(1.0, 1.0 - projected.y))
        coords.append((round(x, 6), round(y, 6)))

    polygon = hull(coords)
    if len(polygon) < 3:
        return []
    return polygon


def polygon_area_ratio(polygon: list[tuple[float, float]]) -> float:
    if len(polygon) < 3:
        return 0.0
    area = 0.0
    for index, point in enumerate(polygon):
        next_point = polygon[(index + 1) % len(polygon)]
        area += point[0] * next_point[1] - next_point[0] * point[1]
    return abs(area) * 0.5


def polygon_bbox(polygon: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def bbox_area(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def bbox_overlap_ratio(
    bbox_a: tuple[float, float, float, float],
    bbox_b: tuple[float, float, float, float],
) -> float:
    left = max(bbox_a[0], bbox_b[0])
    top = max(bbox_a[1], bbox_b[1])
    right = min(bbox_a[2], bbox_b[2])
    bottom = min(bbox_a[3], bbox_b[3])
    intersection = bbox_area((left, top, right, bottom))
    if intersection <= 0.0:
        return 0.0
    smaller_area = min(bbox_area(bbox_a), bbox_area(bbox_b))
    if smaller_area <= 0.0:
        return 0.0
    return intersection / smaller_area


def max_polygon_overlap_ratio(
    polygon: list[tuple[float, float]],
    existing_polygons: list[list[tuple[float, float]]],
) -> float:
    bbox = polygon_bbox(polygon)
    if not existing_polygons:
        return 0.0
    return max(bbox_overlap_ratio(bbox, polygon_bbox(existing)) for existing in existing_polygons)


def remove_object(obj: bpy.types.Object) -> None:
    mesh = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    if mesh.users == 0:
        bpy.data.meshes.remove(mesh)


def add_visible_primitive(
    kind: str,
    shape_index: int,
    shape_count: int,
    allow_occlusion: bool,
    rng: random.Random,
    camera: bpy.types.Object,
    args: argparse.Namespace,
    existing_polygons: list[list[tuple[float, float]]],
) -> tuple[bpy.types.Object, list[tuple[float, float]]]:
    best_obj: bpy.types.Object | None = None
    best_polygon: list[tuple[float, float]] = []
    best_score = float("inf")
    max_attempts = 80

    for _attempt in range(max_attempts):
        obj = add_primitive(
            kind,
            object_location(shape_index, shape_count, allow_occlusion, rng),
            rng,
            args.shape_scale_min,
            args.shape_scale_max,
        )
        polygon = projected_polygon(obj, camera)
        area = polygon_area_ratio(polygon)
        overlap = max_polygon_overlap_ratio(polygon, existing_polygons) if polygon else 1.0
        visible_penalty = max(0.0, args.min_screen_area_ratio - area) * 10.0
        score = overlap + visible_penalty

        if area >= args.min_screen_area_ratio and overlap <= args.max_screen_overlap_ratio:
            if best_obj is not None:
                remove_object(best_obj)
            return obj, polygon

        if score < best_score:
            if best_obj is not None:
                remove_object(best_obj)
            best_obj = obj
            best_polygon = polygon
            best_score = score
        else:
            remove_object(obj)

    if best_obj is None:
        raise RuntimeError(f"Could not place primitive: {kind}")
    return best_obj, best_polygon


def write_yolo_labels(path: Path, labeled_objects: list[tuple[str, list[tuple[float, float]]]], classes: tuple[str, ...]) -> None:
    lines: list[str] = []
    for class_name, polygon in labeled_objects:
        class_id = classes.index(class_name)
        flattened = " ".join(f"{value:.6f}" for point in polygon for value in point)
        lines.append(f"{class_id} {flattened}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_data_yaml(root: Path, classes: tuple[str, ...]) -> None:
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(classes))
    (root / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {root.resolve()}",
                "train: train/images",
                "val: val/images",
                "test: test/images",
                f"nc: {len(classes)}",
                "names:",
                names,
                "",
            ]
        ),
        encoding="utf-8",
    )


def replace_object_material(obj: bpy.types.Object, material: bpy.types.Material) -> None:
    obj.data.materials.clear()
    obj.data.materials.append(material)


def object_id_color(index: int) -> tuple[float, float, float, float]:
    color_id = index + 1
    red = ((color_id * 53) % 255) / 255.0
    green = ((color_id * 97) % 255) / 255.0
    blue = ((color_id * 193) % 255) / 255.0
    return red, green, blue, 1.0


def write_binary_mask(path: Path, width: int, height: int, values: list[float]) -> None:
    image = bpy.data.images.new(path.stem, width=width, height=height, alpha=True)
    image.pixels.foreach_set(values)
    image.filepath_raw = str(path)
    image.file_format = "PNG"
    image.save()
    bpy.data.images.remove(image)


def render_instance_masks(
    mask_dir: Path,
    stem: str,
    objects: list[tuple[str, bpy.types.Object]],
) -> None:
    mask_dir.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    original_filepath = scene.render.filepath
    original_world_color = tuple(scene.world.color)
    original_samples = scene.eevee.taa_render_samples
    original_view_transform = scene.view_settings.view_transform
    original_look = scene.view_settings.look
    original_materials = [(obj, list(obj.data.materials)) for _, obj in objects]

    black = make_emission_material(f"{stem}_mask_black", (0.0, 0.0, 0.0, 1.0))
    id_materials = [
        make_emission_material(f"{stem}_mask_id_{index:02d}", object_id_color(index))
        for index, _object in enumerate(objects)
    ]
    scene.world.color = (0.0, 0.0, 0.0)
    scene.eevee.taa_render_samples = 1
    scene.view_settings.view_transform = "Raw"
    scene.view_settings.look = "None"

    with tempfile.TemporaryDirectory() as temp_dir_name:
        id_path = Path(temp_dir_name) / f"{stem}_instance_ids.png"
        try:
            for object_index, (_kind, obj) in enumerate(objects):
                replace_object_material(obj, id_materials[object_index])
            scene.render.filepath = str(id_path)
            bpy.ops.render.render(write_still=True)

            id_image = bpy.data.images.load(str(id_path))
            width, height = id_image.size
            pixels = list(id_image.pixels)
            target_colors = [object_id_color(index) for index, _object in enumerate(objects)]
            mask_pixels = [
                [0.0 for _ in range(width * height * 4)]
                for _object in objects
            ]
            threshold = 0.18

            for pixel_index in range(width * height):
                offset = pixel_index * 4
                red = pixels[offset]
                green = pixels[offset + 1]
                blue = pixels[offset + 2]
                best_index = -1
                best_distance = float("inf")
                for object_index, color in enumerate(target_colors):
                    distance = (
                        (red - color[0]) * (red - color[0])
                        + (green - color[1]) * (green - color[1])
                        + (blue - color[2]) * (blue - color[2])
                    )
                    if distance < best_distance:
                        best_index = object_index
                        best_distance = distance
                if best_index >= 0 and best_distance <= threshold * threshold:
                    mask = mask_pixels[best_index]
                    mask[offset] = 1.0
                    mask[offset + 1] = 1.0
                    mask[offset + 2] = 1.0
                    mask[offset + 3] = 1.0

            for object_index, (kind, _obj) in enumerate(objects):
                write_binary_mask(
                    mask_dir / f"{stem}_{object_index:02d}_{kind}.png",
                    width,
                    height,
                    mask_pixels[object_index],
                )
            bpy.data.images.remove(id_image)
        finally:
            for obj, materials in original_materials:
                obj.data.materials.clear()
                for material in materials:
                    obj.data.materials.append(material)
            scene.world.color = original_world_color
            scene.eevee.taa_render_samples = original_samples
            scene.view_settings.view_transform = original_view_transform
            scene.view_settings.look = original_look
            scene.render.filepath = original_filepath


def render_depth_and_rgbd(
    depth_path: Path,
    rgbd_path: Path,
    near_depth: float,
    far_depth: float,
) -> None:
    scene = bpy.context.scene
    scene.view_layers[0].use_pass_z = True
    original_use_nodes = scene.use_nodes

    depth_path.parent.mkdir(parents=True, exist_ok=True)
    rgbd_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)
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

        set_alpha = tree.nodes.new(type="CompositorNodeSetAlpha")
        rgbd_output = tree.nodes.new(type="CompositorNodeOutputFile")
        rgbd_output.base_path = str(temp_dir)
        rgbd_output.file_slots[0].path = "rgbd_"
        rgbd_output.format.file_format = "PNG"
        rgbd_output.format.color_mode = "RGBA"
        rgbd_output.format.color_depth = "8"

        depth_output = tree.nodes.new(type="CompositorNodeOutputFile")
        depth_output.base_path = str(temp_dir)
        depth_output.file_slots[0].path = "depth_"
        depth_output.format.file_format = "PNG"
        depth_output.format.color_mode = "BW"
        depth_output.format.color_depth = "8"

        tree.links.new(render_layers.outputs["Depth"], map_range.inputs["Value"])
        tree.links.new(render_layers.outputs["Image"], set_alpha.inputs["Image"])
        tree.links.new(map_range.outputs["Value"], set_alpha.inputs["Alpha"])
        tree.links.new(set_alpha.outputs["Image"], rgbd_output.inputs[0])
        tree.links.new(map_range.outputs["Value"], depth_output.inputs[0])

        bpy.ops.render.render(write_still=False)
        rgbd_files = sorted(temp_dir.glob("rgbd_*.png"))
        depth_files = sorted(temp_dir.glob("depth_*.png"))
        if not rgbd_files or not depth_files:
            raise RuntimeError("RGBD compositor did not write expected PNG outputs.")
        shutil.copyfile(rgbd_files[0], rgbd_path)
        shutil.copyfile(depth_files[0], depth_path)

    scene.node_tree.nodes.clear()
    scene.use_nodes = original_use_nodes


def looks_like_complete_png(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 16:
        return False
    with path.open("rb") as file:
        file.seek(max(0, path.stat().st_size - 32))
        tail = file.read()
    return b"IEND" in tail


def sample_complete(root: Path, split: str, stem: str, args: argparse.Namespace) -> bool:
    required = [
        split_path(root, split, "images") / f"{stem}.png",
        split_path(root, split, "labels") / f"{stem}.txt",
    ]
    if args.write_rgbd:
        required.extend(
            [
                split_path(root, split, "images_rgb") / f"{stem}.png",
                split_path(root, split, "images_rgbd") / f"{stem}.png",
                split_path(root, split, "depth") / f"{stem}.png",
            ]
        )
    if any(not looks_like_complete_png(path) for path in required if path.suffix.lower() == ".png"):
        return False
    if any(not path.is_file() for path in required if path.suffix.lower() != ".png"):
        return False
    if args.write_instance_masks:
        return any(looks_like_complete_png(path) for path in split_path(root, split, args.mask_subdir).glob(f"{stem}_*.png"))
    return True


def generate_dataset(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    stage = stage_for_id(args.curriculum_stage) if args.curriculum_stage else None
    classes = stage.classes if stage else CLASSES
    root = Path(args.output)
    if stage:
        root = root / f"stage{stage.id}_{stage.name}"

    for split in DATASET_SPLITS:
        split_path(root, split, "images").mkdir(parents=True, exist_ok=True)
        if args.write_rgbd:
            split_path(root, split, "images_rgb").mkdir(parents=True, exist_ok=True)
            split_path(root, split, "images_rgbd").mkdir(parents=True, exist_ok=True)
            split_path(root, split, "depth").mkdir(parents=True, exist_ok=True)
        split_path(root, split, "labels").mkdir(parents=True, exist_ok=True)
        if args.write_instance_masks:
            split_path(root, split, args.mask_subdir).mkdir(parents=True, exist_ok=True)
    write_data_yaml(root, classes)
    if args.write_rgbd:
        write_rgbd_data_yaml(root, classes)
    if stage:
        write_stage_manifest(root, stage)

    class_counts = {class_name: 0 for class_name in classes}
    split_counts = {split: 0 for split in DATASET_SPLITS}
    object_counts: list[int] = []
    skipped_existing = 0

    shard_total = sum(1 for index in range(args.start_index, args.end_index) if index % args.shard_count == args.shard_index)
    shard_generated = 0
    for index in range(args.start_index, args.end_index):
        if index % args.shard_count != args.shard_index:
            continue
        split = split_name_for_index(index, args.count, args.train_split, args.val_split)
        stem = f"primitive_{index:04d}"
        if args.finish and sample_complete(root, split, stem, args):
            skipped_existing += 1
            continue
        shard_generated += 1
        clear_scene()
        camera = setup_scene(
            args.width,
            args.height,
            args.fov_degrees,
            args.render_samples,
            args.dark_background_ratio,
            rng,
        )
        shape_count = rng.randint(args.min_shapes, args.max_shapes)
        objects: list[tuple[str, bpy.types.Object]] = []
        object_polygons: list[tuple[str, list[tuple[float, float]]]] = []
        for shape_index, kind in enumerate(choose_shape_kinds(index, shape_count, rng, classes)):
            allow_occlusion = bool(stage.allow_occlusion) if stage else shape_count > 3
            obj, polygon = add_visible_primitive(
                kind,
                shape_index,
                shape_count,
                allow_occlusion,
                rng,
                camera,
                args,
                [existing_polygon for _, existing_polygon in object_polygons],
            )
            objects.append((kind, obj))
            object_polygons.append((kind, polygon))

        labeled_objects = [
            (kind, polygon)
            for kind, polygon in object_polygons
            if polygon and polygon_area_ratio(polygon) >= args.min_screen_area_ratio
        ]

        image_path = split_path(root, split, "images") / f"{stem}.png"
        label_path = split_path(root, split, "labels") / f"{stem}.txt"
        bpy.context.scene.render.filepath = str(image_path)
        bpy.ops.render.render(write_still=True)
        write_yolo_labels(label_path, labeled_objects, classes)
        split_counts[split] += 1
        object_counts.append(len(labeled_objects))
        for kind, _polygon in labeled_objects:
            class_counts[kind] += 1
        if args.write_rgbd:
            rgb_path = split_path(root, split, "images_rgb") / f"{stem}.png"
            shutil.copyfile(image_path, rgb_path)
            depth_path = split_path(root, split, "depth") / f"{stem}.png"
            rgbd_path = split_path(root, split, "images_rgbd") / f"{stem}.png"
            render_depth_and_rgbd(
                depth_path=depth_path,
                rgbd_path=rgbd_path,
                near_depth=args.depth_near,
                far_depth=args.depth_far,
            )
            shutil.copyfile(rgbd_path, image_path)
        if args.write_instance_masks:
            render_instance_masks(split_path(root, split, args.mask_subdir), stem, objects)
        shard_prefix = (
            f"shard {args.shard_index + 1}/{args.shard_count} "
            if args.shard_count > 1
            else ""
        )
        if args.log_every == 0:
            continue
        if shard_generated % args.log_every != 0 and shard_generated != shard_total:
            continue
        print(
            f"{shard_prefix}{shard_generated:04d}/{shard_total:04d} "
            f"global {index + 1:04d}/{args.count}: {image_path} ({len(labeled_objects)} labels)"
        )

    validation = {
        "curriculum_stage": stage.id if stage else None,
        "classes": list(classes),
        "class_counts": class_counts,
        "split_counts": split_counts,
        "object_count_min": min(object_counts) if object_counts else 0,
        "object_count_max": max(object_counts) if object_counts else 0,
        "min_screen_area_ratio": args.min_screen_area_ratio,
        "max_screen_overlap_ratio": args.max_screen_overlap_ratio,
        "shape_scale_min": args.shape_scale_min,
        "shape_scale_max": args.shape_scale_max,
        "write_rgbd": bool(args.write_rgbd),
        "render_samples": args.render_samples,
        "dark_background_ratio": args.dark_background_ratio,
        "mask_render_mode": "single_pass_color_id",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "start_index": args.start_index,
        "end_index": args.end_index,
        "shard_generated": shard_generated,
        "skipped_existing": skipped_existing,
    }
    print(
        f"Shard {args.shard_index + 1}/{args.shard_count} complete: "
        f"generated {shard_generated}/{shard_total} samples, skipped {skipped_existing} existing."
    )
    validation_dir = root / "validation_reports"
    validation_dir.mkdir(parents=True, exist_ok=True)
    if args.shard_count == 1 and args.start_index == 0 and args.end_index == args.count:
        validation_path = root / "validation_report.json"
    else:
        validation_path = validation_dir / (
            f"validation_report_shard_{args.shard_index:02d}_of_{args.shard_count:02d}"
            f"_range_{args.start_index:06d}_{args.end_index:06d}.json"
        )
    validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        from Runtime.guided_cli import guided_blender_tool_main

        raise SystemExit(
            guided_blender_tool_main(
                Path(__file__),
                'Generate synthetic primitive RGBD dataset.',
                [],
                blend_path=None,
            )
        )
    generate_dataset(parse_args())
