from __future__ import annotations

import argparse
import colorsys
import json
import math
import random
import shutil
import sys
from pathlib import Path

try:
    import bpy
except ModuleNotFoundError:
    bpy = None
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Tools.Dataset.generate_primitives_dataset import render_depth_and_rgbd, render_instance_masks  # noqa: E402
from Tools.Dataset.rgbd_curriculum import BASE_CLASSES, DATASET_SPLITS, split_name_for_index, split_path, write_rgbd_data_yaml  # noqa: E402


PRIMITIVE_CLASSES = ("box", "sphere", "cylinder", "cone")
OCCLUSION_BUCKETS = ("none", "mild", "medium", "hard")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate balanced no-plane RGBD primitive detection data.")
    parser.add_argument("--output", default="Datasets/PrimitiveShapesRGBDTarget/no_plane_hard")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--render-samples", type=int, default=16)
    parser.add_argument("--train-split", type=float, default=0.70)
    parser.add_argument("--val-split", type=float, default=0.20)
    parser.add_argument("--depth-near", type=float, default=1.0)
    parser.add_argument("--depth-far", type=float, default=12.0)
    parser.add_argument("--min-objects", type=int, default=4)
    parser.add_argument("--max-objects", type=int, default=7)
    parser.add_argument("--occlusion-buckets", default="none,mild,medium,hard")
    parser.add_argument("--dark-background-ratio", type=float, default=0.60)
    parser.add_argument("--material-variation", type=float, default=0.90)
    parser.add_argument("--camera-distance-min", type=float, default=5.0)
    parser.add_argument("--camera-distance-max", type=float, default=8.0)
    parser.add_argument("--camera-height-min", type=float, default=2.2)
    parser.add_argument("--camera-height-max", type=float, default=4.8)
    parser.add_argument("--fov-degrees-min", type=float, default=48.0)
    parser.add_argument("--fov-degrees-max", type=float, default=74.0)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--finish", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    argv = sys.argv
    script_args = argv[argv.index("--") + 1 :] if "--" in argv else []
    args = parser.parse_args(script_args)
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.count < 1:
        raise ValueError("--count must be at least 1")
    if args.min_objects < 1 or args.max_objects < args.min_objects:
        raise ValueError("--min-objects/--max-objects must satisfy 1 <= min <= max")
    if args.train_split <= 0.0 or args.val_split <= 0.0 or args.train_split + args.val_split >= 1.0:
        raise ValueError("--train-split and --val-split must leave a test split")
    if args.depth_near <= 0.0 or args.depth_far <= args.depth_near:
        raise ValueError("--depth-near and --depth-far must satisfy 0 < near < far")
    if args.shard_count < 1:
        raise ValueError("--shard-count must be at least 1")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < shard-count")
    if args.start_index < 0 or args.start_index >= args.count:
        raise ValueError("--start-index must satisfy 0 <= start-index < count")
    if args.end_index is None:
        args.end_index = args.count
    if args.end_index <= args.start_index or args.end_index > args.count:
        raise ValueError("--end-index must satisfy start-index < end-index <= count")
    buckets = tuple(item.strip() for item in args.occlusion_buckets.split(",") if item.strip())
    if not buckets or any(item not in OCCLUSION_BUCKETS for item in buckets):
        raise ValueError(f"--occlusion-buckets must use: {', '.join(OCCLUSION_BUCKETS)}")
    args.occlusion_buckets = buckets


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def configure_scene(args: argparse.Namespace) -> None:
    scene = bpy.context.scene
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = args.render_samples
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")


def make_material(name: str, color: tuple[float, float, float, float], roughness: float) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = roughness
    return material


def random_color(rng: random.Random, variation: float) -> tuple[float, float, float, float]:
    hue = rng.random()
    saturation = rng.uniform(0.25, min(1.0, 0.40 + variation * 0.60))
    value = rng.uniform(0.22, 0.95)
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return red, green, blue, 1.0


def add_primitive(
    class_name: str,
    object_index: int,
    location: tuple[float, float, float],
    scale: tuple[float, float, float],
    rotation: tuple[float, float, float],
    material: bpy.types.Material,
) -> bpy.types.Object:
    if class_name == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=0.5, location=location, rotation=rotation)
    elif class_name == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=0.5, depth=1.0, location=location, rotation=rotation)
    elif class_name == "cone":
        bpy.ops.mesh.primitive_cone_add(vertices=48, radius1=0.5, radius2=0.0, depth=1.0, location=location, rotation=rotation)
    else:
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=location, rotation=rotation)
    obj = bpy.context.object
    obj.name = f"{object_index:02d}_{class_name}"
    obj.scale = scale
    obj.data.materials.append(material)
    return obj


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def place_camera(args: argparse.Namespace, rng: random.Random) -> None:
    distance = rng.uniform(args.camera_distance_min, args.camera_distance_max)
    theta = math.radians(rng.uniform(20.0, 160.0))
    height = rng.uniform(args.camera_height_min, args.camera_height_max)
    location = (math.cos(theta) * distance, -math.sin(theta) * distance, height)
    bpy.ops.object.camera_add(location=location)
    camera = bpy.context.object
    camera.name = "Camera"
    target = Vector((rng.uniform(-0.25, 0.25), rng.uniform(-0.25, 0.25), rng.uniform(0.55, 1.05)))
    look_at(camera, target)
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.angle = math.radians(rng.uniform(args.fov_degrees_min, args.fov_degrees_max))
    camera.data.clip_start = 0.05
    camera.data.clip_end = 100.0
    bpy.context.scene.camera = camera


def add_lighting(rng: random.Random) -> None:
    bpy.ops.object.light_add(type="AREA", location=(rng.uniform(-3.5, 3.5), rng.uniform(-5.5, -3.0), rng.uniform(4.0, 6.5)))
    key = bpy.context.object
    key.name = "Key_Area"
    key.data.energy = rng.uniform(350.0, 850.0)
    key.data.size = rng.uniform(3.5, 6.5)
    bpy.ops.object.light_add(type="POINT", location=(rng.uniform(1.5, 5.0), rng.uniform(1.5, 5.0), rng.uniform(2.0, 4.5)))
    fill = bpy.context.object
    fill.name = "Fill_Point"
    fill.data.energy = rng.uniform(35.0, 140.0)


def bucket_radius(bucket: str) -> float:
    if bucket == "none":
        return 1.65
    if bucket == "mild":
        return 1.20
    if bucket == "medium":
        return 0.82
    return 0.48


def sample_location(rng: random.Random, bucket: str, placed: list[tuple[float, float]], radius: float) -> tuple[float, float]:
    if bucket == "hard" and placed and rng.random() < 0.70:
        base_x, base_y = rng.choice(placed)
        return base_x + rng.uniform(-0.55, 0.55), base_y + rng.uniform(-0.45, 0.45)
    limit_x = 2.55 if bucket != "hard" else 2.1
    limit_y = 1.55 if bucket != "hard" else 1.15
    for _attempt in range(80):
        x = rng.uniform(-limit_x, limit_x)
        y = rng.uniform(-limit_y, limit_y)
        if all(math.hypot(x - px, y - py) >= radius for px, py in placed):
            return x, y
    return rng.uniform(-limit_x, limit_x), rng.uniform(-limit_y, limit_y)


def primitive_scale(class_name: str, rng: random.Random) -> tuple[float, float, float]:
    xy = rng.uniform(0.75, 1.30)
    if class_name == "sphere":
        diameter = rng.uniform(0.80, 1.45)
        return diameter, diameter, diameter
    if class_name == "box":
        return rng.uniform(0.80, 1.55), rng.uniform(0.70, 1.45), rng.uniform(0.75, 1.80)
    if class_name == "cylinder":
        return xy, xy, rng.uniform(0.75, 1.80)
    return xy, xy, rng.uniform(0.85, 1.85)


def primitive_rotation(class_name: str, rng: random.Random, bucket: str) -> tuple[float, float, float]:
    if class_name in {"cone", "cylinder"} or bucket in {"medium", "hard"}:
        return (
            math.radians(rng.uniform(-180.0, 180.0)),
            math.radians(rng.uniform(-180.0, 180.0)),
            math.radians(rng.uniform(-180.0, 180.0)),
        )
    return (
        math.radians(rng.uniform(-55.0, 55.0)),
        math.radians(rng.uniform(-55.0, 55.0)),
        math.radians(rng.uniform(-180.0, 180.0)),
    )


def class_for_object(global_object_index: int) -> str:
    return PRIMITIVE_CLASSES[global_object_index % len(PRIMITIVE_CLASSES)]


def build_sample(args: argparse.Namespace, sample_index: int, bucket: str) -> list[tuple[str, bpy.types.Object]]:
    clear_scene()
    configure_scene(args)
    rng = random.Random(args.seed + sample_index * 100_003)
    scene = bpy.context.scene
    if rng.random() < args.dark_background_ratio:
        scene.world.color = (rng.uniform(0.015, 0.13), rng.uniform(0.015, 0.13), rng.uniform(0.015, 0.13))
    else:
        scene.world.color = (rng.uniform(0.22, 0.62), rng.uniform(0.22, 0.62), rng.uniform(0.22, 0.62))

    object_count = rng.randint(args.min_objects, args.max_objects)
    start_class_index = sample_index * args.max_objects
    placed: list[tuple[float, float]] = []
    objects: list[tuple[str, bpy.types.Object]] = []
    for object_index in range(object_count):
        class_name = class_for_object(start_class_index + object_index)
        scale = primitive_scale(class_name, rng)
        radius = bucket_radius(bucket)
        x, y = sample_location(rng, bucket, placed, radius)
        placed.append((x, y))
        location = (x, y, max(0.25, scale[2] * 0.5))
        material = make_material(
            f"mat_{sample_index:05d}_{object_index:02d}_{class_name}",
            random_color(rng, args.material_variation),
            rng.uniform(0.38, 0.78),
        )
        obj = add_primitive(class_name, object_index, location, scale, primitive_rotation(class_name, rng, bucket), material)
        objects.append((class_name, obj))

    add_lighting(rng)
    place_camera(args, rng)
    return objects


def looks_complete(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 16:
        return False
    with path.open("rb") as file:
        file.seek(max(0, path.stat().st_size - 32))
        return b"IEND" in file.read()


def sample_complete(root: Path, split: str, stem: str) -> bool:
    required_pngs = [
        split_path(root, split, "images") / f"{stem}.png",
        split_path(root, split, "images_rgb") / f"{stem}.png",
        split_path(root, split, "images_rgbd") / f"{stem}.png",
        split_path(root, split, "depth") / f"{stem}.png",
    ]
    if any(not looks_complete(path) for path in required_pngs):
        return False
    return any(looks_complete(path) for path in split_path(root, split, "masks").glob(f"{stem}_*.png"))


def generate_dataset(args: argparse.Namespace) -> None:
    root = Path(args.output)
    for split in DATASET_SPLITS:
        for artifact in ("images", "images_rgb", "images_rgbd", "depth", "labels", "masks"):
            split_path(root, split, artifact).mkdir(parents=True, exist_ok=True)
    write_rgbd_data_yaml(root, BASE_CLASSES)

    generated = 0
    skipped = 0
    split_counts = {split: 0 for split in DATASET_SPLITS}
    bucket_counts = {bucket: 0 for bucket in OCCLUSION_BUCKETS}
    class_counts = {class_name: 0 for class_name in PRIMITIVE_CLASSES}
    shard_total = sum(1 for index in range(args.start_index, args.end_index) if index % args.shard_count == args.shard_index)

    for index in range(args.start_index, args.end_index):
        if index % args.shard_count != args.shard_index:
            continue
        split = "test" if args.eval_only else split_name_for_index(index, args.count, args.train_split, args.val_split)
        bucket = args.occlusion_buckets[index % len(args.occlusion_buckets)]
        stem = f"no_plane_{index:05d}"
        if args.finish and sample_complete(root, split, stem):
            skipped += 1
            continue

        objects = build_sample(args, index, bucket)
        image_path = split_path(root, split, "images") / f"{stem}.png"
        rgb_path = split_path(root, split, "images_rgb") / f"{stem}.png"
        depth_path = split_path(root, split, "depth") / f"{stem}.png"
        rgbd_path = split_path(root, split, "images_rgbd") / f"{stem}.png"
        bpy.context.scene.render.filepath = str(rgb_path)
        bpy.ops.render.render(write_still=True)
        render_depth_and_rgbd(depth_path, rgbd_path, args.depth_near, args.depth_far)
        shutil.copyfile(rgbd_path, image_path)
        render_instance_masks(split_path(root, split, "masks"), stem, objects)

        generated += 1
        split_counts[split] += 1
        bucket_counts[bucket] += 1
        for class_name, _obj in objects:
            class_counts[class_name] += 1
        if args.log_every and (generated % args.log_every == 0 or generated == shard_total):
            print(f"no-plane {generated:04d}/{shard_total:04d} global {index + 1:05d}/{args.count}: {image_path}")

    manifest = {
        "schema_version": 1,
        "generator": "generate_no_plane_primitives_dataset.py",
        "classes": list(BASE_CLASSES),
        "primitive_classes": list(PRIMITIVE_CLASSES),
        "count": int(args.count),
        "generated": generated,
        "skipped_existing": skipped,
        "split_counts": split_counts,
        "class_counts": class_counts,
        "occlusion_bucket_counts": bucket_counts,
        "occlusion_buckets": list(args.occlusion_buckets),
        "seed": int(args.seed),
        "width": int(args.width),
        "height": int(args.height),
        "render_samples": int(args.render_samples),
        "depth_near": float(args.depth_near),
        "depth_far": float(args.depth_far),
        "min_objects": int(args.min_objects),
        "max_objects": int(args.max_objects),
        "eval_only": bool(args.eval_only),
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
    }
    (root / "target_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Generated {generated} no-plane samples, skipped {skipped}; wrote {root}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        from Runtime.guided_cli import guided_blender_tool_main

        raise SystemExit(
            guided_blender_tool_main(
                Path(__file__),
                'Generate no-plane synthetic primitive RGBD dataset.',
                ['--output', 'Datasets/PrimitiveShapesNoPlane'],
                blend_path=None,
            )
        )
    generate_dataset(parse_args())
