from __future__ import annotations

import argparse
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Tools.Dataset.generate_no_plane_primitives_dataset import (  # noqa: E402
    OCCLUSION_BUCKETS,
    PRIMITIVE_CLASSES,
    add_lighting,
    add_primitive,
    bucket_radius,
    class_for_object,
    clear_scene,
    configure_scene,
    looks_complete,
    make_material,
    place_camera,
    primitive_rotation,
    primitive_scale,
    random_color,
    sample_location,
)
from Tools.Dataset.generate_primitives_dataset import render_depth_and_rgbd, render_instance_masks  # noqa: E402
from Tools.Dataset.rgbd_curriculum import BASE_CLASSES, DATASET_SPLITS, split_name_for_index, split_path, write_rgbd_data_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RGBD primitive detection data with labeled room-like planes.")
    parser.add_argument("--output", default="Datasets/PrimitiveShapesRGBDTarget/plane_context")
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
    parser.add_argument("--dark-background-ratio", type=float, default=0.45)
    parser.add_argument("--material-variation", type=float, default=0.85)
    parser.add_argument("--camera-distance-min", type=float, default=5.0)
    parser.add_argument("--camera-distance-max", type=float, default=8.5)
    parser.add_argument("--camera-height-min", type=float, default=2.0)
    parser.add_argument("--camera-height-max", type=float, default=5.0)
    parser.add_argument("--fov-degrees-min", type=float, default=42.0)
    parser.add_argument("--fov-degrees-max", type=float, default=68.0)
    parser.add_argument("--back-wall-probability", type=float, default=0.85)
    parser.add_argument("--side-wall-probability", type=float, default=0.45)
    parser.add_argument("--wall-jitter-degrees", type=float, default=3.0)
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
    for name in ("back_wall_probability", "side_wall_probability"):
        value = float(getattr(args, name))
        if value < 0.0 or value > 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1")


def add_plane_slab(
    name: str,
    location: tuple[float, float, float],
    dimensions: tuple[float, float, float],
    rotation: tuple[float, float, float],
    material: bpy.types.Material,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location, rotation=rotation)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)
    return obj


def plane_rotation(base: tuple[float, float, float], jitter_degrees: float, rng: random.Random) -> tuple[float, float, float]:
    jitter = math.radians(max(0.0, jitter_degrees))
    return tuple(value + rng.uniform(-jitter, jitter) for value in base)


def add_room_planes(args: argparse.Namespace, rng: random.Random, sample_index: int) -> list[tuple[str, bpy.types.Object]]:
    room_width = rng.uniform(6.5, 9.0)
    room_depth = rng.uniform(5.0, 7.5)
    wall_height = rng.uniform(3.0, 4.8)
    thickness = rng.uniform(0.035, 0.070)
    material = make_material(
        f"mat_{sample_index:05d}_room_plane",
        random_color(rng, max(0.25, args.material_variation * 0.55)),
        rng.uniform(0.55, 0.90),
    )
    planes: list[tuple[str, bpy.types.Object]] = []
    planes.append(
        (
            "plane",
            add_plane_slab(
                "00_plane_floor",
                (0.0, 0.0, -thickness * 0.5),
                (room_width, room_depth, thickness),
                plane_rotation((0.0, 0.0, 0.0), args.wall_jitter_degrees * 0.35, rng),
                material,
            ),
        )
    )
    if rng.random() < args.back_wall_probability:
        planes.append(
            (
                "plane",
                add_plane_slab(
                    "01_plane_back_wall",
                    (0.0, room_depth * 0.5, wall_height * 0.5),
                    (room_width, thickness, wall_height),
                    plane_rotation((0.0, 0.0, 0.0), args.wall_jitter_degrees, rng),
                    material,
                ),
            )
        )
    if rng.random() < args.side_wall_probability:
        side = -1.0 if rng.random() < 0.5 else 1.0
        planes.append(
            (
                "plane",
                add_plane_slab(
                    "02_plane_side_wall",
                    (side * room_width * 0.5, 0.0, wall_height * 0.5),
                    (thickness, room_depth, wall_height),
                    plane_rotation((0.0, 0.0, 0.0), args.wall_jitter_degrees, rng),
                    material,
                ),
            )
        )
    return planes


def build_sample(args: argparse.Namespace, sample_index: int, bucket: str) -> list[tuple[str, bpy.types.Object]]:
    clear_scene()
    configure_scene(args)
    rng = random.Random(args.seed + sample_index * 100_003)
    scene = bpy.context.scene
    if rng.random() < args.dark_background_ratio:
        scene.world.color = (rng.uniform(0.015, 0.12), rng.uniform(0.015, 0.12), rng.uniform(0.015, 0.12))
    else:
        scene.world.color = (rng.uniform(0.18, 0.50), rng.uniform(0.18, 0.50), rng.uniform(0.18, 0.50))

    objects = add_room_planes(args, rng, sample_index)
    object_count = rng.randint(args.min_objects, args.max_objects)
    start_class_index = sample_index * args.max_objects
    placed: list[tuple[float, float]] = []
    for object_index in range(object_count):
        class_name = class_for_object(start_class_index + object_index)
        scale = primitive_scale(class_name, rng)
        radius = bucket_radius(bucket)
        x, y = sample_location(rng, bucket, placed, radius)
        y = min(2.15, max(-1.95, y))
        placed.append((x, y))
        location = (x, y, max(0.25, scale[2] * 0.5))
        material = make_material(
            f"mat_{sample_index:05d}_{object_index:02d}_{class_name}",
            random_color(rng, args.material_variation),
            rng.uniform(0.35, 0.80),
        )
        obj = add_primitive(class_name, object_index + len(objects), location, scale, primitive_rotation(class_name, rng, bucket), material)
        objects.append((class_name, obj))

    add_lighting(rng)
    place_camera(args, rng)
    return objects


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
    foreground_counts = {class_name: 0 for class_name in PRIMITIVE_CLASSES}
    plane_count = 0
    shard_total = sum(1 for index in range(args.start_index, args.end_index) if index % args.shard_count == args.shard_index)

    for index in range(args.start_index, args.end_index):
        if index % args.shard_count != args.shard_index:
            continue
        split = "test" if args.eval_only else split_name_for_index(index, args.count, args.train_split, args.val_split)
        bucket = args.occlusion_buckets[index % len(args.occlusion_buckets)]
        stem = f"plane_context_{index:05d}"
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
            if class_name == "plane":
                plane_count += 1
            else:
                foreground_counts[class_name] += 1
        if args.log_every and (generated % args.log_every == 0 or generated == shard_total):
            print(f"plane-context {generated:04d}/{shard_total:04d} global {index + 1:05d}/{args.count}: {image_path}")

    class_counts = {**foreground_counts, "plane": plane_count}
    manifest = {
        "schema_version": 1,
        "generator": "generate_plane_context_primitives_dataset.py",
        "classes": list(BASE_CLASSES),
        "foreground_classes": list(PRIMITIVE_CLASSES),
        "plane_class": "plane",
        "count": int(args.count),
        "generated": generated,
        "skipped_existing": skipped,
        "split_counts": split_counts,
        "class_counts": class_counts,
        "foreground_class_counts": foreground_counts,
        "plane_count": plane_count,
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
        "back_wall_probability": float(args.back_wall_probability),
        "side_wall_probability": float(args.side_wall_probability),
        "wall_jitter_degrees": float(args.wall_jitter_degrees),
        "eval_only": bool(args.eval_only),
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
    }
    (root / "target_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Generated {generated} plane-context samples, skipped {skipped}; wrote {root}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        from Runtime.guided_cli import guided_blender_tool_main

        raise SystemExit(
            guided_blender_tool_main(
                Path(__file__),
                'Generate plane-context primitive RGBD dataset.',
                ['--output', 'Datasets/PrimitiveShapesPlaneContext'],
                blend_path=None,
            )
        )
    generate_dataset(parse_args())
