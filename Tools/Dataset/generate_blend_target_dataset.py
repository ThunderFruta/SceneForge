from __future__ import annotations

import argparse
import colorsys
import json
import random
import shutil
import sys
from pathlib import Path

try:
    import bpy
except ModuleNotFoundError:
    bpy = None
from mathutils import Euler, Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Tools.Dataset.generate_primitives_dataset import (  # noqa: E402
    render_depth_and_rgbd,
    render_instance_masks,
)
from Tools.Dataset.rgbd_curriculum import (  # noqa: E402
    BASE_CLASSES,
    DATASET_SPLITS,
    split_name_for_index,
    split_path,
    write_rgbd_data_yaml,
)


LABEL_BY_NAME = {
    "box": "box",
    "cube": "box",
    "cuboid": "box",
    "cylinder": "cylinder",
    "cone": "cone",
    "sphere": "sphere",
    "plane": "plane",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render target-style RGBD YOLO data from the currently loaded .blend file."
    )
    parser.add_argument("--output", default="Datasets/PrimitiveShapesRGBDTarget/shapes_blend")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--render-samples", type=int, default=16)
    parser.add_argument("--train-split", type=float, default=0.70)
    parser.add_argument("--val-split", type=float, default=0.20)
    parser.add_argument("--depth-near", type=float, default=1.0)
    parser.add_argument("--depth-far", type=float, default=8.0)
    parser.add_argument("--camera-jitter", type=float, default=0.28)
    parser.add_argument("--target-jitter", type=float, default=0.12)
    parser.add_argument("--fov-jitter-degrees", type=float, default=3.0)
    parser.add_argument(
        "--object-rotation-degrees",
        type=float,
        default=0.0,
        help="Maximum random per-object Euler rotation jitter in degrees. 0 preserves original object rotations.",
    )
    parser.add_argument(
        "--random-object-rotation",
        action="store_true",
        help="Use fully random per-object Euler rotations instead of jitter around original rotations.",
    )
    parser.add_argument("--dark-background-ratio", type=float, default=0.35)
    parser.add_argument("--material-variation", type=float, default=0.85)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Write all rendered samples to the test split. Use this for held-out target evaluations.",
    )
    parser.add_argument(
        "--exact-first",
        action="store_true",
        help="Keep global sample 0 at the exact original camera/materials in non-eval datasets.",
    )
    parser.add_argument("--finish", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    argv = sys.argv
    script_args = argv[argv.index("--") + 1 :] if "--" in argv else []
    args = parser.parse_args(script_args)
    if args.count < 1:
        raise ValueError("--count must be at least 1")
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
    if args.log_every < 0:
        raise ValueError("--log-every must be greater than or equal to 0")
    if args.object_rotation_degrees < 0.0:
        raise ValueError("--object-rotation-degrees must be greater than or equal to 0")
    return args


def class_for_object(obj: bpy.types.Object) -> str | None:
    name = obj.name.lower()
    data_name = obj.data.name.lower() if getattr(obj, "data", None) else ""
    for token, class_name in LABEL_BY_NAME.items():
        if token in name or token in data_name:
            return class_name
    return None


def labeled_objects() -> list[tuple[str, bpy.types.Object]]:
    objects: list[tuple[str, bpy.types.Object]] = []
    for obj in sorted(bpy.context.scene.objects, key=lambda item: item.name):
        if obj.type != "MESH":
            continue
        class_name = class_for_object(obj)
        if class_name is None:
            continue
        objects.append((class_name, obj))
    if not objects:
        raise ValueError("No labeled mesh objects were found in the loaded blend.")
    return objects


def mesh_center(objects: list[tuple[str, bpy.types.Object]]) -> Vector:
    points = [obj.matrix_world.translation for _class_name, obj in objects]
    return sum(points, Vector()) / len(points)


def store_original_state(objects: list[tuple[str, bpy.types.Object]]) -> dict:
    scene = bpy.context.scene
    camera = scene.camera
    if camera is None:
        raise ValueError("The blend file has no active camera.")
    return {
        "camera_location": camera.location.copy(),
        "camera_rotation": camera.rotation_euler.copy(),
        "camera_angle": float(camera.data.angle),
        "camera_shift_x": float(camera.data.shift_x),
        "camera_shift_y": float(camera.data.shift_y),
        "camera_sensor_fit": camera.data.sensor_fit,
        "world_color": tuple(scene.world.color) if scene.world else (0.05, 0.05, 0.05),
        "materials": [(obj, list(obj.data.materials)) for _class_name, obj in objects],
        "object_rotation_modes": {obj.name: obj.rotation_mode for _class_name, obj in objects},
        "object_rotations": {obj.name: obj.rotation_euler.copy() for _class_name, obj in objects},
        "lights": [
            (
                obj,
                obj.location.copy(),
                float(obj.data.energy) if hasattr(obj.data, "energy") else 0.0,
                tuple(obj.data.color) if hasattr(obj.data, "color") else (1.0, 1.0, 1.0),
            )
            for obj in scene.objects
            if obj.type == "LIGHT"
        ],
    }


def restore_original_state(state: dict) -> None:
    scene = bpy.context.scene
    camera = scene.camera
    camera.location = state["camera_location"]
    camera.rotation_euler = state["camera_rotation"]
    camera.data.angle = state["camera_angle"]
    camera.data.shift_x = state["camera_shift_x"]
    camera.data.shift_y = state["camera_shift_y"]
    camera.data.sensor_fit = state["camera_sensor_fit"]
    if scene.world:
        scene.world.color = state["world_color"]
    for obj, materials in state["materials"]:
        obj.data.materials.clear()
        for material in materials:
            obj.data.materials.append(material)
    for obj_name, rotation_euler in state["object_rotations"].items():
        obj = bpy.data.objects.get(obj_name)
        if obj is None:
            continue
        obj.rotation_mode = state["object_rotation_modes"][obj_name]
        obj.rotation_euler = rotation_euler
    for obj, location, energy, color in state["lights"]:
        obj.location = location
        if hasattr(obj.data, "energy"):
            obj.data.energy = energy
        if hasattr(obj.data, "color"):
            obj.data.color = color


def make_material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = 0.58
    return material


def random_color(rng: random.Random, variation: float) -> tuple[float, float, float, float]:
    hue = rng.random()
    saturation = rng.uniform(0.22, min(1.0, 0.35 + variation * 0.70))
    if rng.random() < 0.55:
        value = rng.uniform(0.10, 0.42 + variation * 0.18)
    else:
        value = rng.uniform(0.42, 0.90)
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return red, green, blue, 1.0


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def configure_render(width: int, height: int, samples: int) -> None:
    scene = bpy.context.scene
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = samples
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.camera.data.sensor_fit = "HORIZONTAL"


def apply_variation(
    objects: list[tuple[str, bpy.types.Object]],
    center: Vector,
    state: dict,
    args: argparse.Namespace,
    rng: random.Random,
    index: int,
) -> None:
    restore_original_state(state)
    exact_sample = index == 0 and (args.eval_only or args.exact_first)
    scene = bpy.context.scene
    camera = scene.camera
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    if exact_sample:
        scene.world.color = state["world_color"]
    elif rng.random() < args.dark_background_ratio:
        scene.world.color = (
            rng.uniform(0.015, 0.16),
            rng.uniform(0.015, 0.16),
            rng.uniform(0.015, 0.16),
        )
    else:
        scene.world.color = (
            rng.uniform(0.28, 0.72),
            rng.uniform(0.28, 0.72),
            rng.uniform(0.28, 0.72),
        )

    if not exact_sample:
        jitter = args.camera_jitter
        camera.location = state["camera_location"] + Vector(
            (
                rng.uniform(-jitter, jitter),
                rng.uniform(-jitter, jitter),
                rng.uniform(-jitter * 0.45, jitter * 0.45),
            )
        )
        target = center + Vector(
            (
                rng.uniform(-args.target_jitter, args.target_jitter),
                rng.uniform(-args.target_jitter, args.target_jitter),
                rng.uniform(-args.target_jitter, args.target_jitter),
            )
        )
        look_at(camera, target)
        camera.data.angle = max(
            0.1,
            state["camera_angle"] + rng.uniform(-args.fov_jitter_degrees, args.fov_jitter_degrees) * 0.017453292519943295,
        )

    for class_name, obj in objects:
        if not exact_sample and args.object_rotation_degrees > 0.0:
            rotation_jitter = args.object_rotation_degrees * 0.017453292519943295
            obj.rotation_mode = "XYZ"
            base_rotation = state["object_rotations"][obj.name]
            if args.random_object_rotation:
                obj.rotation_euler = Euler(
                    (
                        rng.uniform(-rotation_jitter, rotation_jitter),
                        rng.uniform(-rotation_jitter, rotation_jitter),
                        rng.uniform(-rotation_jitter, rotation_jitter),
                    ),
                    "XYZ",
                )
            else:
                obj.rotation_euler = Euler(
                    (
                        base_rotation.x + rng.uniform(-rotation_jitter, rotation_jitter),
                        base_rotation.y + rng.uniform(-rotation_jitter, rotation_jitter),
                        base_rotation.z + rng.uniform(-rotation_jitter, rotation_jitter),
                    ),
                    "XYZ",
                )
        if exact_sample or rng.random() > args.material_variation:
            continue
        obj.data.materials.clear()
        obj.data.materials.append(make_material(f"target_{index:04d}_{obj.name}_{class_name}", random_color(rng, args.material_variation)))

    for light_obj, base_location, base_energy, _base_color in state["lights"]:
        if exact_sample:
            continue
        light_obj.location = base_location + Vector(
            (
                rng.uniform(-1.3, 1.3),
                rng.uniform(-1.3, 1.3),
                rng.uniform(-0.7, 0.7),
            )
        )
        if hasattr(light_obj.data, "energy"):
            light_obj.data.energy = max(20.0, base_energy * rng.uniform(0.35, 1.75))
        if hasattr(light_obj.data, "color"):
            light_obj.data.color = (
                rng.uniform(0.75, 1.0),
                rng.uniform(0.75, 1.0),
                rng.uniform(0.75, 1.0),
            )


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
    rng = random.Random(args.seed)
    objects = labeled_objects()
    center = mesh_center(objects)
    state = store_original_state(objects)
    root = Path(args.output)
    for split in DATASET_SPLITS:
        for artifact in ("images", "images_rgb", "images_rgbd", "depth", "labels", "masks"):
            split_path(root, split, artifact).mkdir(parents=True, exist_ok=True)
    write_rgbd_data_yaml(root, BASE_CLASSES)

    configure_render(args.width, args.height, args.render_samples)
    split_counts = {split: 0 for split in DATASET_SPLITS}
    generated = 0
    skipped = 0
    shard_total = sum(1 for index in range(args.start_index, args.end_index) if index % args.shard_count == args.shard_index)

    for index in range(args.start_index, args.end_index):
        if index % args.shard_count != args.shard_index:
            continue
        split = "test" if args.eval_only else split_name_for_index(index, args.count, args.train_split, args.val_split)
        stem = f"target_{index:04d}"
        if args.finish and sample_complete(root, split, stem):
            skipped += 1
            continue
        apply_variation(objects, center, state, args, rng, index)
        image_path = split_path(root, split, "images") / f"{stem}.png"
        rgb_path = split_path(root, split, "images_rgb") / f"{stem}.png"
        depth_path = split_path(root, split, "depth") / f"{stem}.png"
        rgbd_path = split_path(root, split, "images_rgbd") / f"{stem}.png"
        bpy.context.scene.render.filepath = str(rgb_path)
        bpy.ops.render.render(write_still=True)
        render_depth_and_rgbd(depth_path, rgbd_path, args.depth_near, args.depth_far)
        shutil.copyfile(rgbd_path, image_path)
        render_instance_masks(split_path(root, split, "masks"), stem, objects)
        split_counts[split] += 1
        generated += 1
        if args.log_every and (generated % args.log_every == 0 or generated == shard_total):
            print(f"target {generated:04d}/{shard_total:04d} global {index + 1:04d}/{args.count}: {image_path}")

    manifest = {
        "schema_version": 1,
        "source_blend": bpy.data.filepath,
        "classes": list(BASE_CLASSES),
        "objects": [{"name": obj.name, "class_name": class_name} for class_name, obj in objects],
        "count": args.count,
        "split_counts": split_counts,
        "seed": args.seed,
        "width": args.width,
        "height": args.height,
        "render_samples": args.render_samples,
        "depth_near": args.depth_near,
        "depth_far": args.depth_far,
        "camera_jitter": args.camera_jitter,
        "target_jitter": args.target_jitter,
        "fov_jitter_degrees": args.fov_jitter_degrees,
        "object_rotation_degrees": args.object_rotation_degrees,
        "random_object_rotation": bool(args.random_object_rotation),
        "generated": generated,
        "skipped_existing": skipped,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "eval_only": bool(args.eval_only),
        "exact_first": bool(args.exact_first),
    }
    (root / "target_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    restore_original_state(state)
    print(f"Generated {generated} target samples, skipped {skipped}; wrote {root}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        from Runtime.guided_cli import guided_blender_tool_main

        raise SystemExit(
            guided_blender_tool_main(
                Path(__file__),
                'Generate target RGBD data from a labeled .blend file.',
                ['--reference-blend', 'Assets/Samples/roomScene.blend'],
                blend_path='Assets/Samples/roomScene.blend',
            )
        )
    generate_dataset(parse_args())
