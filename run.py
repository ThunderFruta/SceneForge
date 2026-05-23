from __future__ import annotations

import argparse
from pathlib import Path
import sys

from Core.Utils.output_paths import resolve_output_blend_path
from Pipeline.ImageToMesh.image_to_mesh_pipeline import run_image_to_mesh_pipeline
from Pipeline.StructuredScene.structured_scene_pipeline import run_structured_scene_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an image and optional depth map into a textured Blender file."
    )
    parser.add_argument("--image", required=True, help="Path to the source image.")
    parser.add_argument("--depth", help="Optional grayscale depth image.")
    parser.add_argument(
        "--output",
        default="Output",
        help=(
            "Output path. A timestamped run folder is always created. "
            "Use a directory like Output or a desired file name like Output/scene.blend."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("relief", "structured"),
        default="relief",
        help="Reconstruction mode. Relief keeps the current depth-sheet behavior; structured builds planes plus detail patches.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help="Maximum grid points along the largest image dimension.",
    )
    parser.add_argument(
        "--depth-strength",
        type=float,
        default=1.0,
        help="Multiplier applied to normalized depth values.",
    )
    texture_group = parser.add_mutually_exclusive_group()
    texture_group.add_argument(
        "--texture",
        dest="texture",
        action="store_true",
        help="Write a material and texture image next to the OBJ.",
    )
    texture_group.add_argument(
        "--no-texture",
        dest="texture",
        action="store_false",
        help="Write geometry only.",
    )
    parser.add_argument(
        "--obj",
        action="store_true",
        help="Also keep a sidecar OBJ/MTL/texture bundle next to the .blend file.",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help=(
            "Structured mode only: include leftover relief detail patches. "
            "Off by default while plane reconstruction stabilizes."
        ),
    )
    solidify_group = parser.add_mutually_exclusive_group()
    solidify_group.add_argument(
        "--solidify",
        dest="solidify",
        action="store_true",
        help="Structured mode only: add thin side walls to visible scan boundaries.",
    )
    solidify_group.add_argument(
        "--no-solidify",
        dest="solidify",
        action="store_false",
        help="Structured mode only: keep front-facing surfaces without side walls.",
    )
    parser.add_argument(
        "--solidify-thickness",
        type=float,
        default=0.04,
        help="Structured mode only: side-wall thickness before Blender scale.",
    )
    parser.add_argument(
        "--depth-edge-threshold",
        type=float,
        default=0.12,
        help="Structured mode only: reject faces spanning larger normalized depth jumps.",
    )
    parser.add_argument(
        "--blender",
        default="blender",
        help="Blender executable name or path.",
    )
    parser.set_defaults(texture=True, solidify=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = resolve_output_blend_path(
        args.output,
        mode=args.mode,
        image_path=args.image,
    )
    try:
        if args.mode == "structured":
            result = run_structured_scene_pipeline(
                image_path=Path(args.image),
                output_path=output_path,
                depth_path=Path(args.depth) if args.depth else None,
                resolution=args.resolution,
                depth_strength=args.depth_strength,
                write_texture=args.texture,
                keep_obj=args.obj,
                blender_executable=args.blender,
                include_details=args.details,
                solidify=args.solidify,
                solidify_thickness=args.solidify_thickness,
                depth_edge_threshold=args.depth_edge_threshold,
            )
        else:
            result = run_image_to_mesh_pipeline(
                image_path=Path(args.image),
                output_path=output_path,
                depth_path=Path(args.depth) if args.depth else None,
                resolution=args.resolution,
                depth_strength=args.depth_strength,
                write_texture=args.texture,
                keep_obj=args.obj,
                blender_executable=args.blender,
            )
    except RuntimeError as error:
        print(f"SceneForge error: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    print(f"Wrote blend: {result.blend_path}")
    print(f"Wrote preview: {result.preview_path}")
    if result.obj_result:
        print(f"Wrote OBJ: {result.obj_result.obj_path}")
        if result.obj_result.mtl_path:
            print(f"Wrote MTL: {result.obj_result.mtl_path}")
        if result.obj_result.texture_path:
            print(f"Wrote texture: {result.obj_result.texture_path}")


if __name__ == "__main__":
    main()
