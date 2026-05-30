from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OPEN_VOCAB_BACKENDS = ("sam3", "groundingdino-sam3", "ram-groundingdino-sam3")
PUBLIC_DETECTOR_BACKENDS = OPEN_VOCAB_BACKENDS
PRIMITIVE_SOURCES = ("none", "detector-label", "clip")


class CliError(RuntimeError):
    """User-facing CLI failure with an exit code of 2."""



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SceneForge staged SAM3/object mesh and empty-room VGGT scene pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)


    preflight = subparsers.add_parser("check-open-vocab-integration", help="Preflight local GroundingDINO/SAM3 repo and model paths.")
    preflight.add_argument("--backend", choices=OPEN_VOCAB_BACKENDS, default="groundingdino-sam3")
    preflight.add_argument("--groundingdino-repo-dir", default="Models/OpenVocabulary/GroundingDINO/repo")
    preflight.add_argument(
        "--groundingdino-config",
        default="Models/OpenVocabulary/GroundingDINO/repo/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    )
    preflight.add_argument(
        "--groundingdino-checkpoint",
        default="Models/OpenVocabulary/GroundingDINO/weights/groundingdino_swint_ogc.pth",
    )
    preflight.add_argument("--sam3-repo-dir", default="Models/OpenVocabulary/SAM3/repo")
    preflight.add_argument("--sam3-model-dir", default="Models/OpenVocabulary/SAM3/hf")
    preflight.add_argument("--ram-repo-dir")
    preflight.add_argument("--ram-checkpoint")
    preflight.add_argument("--text-prompt", default="chair . table . box . sphere . cylinder . cone . sofa . lamp . plant . flower . flowers . vase . flower pot . person . foreground object .")
    preflight.add_argument("--output", default="Output/Latest/open_vocab_preflight.json")
    preflight.set_defaults(func=cmd_check_open_vocab_integration)


    import_probe = subparsers.add_parser("probe-open-vocab-imports", help="Probe local GroundingDINO/SAM3 imports without loading checkpoints.")
    import_probe.add_argument("--backend", choices=OPEN_VOCAB_BACKENDS, default="groundingdino-sam3")
    import_probe.add_argument("--groundingdino-repo-dir", default="Models/OpenVocabulary/GroundingDINO/repo")
    import_probe.add_argument("--sam3-repo-dir", default="Models/OpenVocabulary/SAM3/repo")
    import_probe.add_argument("--ram-repo-dir")
    import_probe.add_argument("--ram-checkpoint")
    import_probe.add_argument("--output", default="Output/Latest/open_vocab_import_probe.json")
    import_probe.set_defaults(func=cmd_probe_open_vocab_imports)


    prepare_open_vocab = subparsers.add_parser("prepare-open-vocab-layout", help="Create local GroundingDINO/SAM3 layout and setup manifest.")
    prepare_open_vocab.add_argument("--root", default="Models/OpenVocabulary")
    prepare_open_vocab.add_argument("--no-create-dirs", action="store_true")
    prepare_open_vocab.add_argument("--no-script", action="store_true")
    prepare_open_vocab.set_defaults(func=cmd_prepare_open_vocab_layout)


    readiness = subparsers.add_parser("audit-open-vocab-readiness", help="Run non-inference readiness checks for GroundingDINO/SAM3 integration.")
    readiness.add_argument("--root", default="Models/OpenVocabulary")
    readiness.add_argument("--backend", choices=OPEN_VOCAB_BACKENDS, default="groundingdino-sam3")
    readiness.add_argument("--ram-repo-dir")
    readiness.add_argument("--ram-checkpoint")
    readiness.add_argument("--text-prompt", default="chair . table . box . sphere . cylinder . cone . sofa . lamp . plant . flower . flowers . vase . flower pot . person . foreground object .")
    readiness.add_argument("--skip-import-probe", action="store_true")
    readiness.add_argument("--output", default="Output/Latest/open_vocab_readiness.json")
    readiness.set_defaults(func=cmd_audit_open_vocab_readiness)


    smoke = subparsers.add_parser("run-open-vocab-smoke", help="Run guarded GroundingDINO/SAM3 proposal smoke test.")
    smoke.add_argument("--root", default="Models/OpenVocabulary")
    smoke.add_argument("--backend", choices=OPEN_VOCAB_BACKENDS, default="groundingdino-sam3")
    smoke.add_argument("--ram-repo-dir")
    smoke.add_argument("--ram-checkpoint")
    smoke.add_argument("--text-prompt", default="chair . table . box . sphere . cylinder . cone . sofa . lamp . plant . flower . flowers . vase . flower pot . person . foreground object .")
    smoke.add_argument("--output", default="Output/Latest/open_vocab_smoke.json")
    smoke.set_defaults(func=cmd_run_open_vocab_smoke)

    render_png = subparsers.add_parser("render-blend-png", help="Render a .blend file to a PNG image only.")
    render_png.add_argument("--reference-blend", required=True)
    render_png.add_argument("--output", default="Output/Latest/render/image.png")
    render_png.add_argument("--camera-name")
    render_png.add_argument("--blender", default="blender")
    render_png.add_argument("--width", type=int, default=1280)
    render_png.add_argument("--height", type=int, default=720)
    render_png.add_argument("--render-samples", type=int, default=2048)
    render_png.add_argument("--render-quality", choices=("fast", "balanced", "quality"), default="balanced")
    render_png.add_argument("--render-engine", default="CYCLES", choices=("auto", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "CYCLES"))
    render_png.add_argument("--cycles-device-filter", default="4080")
    render_png.add_argument("--exposure", default="auto")
    render_png.add_argument("--gamma", type=float, default=1.0)
    render_png.set_defaults(func=cmd_render_blend_png)

    render_scene_camera = subparsers.add_parser("render-scene-camera-view", help="Render a composed OBJ/GLB scene from the source image camera.")
    render_scene_camera.add_argument("--scene", default="Output/Latest/scene/scene.glb")
    render_scene_camera.add_argument("--output", default="Output/Latest/scene/source_camera_render.png")
    render_scene_camera.add_argument("--alignment-report", default="Output/Latest/scene/scene_alignment.json")
    render_scene_camera.add_argument("--blender", default="blender")
    render_scene_camera.add_argument("--width", type=int)
    render_scene_camera.add_argument("--height", type=int)
    render_scene_camera.add_argument("--fov-degrees", type=float)
    render_scene_camera.add_argument("--camera-mode", choices=("source", "fit-preview"), default="source")
    render_scene_camera.set_defaults(func=cmd_render_scene_camera_view)

    run_vggt = subparsers.add_parser("run-vggt", help="Run VGGT geometry capture for one RGB image.")
    run_vggt.add_argument("--image", required=True)
    run_vggt.add_argument("--output", default="Output/Latest/objects_vggt")
    run_vggt.add_argument("--backend", choices=("vggt", "fake"), default="vggt")
    run_vggt.add_argument("--model", default="facebook/VGGT-1B")
    run_vggt.add_argument("--vggt-repo-dir")
    run_vggt.add_argument("--vggt-checkpoint")
    run_vggt.add_argument("--vggt-cache-dir", default="Models/Geometry/VGGT/hf-cache")
    run_vggt.add_argument(
        "--vggt-local-only",
        action="store_true",
        help="Load VGGT only from the local Hugging Face cache without checking the Hub.",
    )
    run_vggt.add_argument("--obj-stride", type=int, default=8, help="Pixel stride for the sampled VGGT OBJ mesh.")
    run_vggt.add_argument("--mesh-stem", default="vggt_mesh", help="Base filename for sampled OBJ/GLB mesh outputs.")
    run_vggt.add_argument("--device", default="auto")
    run_vggt.set_defaults(func=cmd_run_vggt)

    empty_room = subparsers.add_parser("generate-empty-room", help="Create an empty-room image and foreground removal artifacts.")
    empty_room.add_argument("--image", required=True)
    empty_room.add_argument("--detections", required=True)
    empty_room.add_argument("--objects", default="Output/Latest/objects")
    empty_room.add_argument("--output", default="Output/Latest/background")
    empty_room.add_argument("--empty-room-backend", choices=("openai-image", "fake"), default="openai-image")
    empty_room.add_argument("--empty-room-model", default="gpt-image-1.5")
    empty_room.add_argument("--fill-mode", choices=("transparent", "neutral", "black"), default="transparent")
    empty_room.add_argument("--mask-dilation-px", type=int, default=10)
    empty_room.add_argument("--mask-feather-px", type=int, default=0)
    empty_room.add_argument("--include-detection-id", dest="include_detection_ids", action="append", default=[])
    empty_room.add_argument("--exclude-detection-id", dest="exclude_detection_ids", action="append", default=[])
    empty_room.add_argument("--allow-rectangular-fallback-masks", action="store_true")
    empty_room.add_argument("--max-mask-coverage", type=float, default=0.55)
    empty_room.set_defaults(func=cmd_generate_empty_room)

    empty_room_vggt = subparsers.add_parser(
        "construct-empty-room",
        aliases=("run-empty-room-vggt",),
        help="Generate an empty room, run VGGT on it, and export OBJ/GLB background mesh artifacts.",
    )
    empty_room_vggt.add_argument("--image", required=True)
    empty_room_vggt.add_argument("--detections", required=True)
    empty_room_vggt.add_argument("--objects", default="Output/Latest/objects")
    empty_room_vggt.add_argument("--output", default="Output/Latest/background")
    empty_room_vggt.add_argument("--empty-room-backend", choices=("openai-image", "fake"), default="openai-image")
    empty_room_vggt.add_argument("--empty-room-model", default="gpt-image-1.5")
    empty_room_vggt.add_argument("--fill-mode", choices=("transparent", "neutral", "black"), default="transparent")
    empty_room_vggt.add_argument("--mask-dilation-px", type=int, default=10)
    empty_room_vggt.add_argument("--mask-feather-px", type=int, default=0)
    empty_room_vggt.add_argument("--include-detection-id", dest="include_detection_ids", action="append", default=[])
    empty_room_vggt.add_argument("--exclude-detection-id", dest="exclude_detection_ids", action="append", default=[])
    empty_room_vggt.add_argument("--allow-rectangular-fallback-masks", action="store_true")
    empty_room_vggt.add_argument("--max-mask-coverage", type=float, default=0.55)
    empty_room_vggt.add_argument("--vggt-backend", choices=("vggt", "fake"), default="vggt")
    empty_room_vggt.add_argument("--vggt-model", default="facebook/VGGT-1B")
    empty_room_vggt.add_argument("--vggt-repo-dir")
    empty_room_vggt.add_argument("--vggt-checkpoint")
    empty_room_vggt.add_argument("--vggt-cache-dir", default="Models/Geometry/VGGT/hf-cache")
    empty_room_vggt.add_argument("--vggt-local-only", action="store_true")
    empty_room_vggt.add_argument("--obj-stride", type=int, default=8)
    empty_room_vggt.add_argument("--mesh-stem", default="empty_room_mesh")
    empty_room_vggt.add_argument("--device", default="auto")
    empty_room_vggt.set_defaults(func=cmd_run_empty_room_vggt)

    fit_vggt_boxes = subparsers.add_parser("fit-vggt-boxes", help="Split VGGT geometry by SAM regions and fit per-object boxes.")
    fit_vggt_boxes.add_argument("--detections", default="Output/Latest/detect/detections.json")
    fit_vggt_boxes.add_argument("--objects", default="Output/Latest/objects")
    fit_vggt_boxes.add_argument("--vggt", default="Output/Latest/objects_vggt")
    fit_vggt_boxes.add_argument("--output", default="Output/Latest/objects_vggt")
    fit_vggt_boxes.add_argument("--box-mode", choices=("auto", "aabb", "obb"), default="auto")
    fit_vggt_boxes.add_argument("--min-valid-points", type=int, default=64)
    fit_vggt_boxes.set_defaults(func=cmd_fit_vggt_boxes)

    fit_empty_room_planes = subparsers.add_parser("fit-empty-room-planes", help="Fit XYZ-aligned structural planes from empty-room VGGT points.")
    fit_empty_room_planes.add_argument("--background", default="Output/Latest/background")
    fit_empty_room_planes.add_argument("--output", default="Output/Latest/background")
    fit_empty_room_planes.add_argument("--stride", type=int, default=8)
    fit_empty_room_planes.add_argument("--mesh-name", default="empty_room_planes.glb")
    fit_empty_room_planes.add_argument("--padding-ratio", type=float, default=0.08)
    fit_empty_room_planes.set_defaults(func=cmd_fit_empty_room_planes)

    choose_supports = subparsers.add_parser("choose-object-supports", help="Choose explicit support planes for object placements.")
    choose_supports.add_argument("--object-geometry", default="Output/Latest/objects_vggt/object_geometry.json")
    choose_supports.add_argument("--planes", default="Output/Latest/background/plane_detections.json")
    choose_supports.add_argument("--detections", default="Output/Latest/detect/detections.json")
    choose_supports.add_argument("--objects", default="Output/Latest/objects")
    choose_supports.add_argument("--output", default="Output/Latest/placement")
    choose_supports.add_argument("--object-mesh-name", default="hunyuan3d_textured.glb")
    choose_supports.add_argument(
        "--placement-orientation",
        choices=("upright", "obb"),
        default="upright",
        help="Use upright visual object meshes by default; obb preserves raw VGGT PCA box rotation.",
    )
    choose_supports.add_argument("--object-scale-factor", type=float, default=0.85)
    choose_supports.add_argument("--include-review", action="store_true")
    choose_supports.set_defaults(func=cmd_choose_object_supports)

    build_fit_targets = subparsers.add_parser("build-object-fit-targets", help="Collect mesh, mask, bbox, VGGT point, and support evidence for placement fitting.")
    build_fit_targets.add_argument("--object-geometry", default="Output/Latest/objects_vggt/object_geometry.json")
    build_fit_targets.add_argument("--supports", default="Output/Latest/placement/object_supports.json")
    build_fit_targets.add_argument("--objects", default="Output/Latest/objects")
    build_fit_targets.add_argument("--output", default="Output/Latest/placement")
    build_fit_targets.add_argument("--object-mesh-name", default="hunyuan3d_textured.glb")
    build_fit_targets.set_defaults(func=cmd_build_object_fit_targets)

    fit_placements = subparsers.add_parser("fit-object-placements", help="Fit object meshes to explicit support records and write object_placements.json.")
    fit_placements.add_argument("--supports", default="Output/Latest/placement/object_supports.json")
    fit_placements.add_argument("--fit-targets", default="Output/Latest/placement/object_fit_targets.json")
    fit_placements.add_argument("--output", default="Output/Latest/placement")
    fit_placements.add_argument("--placement-orientation", choices=("upright", "obb"), default="upright")
    fit_placements.add_argument("--object-scale-factor", type=float, default=0.85)
    fit_placements.add_argument("--no-optimize-placements", action="store_true")
    fit_placements.set_defaults(func=cmd_fit_object_placements)

    compose_scene = subparsers.add_parser("compose-scene", help="Combine empty-room VGGT background geometry, object placements, and object meshes into one GLB scene.")
    compose_scene.add_argument("--background", default="Output/Latest/background/empty_room_mesh.glb")
    compose_scene.add_argument("--objects", default="Output/Latest/objects")
    compose_scene.add_argument("--object-geometry", default="Output/Latest/objects_vggt/object_geometry.json")
    compose_scene.add_argument("--placements", help="Use explicit placement/object_placements.json records instead of fitting directly from object_geometry.json.")
    compose_scene.add_argument("--output", default="Output/Latest/scene")
    compose_scene.add_argument("--output-name", default="scene.glb")
    compose_scene.add_argument("--object-mesh-name", default="hunyuan3d_textured.glb")
    compose_scene.add_argument(
        "--placement-orientation",
        choices=("upright", "obb"),
        default="upright",
        help="Use upright visual object meshes by default; obb preserves raw VGGT PCA box rotation.",
    )
    compose_scene.add_argument(
        "--object-scale-factor",
        type=float,
        default=0.85,
        help="Scale placed object detail meshes inside their VGGT boxes.",
    )
    compose_scene.add_argument(
        "--background-fit",
        choices=("room-corner", "camera-clipped", "placement-bounds", "raw"),
        default="camera-clipped",
        help="Use clipped empty-room VGGT points, a structural room corner, fitted background mesh, or raw GLB background.",
    )
    compose_scene.add_argument("--background-vggt-dir")
    compose_scene.add_argument("--background-stride", type=int, default=16)
    compose_scene.add_argument(
        "--clip-background-masks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Clip source object-mask regions out of the VGGT background. Disabled by default because the AI empty-room image should already contain filled background surfaces.",
    )
    compose_scene.add_argument("--background-clip-dilation-px", type=int, default=8)
    compose_scene.add_argument("--no-snap-objects-to-floor", action="store_true")
    compose_scene.add_argument("--no-optimize-placements", action="store_true")
    compose_scene.add_argument("--source-image")
    compose_scene.add_argument("--background-margin", type=float, default=1.0)
    compose_scene.add_argument(
        "--background-depth-offset",
        type=float,
        default=0.12,
        help="Push the fitted background behind the nearest placed objects in GLB depth units.",
    )
    compose_scene.add_argument(
        "--include-review",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include placements marked needs_review instead of skipping them (default: enabled).",
    )
    compose_scene.set_defaults(func=cmd_compose_scene)

    detect = subparsers.add_parser("detect-shapes", help="Write object proposal detections.json and overlay.png.")
    detect.add_argument("--image", required=True)
    detect.add_argument("--depth")
    detect.add_argument("--edge-map")
    detect.add_argument("--output", required=True)
    detect.add_argument("--backend", choices=PUBLIC_DETECTOR_BACKENDS, default="groundingdino-sam3")
    detect.add_argument("--detector-model")
    detect.add_argument("--detector-weights")
    detect.add_argument("--clip-model-dir")
    detect.add_argument("--device", default="auto")
    detect.add_argument("--primitive-source", choices=PRIMITIVE_SOURCES, default="none")
    detect.add_argument("--confidence", type=float, default=0.25)
    detect.add_argument("--overlap-iou-threshold", type=float, default=0.50)
    detect.add_argument("--rgbd-channel-weights", default="0.25,0.25,0.25,0.25")
    add_open_vocabulary_detector_args(detect)
    add_completion_args(detect)
    detect.set_defaults(func=cmd_detect_shapes)

    complete = subparsers.add_parser("complete-objects", help="Run object crop completion over an existing objects directory.")
    complete.add_argument("--objects", default="Output/Latest/objects")
    add_completion_args(complete)
    complete.set_defaults(
        completion_backend="openai-image",
        completion_model="gpt-5.5",
        completion_device="auto",
        completion_steps=28,
        completion_guidance_scale=6.0,
        completion_quantization="4bit",
    )
    complete.set_defaults(func=cmd_complete_objects)

    reconstruct_objects = subparsers.add_parser(
        "reconstruct-objects",
        help="Run object-level TripoSR mesh reconstruction over completed or masked object crops.",
    )
    reconstruct_objects.add_argument("--objects", default="Output/Latest/objects")
    reconstruct_objects.add_argument("--backend", choices=("hunyuan3d", "triposr", "sam3d-objects"), default="hunyuan3d")
    reconstruct_objects.add_argument("--model-dir", default="Models/Mesh/TripoSR")
    reconstruct_objects.add_argument("--model", default="tencent/Hunyuan3D-2.1")
    reconstruct_objects.add_argument("--device", default="auto")
    reconstruct_objects.add_argument("--source", choices=("auto", "completed", "masked"), default="completed")
    reconstruct_objects.add_argument(
        "--completed-mask-backend",
        choices=("auto", "sam3", "foreground", "original-alpha"),
        default="auto",
        help="How to build completed_mask.png for completed object crops before 3D reconstruction.",
    )
    reconstruct_objects.add_argument("--completed-mask-prompt", help="Override the per-object SAM3 prompt used for completed masks.")
    reconstruct_objects.add_argument("--completed-mask-score-threshold", type=float, default=0.25)
    reconstruct_objects.add_argument("--completed-mask-sam3-repo-dir", default="Models/OpenVocabulary/SAM3/repo")
    reconstruct_objects.add_argument("--completed-mask-sam3-model-dir", default="Models/OpenVocabulary/SAM3/hf")
    reconstruct_objects.add_argument("--max-objects", type=int, default=0)
    reconstruct_objects.add_argument(
        "--with-texture",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run Hunyuan3D paint after shape reconstruction (default: enabled).",
    )
    reconstruct_objects.add_argument("--texture-resolution", type=int, default=512)
    reconstruct_objects.add_argument("--texture-views", type=int, default=6)
    reconstruct_objects.add_argument(
        "--texture-prompt",
        help="Prompt passed to Hunyuan3D Paint for material and texture guidance.",
    )
    reconstruct_objects.add_argument(
        "--texture-reference-mode",
        choices=("original", "masked-crop"),
        default="original",
        help="Image reference passed to Hunyuan3D Paint. original matches the upstream-style input; masked-crop uses SceneForge's masked reference crop.",
    )
    reconstruct_objects.add_argument(
        "--texture-remesh",
        dest="texture_remesh",
        action="store_true",
        default=True,
        help="Let Hunyuan3D Paint remesh before UV wrapping (default).",
    )
    reconstruct_objects.add_argument(
        "--no-texture-remesh",
        dest="texture_remesh",
        action="store_false",
        help="Preserve the reconstructed mesh before Hunyuan3D Paint UV wrapping. This can hang on high-poly meshes.",
    )
    reconstruct_objects.add_argument("--texture-matte-backend", choices=("auto", "bria-rmbg", "mask"), default="auto")
    reconstruct_objects.add_argument("--texture-matte-model-dir", default="Models/Segmentation/BRIA/RMBG-2.0")
    reconstruct_objects.add_argument("--sam3d-objects-repo-dir")
    reconstruct_objects.add_argument("--sam3d-objects-checkpoint")
    reconstruct_objects.add_argument(
        "--sam3d-objects-command",
        help="Optional external command template for SAM 3D Objects. Supports {image}, {mask}, {output}, {object_dir}, {repo_dir}, {checkpoint}, and {device}.",
    )
    reconstruct_objects.set_defaults(func=cmd_reconstruct_objects)


    metrics = subparsers.add_parser("compare-metrics", help="Compare original/generated metrics render folders.")
    metrics.add_argument("--original-metrics", required=True)
    metrics.add_argument("--generated-metrics", required=True)
    metrics.add_argument("--output", required=True)
    metrics.add_argument("--depth-check")
    metrics.set_defaults(func=cmd_compare_metrics)


    return parser


def add_open_vocabulary_detector_args(parser: argparse.ArgumentParser) -> None:
    from Tools.Integration.open_vocab_runtime import prompt_preset_names

    parser.add_argument("--open-vocab-root")
    parser.add_argument("--text-prompt-preset", choices=prompt_preset_names(), default="scene-primitives-v1")
    parser.add_argument("--text-prompt")
    parser.add_argument(
        "--refresh-text-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refresh generated open-vocabulary term list before running (default: enabled)",
    )
    parser.add_argument("--text-prompt-refresh-path", default="Output/Latest/qwen_object_vocab.json")
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--groundingdino-repo-dir")
    parser.add_argument("--groundingdino-config")
    parser.add_argument("--groundingdino-checkpoint")
    parser.add_argument("--ram-repo-dir")
    parser.add_argument("--ram-checkpoint")
    parser.add_argument("--sam3-repo-dir")
    parser.add_argument("--sam3-model-dir")


def add_completion_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--completion-backend", choices=("none", "sdxl-inpaint", "flux-fill", "openai-image"), default="none")
    parser.add_argument("--completion-model", default="Models/Completion/SDXLInpaint")
    parser.add_argument("--completion-device", default="auto")
    parser.add_argument("--completion-steps", type=int, default=24)
    parser.add_argument("--completion-guidance-scale", type=float, default=6.5)
    parser.add_argument("--completion-strength", type=float, default=0.55)
    parser.add_argument("--completion-canvas-size", type=int, default=1024)
    parser.add_argument("--completion-seed", type=int, default=20260528)
    parser.add_argument("--completion-max-objects", type=int, default=0)
    parser.add_argument("--completion-quantization", choices=("none", "8bit", "4bit"), default="4bit")
    parser.add_argument(
        "--completion-context-mode",
        choices=("reference-square", "application-query"),
        default="reference-square",
        help="OpenAI object completion context layout. application-query writes application_query.png beside the object.",
    )


def _resolve_open_vocabulary_runtime_args(args: argparse.Namespace, *, enforce_readiness: bool) -> None:
    from Tools.Integration.open_vocab_runtime import resolve_open_vocab_options

    options = resolve_open_vocab_options(
        backend=getattr(args, "backend", getattr(args, "detector_backend", "")),
        open_vocab_root=getattr(args, "open_vocab_root", None),
        text_prompt=getattr(args, "text_prompt", None),
        text_prompt_preset=getattr(args, "text_prompt_preset", None),
        refresh_text_prompt=getattr(args, "refresh_text_prompt", True),
        text_prompt_refresh_path=getattr(args, "text_prompt_refresh_path", None),
        groundingdino_repo_dir=getattr(args, "groundingdino_repo_dir", None),
        groundingdino_config=getattr(args, "groundingdino_config", None),
        groundingdino_checkpoint=getattr(args, "groundingdino_checkpoint", None),
        ram_repo_dir=getattr(args, "ram_repo_dir", None),
        ram_checkpoint=getattr(args, "ram_checkpoint", None),
        sam3_repo_dir=getattr(args, "sam3_repo_dir", None),
        sam3_model_dir=getattr(args, "sam3_model_dir", None),
    )
    args.text_prompt = options["text_prompt"]
    args.text_prompt_preset = options["text_prompt_preset"]
    for key, value in options["paths"].items():
        if value is not None:
            setattr(args, key, value)
    metadata = dict(options["metadata"])
    if options["enabled"] and enforce_readiness:
        if getattr(args, "open_vocab_root", None):
            from Tools.Integration.open_vocab_readiness import build_report

            report = build_report(
                root_dir=args.open_vocab_root,
                backend=getattr(args, "backend", getattr(args, "detector_backend", "groundingdino-sam3")),
                text_prompt=args.text_prompt,
                ram_repo_dir=getattr(args, "ram_repo_dir", None),
                ram_checkpoint=getattr(args, "ram_checkpoint", None),
                run_import_probe=True,
            )
            metadata["readiness_status"] = report["status"]
            metadata["ready_for_smoke_test"] = bool(report["ready_for_smoke_test"])
            metadata["sam3_access"] = report.get("sam3_access")
            if not report["ready_for_smoke_test"]:
                raise CliError(
                    "Open-vocabulary integration is not ready for reconstruction: "
                    f"{report['status']}. Run audit-open-vocab-readiness for details."
                )
        else:
            metadata["readiness_status"] = "not_checked_explicit_paths"
            metadata["ready_for_smoke_test"] = None
    args.open_vocab_metadata = metadata if options["enabled"] else None


def cmd_check_open_vocab_integration(args: argparse.Namespace) -> int:
    from Tools.Integration.open_vocab_preflight import build_report, print_summary, write_report

    report = build_report(
        backend=args.backend,
        groundingdino_repo_dir=args.groundingdino_repo_dir,
        groundingdino_config=args.groundingdino_config,
        groundingdino_checkpoint=args.groundingdino_checkpoint,
        sam3_repo_dir=args.sam3_repo_dir,
        sam3_model_dir=args.sam3_model_dir,
        ram_repo_dir=getattr(args, "ram_repo_dir", None),
        ram_checkpoint=getattr(args, "ram_checkpoint", None),
        text_prompt=args.text_prompt,
    )
    write_report(report, args.output)
    print_summary(report)
    if not report["ready"]:
        raise CliError(f"Open-vocabulary integration is not ready; wrote {args.output}")
    return 0


def cmd_probe_open_vocab_imports(args: argparse.Namespace) -> int:
    from Tools.Integration.open_vocab_import_probe import build_report, print_summary, write_report

    report = build_report(
        backend=args.backend,
        groundingdino_repo_dir=args.groundingdino_repo_dir,
        sam3_repo_dir=args.sam3_repo_dir,
        ram_repo_dir=getattr(args, "ram_repo_dir", None),
        ram_checkpoint=getattr(args, "ram_checkpoint", None),
    )
    write_report(report, args.output)
    print_summary(report)
    if not report["ready"]:
        raise CliError(f"Open-vocabulary imports are not ready; wrote {args.output}")
    return 0


def cmd_prepare_open_vocab_layout(args: argparse.Namespace) -> int:
    from Tools.Integration.open_vocab_setup import prepare_layout, print_summary

    manifest = prepare_layout(
        args.root,
        create_dirs=not args.no_create_dirs,
        write_script=not args.no_script,
    )
    print_summary(manifest)
    return 0


def cmd_audit_open_vocab_readiness(args: argparse.Namespace) -> int:
    from Tools.Integration.open_vocab_readiness import build_report, print_summary, write_report

    report = build_report(
        root_dir=args.root,
        backend=args.backend,
        text_prompt=args.text_prompt,
        ram_repo_dir=getattr(args, "ram_repo_dir", None),
        ram_checkpoint=getattr(args, "ram_checkpoint", None),
        run_import_probe=not args.skip_import_probe,
    )
    write_report(report, args.output)
    print_summary(report)
    if not report["ready_for_smoke_test"]:
        raise CliError(f"Open-vocabulary integration is not ready for smoke test; wrote {args.output}")
    return 0


def cmd_run_open_vocab_smoke(args: argparse.Namespace) -> int:
    from Tools.Integration.open_vocab_smoke import print_summary, run_smoke_test

    report = run_smoke_test(
        root_dir=args.root,
        backend=args.backend,
        ram_repo_dir=getattr(args, "ram_repo_dir", None),
        ram_checkpoint=getattr(args, "ram_checkpoint", None),
        text_prompt=args.text_prompt,
        output=args.output,
    )
    print_summary(report)
    if report["status"] != "passed":
        raise CliError(f"Open-vocabulary smoke test did not pass; wrote {args.output}")
    return 0



def cmd_render_blend_png(args: argparse.Namespace) -> int:
    blend_path = _require_file(args.reference_blend, "--reference-blend")
    script_path = ROOT / "Tools" / "Scripts" / "render_blend_png.py"
    if not script_path.is_file():
        raise CliError(f"Render script does not exist: {script_path}")
    command = _blender_script_command(
        blender=args.blender,
        blend=blend_path,
        script=script_path,
        script_args=[
            "--output",
            str(args.output),
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--render-samples",
            str(args.render_samples),
            "--render-quality",
            str(args.render_quality),
            "--render-engine",
            str(args.render_engine),
            "--cycles-device-filter",
            str(args.cycles_device_filter),
            "--exposure",
            str(args.exposure),
            "--gamma",
            str(args.gamma),
        ],
    )
    if args.camera_name:
        command.extend(["--camera-name", args.camera_name])
    _run_subprocess(command)
    print(f"Wrote {Path(args.output)}")
    return 0


def cmd_render_scene_camera_view(args: argparse.Namespace) -> int:
    scene_path = _require_file(args.scene, "--scene")
    script_path = ROOT / "Tools" / "Scripts" / "render_scene_camera_view.py"
    if not script_path.is_file():
        raise CliError(f"Render script does not exist: {script_path}")
    script_args = [
        str(scene_path),
        str(args.output),
    ]
    if args.alignment_report:
        script_args.extend(["--alignment-report", str(_require_file(args.alignment_report, "--alignment-report"))])
    if args.width:
        script_args.extend(["--width", str(args.width)])
    if args.height:
        script_args.extend(["--height", str(args.height)])
    if args.fov_degrees:
        script_args.extend(["--fov-degrees", str(args.fov_degrees)])
    script_args.extend(["--camera-mode", str(args.camera_mode)])
    command = _blender_script_command(
        blender=args.blender,
        script=script_path,
        script_args=script_args,
    )
    _run_subprocess(command)
    print(f"Wrote {Path(args.output)}")
    return 0


def cmd_run_vggt(args: argparse.Namespace) -> int:
    from SceneGeometry.VGGT.pipeline import run_vggt_image_geometry

    report = run_vggt_image_geometry(
        image_path=args.image,
        output_dir=args.output,
        backend=args.backend,
        model=args.model,
        repo_dir=args.vggt_repo_dir,
        checkpoint=args.vggt_checkpoint,
        device=args.device,
        local_only=args.vggt_local_only,
        cache_dir=args.vggt_cache_dir,
        obj_stride=args.obj_stride,
        mesh_stem=args.mesh_stem,
    )
    output_dir = Path(args.output)
    print(f"Wrote {output_dir / 'vggt_geometry.json'}")
    print(f"Wrote {output_dir / report['artifacts']['depth_png']}")
    print(f"Wrote {output_dir / report['artifacts']['points_xyz']}")
    print(f"Wrote {output_dir / report['artifacts']['mesh_obj']}")
    print(f"Wrote {output_dir / report['artifacts']['mesh_glb']}")
    return 0


def cmd_generate_empty_room(args: argparse.Namespace) -> int:
    from BackgroundReconstruction.empty_room import generate_empty_room, parse_detection_id_set

    include_ids = set()
    for value in args.include_detection_ids:
        include_ids.update(parse_detection_id_set(value))
    exclude_ids = set()
    for value in args.exclude_detection_ids:
        exclude_ids.update(parse_detection_id_set(value))
    report = generate_empty_room(
        image_path=args.image,
        detections_path=args.detections,
        objects_dir=args.objects,
        output_dir=args.output,
        backend=args.empty_room_backend,
        model=args.empty_room_model,
        fill_mode=args.fill_mode,
        mask_dilation_px=args.mask_dilation_px,
        mask_feather_px=args.mask_feather_px,
        include_detection_ids=include_ids,
        exclude_detection_ids=exclude_ids,
        allow_rectangular_fallback_masks=args.allow_rectangular_fallback_masks,
        max_mask_coverage=args.max_mask_coverage,
    )
    output_dir = Path(args.output)
    print(f"Wrote {output_dir / 'empty_room_metadata.json'}")
    print(f"Wrote {output_dir / 'empty_room_mask.png'}")
    print(f"Wrote {output_dir / 'empty_room_openai_input.png'}")
    print(f"Wrote {output_dir / 'empty_room.png'}")
    if report.get("needs_review"):
        print(f"Empty-room output needs review: {', '.join(report.get('warnings', []))}")
    return 0


def cmd_run_empty_room_vggt(args: argparse.Namespace) -> int:
    from BackgroundReconstruction.empty_room import generate_empty_room
    from SceneGeometry.VGGT.pipeline import run_vggt_image_geometry

    include_ids, exclude_ids = _parse_detection_id_args(args)
    output_dir = Path(args.output)
    empty_room_report = generate_empty_room(
        image_path=args.image,
        detections_path=args.detections,
        objects_dir=args.objects,
        output_dir=output_dir,
        backend=args.empty_room_backend,
        model=args.empty_room_model,
        fill_mode=args.fill_mode,
        mask_dilation_px=args.mask_dilation_px,
        mask_feather_px=args.mask_feather_px,
        include_detection_ids=include_ids,
        exclude_detection_ids=exclude_ids,
        allow_rectangular_fallback_masks=args.allow_rectangular_fallback_masks,
        max_mask_coverage=args.max_mask_coverage,
    )
    vggt_report = run_vggt_image_geometry(
        image_path=output_dir / "empty_room.png",
        output_dir=output_dir,
        backend=args.vggt_backend,
        model=args.vggt_model,
        repo_dir=args.vggt_repo_dir,
        checkpoint=args.vggt_checkpoint,
        device=args.device,
        local_only=args.vggt_local_only,
        cache_dir=args.vggt_cache_dir,
        obj_stride=args.obj_stride,
        mesh_stem=args.mesh_stem,
    )
    print(f"Wrote {output_dir / 'empty_room_metadata.json'}")
    print(f"Wrote {output_dir / 'empty_room.png'}")
    print(f"Wrote {output_dir / 'vggt_geometry.json'}")
    print(f"Wrote {output_dir / vggt_report['artifacts']['mesh_obj']}")
    print(f"Wrote {output_dir / vggt_report['artifacts']['mesh_glb']}")
    if empty_room_report.get("needs_review"):
        print(f"Empty-room output needs review: {', '.join(empty_room_report.get('warnings', []))}")
    return 0


def _parse_detection_id_args(args: argparse.Namespace) -> tuple[set[int], set[int]]:
    from BackgroundReconstruction.empty_room import parse_detection_id_set

    include_ids = set()
    for value in getattr(args, "include_detection_ids", []):
        include_ids.update(parse_detection_id_set(value))
    exclude_ids = set()
    for value in getattr(args, "exclude_detection_ids", []):
        exclude_ids.update(parse_detection_id_set(value))
    return include_ids, exclude_ids


def cmd_fit_vggt_boxes(args: argparse.Namespace) -> int:
    from SceneGeometry.VGGT.regions import fit_vggt_boxes

    report = fit_vggt_boxes(
        detections_path=args.detections,
        objects_dir=args.objects,
        vggt_dir=args.vggt,
        output_dir=args.output,
        box_mode=args.box_mode,
        min_valid_points=args.min_valid_points,
    )
    output_dir = Path(args.output)
    print(f"Wrote {output_dir / 'object_geometry.json'}")
    print(f"Wrote {report['artifacts']['boxes_obj']}")
    print(
        "Fitted "
        f"{report['summary']['fit_count']}/{report['summary']['detection_count']} objects "
        f"({report['summary']['failed_count']} failed, {report['summary']['needs_review_count']} needs review)"
    )
    return 0


def cmd_fit_empty_room_planes(args: argparse.Namespace) -> int:
    from SceneGeometry.Planes.empty_room import fit_empty_room_planes

    report = fit_empty_room_planes(
        background_dir=args.background,
        output_dir=args.output,
        stride=args.stride,
        mesh_name=args.mesh_name,
        padding_ratio=args.padding_ratio,
    )
    print(f"Wrote {Path(args.output) / 'plane_detections.json'}")
    print(f"Wrote {report['artifacts']['planes_glb']}")
    print(f"Fitted {len(report['planes'])} XYZ-aligned empty-room planes")
    return 0


def cmd_choose_object_supports(args: argparse.Namespace) -> int:
    from SceneComposition.placement import choose_object_supports

    report = choose_object_supports(
        object_geometry_path=args.object_geometry,
        planes_path=args.planes,
        detections_path=args.detections,
        objects_dir=args.objects,
        output_dir=args.output,
        object_mesh_name=args.object_mesh_name,
        include_review=args.include_review,
        placement_orientation=args.placement_orientation,
        object_scale_factor=args.object_scale_factor,
    )
    output_dir = Path(args.output)
    print(f"Wrote {output_dir / 'object_supports.json'}")
    print(
        "Chose supports for "
        f"{report['summary']['accepted_count']}/{report['summary']['object_count']} objects "
        f"({report['summary']['needs_review_count']} needs review)"
    )
    return 0


def cmd_build_object_fit_targets(args: argparse.Namespace) -> int:
    from SceneComposition.placement import build_object_fit_targets

    report = build_object_fit_targets(
        object_geometry_path=args.object_geometry,
        supports_path=args.supports,
        objects_dir=args.objects,
        output_dir=args.output,
        object_mesh_name=args.object_mesh_name,
    )
    output_dir = Path(args.output)
    print(f"Wrote {output_dir / 'object_fit_targets.json'}")
    print(
        "Built fit targets for "
        f"{report['summary']['ready_count']}/{report['summary']['object_count']} objects "
        f"({report['summary']['failed_count']} failed, {report['summary']['needs_review_count']} needs review)"
    )
    return 0


def cmd_fit_object_placements(args: argparse.Namespace) -> int:
    from SceneComposition.placement import fit_object_placements

    report = fit_object_placements(
        supports_path=args.supports,
        fit_targets_path=args.fit_targets,
        output_dir=args.output,
        placement_orientation=args.placement_orientation,
        object_scale_factor=args.object_scale_factor,
        optimize_placements=not args.no_optimize_placements,
    )
    output_dir = Path(args.output)
    print(f"Wrote {output_dir / 'object_placements.json'}")
    print(f"Wrote {output_dir / 'placement_quality.json'}")
    print(
        "Fitted placements for "
        f"{report['summary']['accepted_count']}/{report['summary']['placement_count']} objects "
        f"({report['summary']['failed_count']} failed, {report['summary']['needs_review_count']} needs review)"
    )
    return 0


def cmd_compose_scene(args: argparse.Namespace) -> int:
    from SceneComposition.composer import compose_scene

    report = compose_scene(
        background_path=args.background,
        objects_dir=args.objects,
        object_geometry_path=args.object_geometry,
        placements_path=args.placements,
        output_dir=args.output,
        output_name=args.output_name,
        object_mesh_name=args.object_mesh_name,
        include_review=args.include_review,
        placement_orientation=args.placement_orientation,
        object_scale_factor=args.object_scale_factor,
        background_fit=args.background_fit,
        background_margin=args.background_margin,
        background_depth_offset=args.background_depth_offset,
        background_vggt_dir=args.background_vggt_dir,
        background_stride=args.background_stride,
        clip_background_masks=args.clip_background_masks,
        background_clip_dilation_px=args.background_clip_dilation_px,
        snap_objects_to_floor=not args.no_snap_objects_to_floor,
        optimize_placements=not args.no_optimize_placements,
        source_image_path=args.source_image,
    )
    output_dir = Path(args.output)
    print(f"Wrote {report['artifacts']['scene_glb']}")
    print(f"Wrote {output_dir / 'scene_alignment.json'}")
    if report["artifacts"].get("input_vs_projection_overlay"):
        print(f"Wrote {report['artifacts']['input_vs_projection_overlay']}")
    print(
        "Composed "
        f"{report['summary']['composed_count']}/{report['summary']['placement_count']} objects "
        f"({report['summary']['skipped_count']} skipped, {report['summary']['failed_count']} failed)"
    )
    return 0


def cmd_detect_shapes(args: argparse.Namespace) -> int:
    from Segmentation.factory import DetectShapesBackendConfig, build_detect_shapes_runtime
    from ShapeDetection.pipeline import run_shape_detection

    _resolve_open_vocabulary_runtime_args(args, enforce_readiness=bool(getattr(args, "open_vocab_root", None)))
    runtime = build_detect_shapes_runtime(
        DetectShapesBackendConfig(
            backend=args.backend,
            depth=args.depth,
            edge_map=args.edge_map,
            detector_model=args.detector_model,
            detector_weights=args.detector_weights,
            clip_model_dir=args.clip_model_dir,
            device=args.device,
            primitive_source=args.primitive_source,
            confidence=args.confidence,
            overlap_iou_threshold=args.overlap_iou_threshold,
            rgbd_channel_weights=args.rgbd_channel_weights,
            text_prompt=args.text_prompt,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            text_prompt_preset=args.text_prompt_preset,
            open_vocab_metadata=getattr(args, "open_vocab_metadata", None),
            groundingdino_repo_dir=args.groundingdino_repo_dir,
            groundingdino_config=args.groundingdino_config,
            groundingdino_checkpoint=args.groundingdino_checkpoint,
            ram_repo_dir=args.ram_repo_dir,
            ram_checkpoint=args.ram_checkpoint,
            sam3_repo_dir=args.sam3_repo_dir,
            sam3_model_dir=args.sam3_model_dir,
        ),
        require_file=_require_file,
        require_dir=_require_dir,
    )
    run_shape_detection(
        image_path=args.image,
        output_dir=args.output,
        segmenter=runtime.segmenter,
        classifier=runtime.classifier,
        model_info=runtime.model_info,
        completion_backend=args.completion_backend,
        completion_model=args.completion_model,
        completion_device=args.completion_device,
        completion_steps=args.completion_steps,
        completion_guidance_scale=args.completion_guidance_scale,
        completion_strength=args.completion_strength,
        completion_canvas_size=args.completion_canvas_size,
        completion_seed=args.completion_seed,
        completion_max_objects=args.completion_max_objects,
        completion_quantization=args.completion_quantization,
        completion_context_mode=args.completion_context_mode,
    )
    print(f"Wrote {Path(args.output) / 'detections.json'}")
    print(f"Wrote {Path(args.output) / 'overlay.png'}")
    return 0


def cmd_complete_objects(args: argparse.Namespace) -> int:
    from ShapeDetection.pipeline import run_object_completion

    run_object_completion(
        objects_dir=Path(args.objects),
        backend=args.completion_backend,
        model_dir=args.completion_model,
        device=args.completion_device,
        steps=args.completion_steps,
        guidance_scale=args.completion_guidance_scale,
        strength=args.completion_strength,
        canvas_size=args.completion_canvas_size,
        seed=args.completion_seed,
        max_objects=args.completion_max_objects,
        quantization=args.completion_quantization,
        context_mode=args.completion_context_mode,
    )
    print(f"Wrote {Path(args.objects) / 'completion_manifest.json'}")
    return 0


def cmd_reconstruct_objects(args: argparse.Namespace) -> int:
    if args.backend == "hunyuan3d":
        from ObjectReconstruction.hunyuan3d_objects import run_hunyuan3d_object_reconstruction

        run_hunyuan3d_object_reconstruction(
            objects_dir=Path(args.objects),
            model=args.model,
            device=args.device,
            source=args.source,
            max_objects=args.max_objects,
            with_texture=args.with_texture,
            texture_resolution=args.texture_resolution,
            texture_views=args.texture_views,
            texture_use_remesh=args.texture_remesh,
            texture_prompt=args.texture_prompt,
            texture_reference_mode=args.texture_reference_mode,
            texture_matte_backend=args.texture_matte_backend,
            texture_matte_model_dir=args.texture_matte_model_dir,
            completed_mask_backend=args.completed_mask_backend,
            completed_mask_sam3_repo_dir=args.completed_mask_sam3_repo_dir,
            completed_mask_sam3_model_dir=args.completed_mask_sam3_model_dir,
            completed_mask_prompt=args.completed_mask_prompt,
            completed_mask_score_threshold=args.completed_mask_score_threshold,
        )
        print(f"Wrote {Path(args.objects) / 'hunyuan3d_manifest.json'}")
        return 0
    if args.backend == "triposr":
        model_dir = _require_dir(args.model_dir, "--model-dir")
        from ObjectReconstruction.triposr_objects import run_triposr_object_reconstruction

        run_triposr_object_reconstruction(
            objects_dir=Path(args.objects),
            model_dir=model_dir,
            device=args.device,
            source=args.source,
            max_objects=args.max_objects,
            completed_mask_backend=args.completed_mask_backend,
            completed_mask_prompt=args.completed_mask_prompt,
        )
        print(f"Wrote {Path(args.objects) / 'triposr_manifest.json'}")
        return 0
    if args.backend == "sam3d-objects":
        from ObjectReconstruction.sam3d_objects import run_sam3d_objects_reconstruction

        run_sam3d_objects_reconstruction(
            objects_dir=Path(args.objects),
            repo_dir=args.sam3d_objects_repo_dir,
            checkpoint=args.sam3d_objects_checkpoint,
            command_template=args.sam3d_objects_command,
            model=args.model,
            device=args.device,
            source=args.source,
            max_objects=args.max_objects,
            completed_mask_backend=args.completed_mask_backend,
        )
        print(f"Wrote {Path(args.objects) / 'sam3d_objects_manifest.json'}")
        return 0
    raise CliError(f"Unsupported object reconstruction backend: {args.backend}")



def cmd_compare_metrics(args: argparse.Namespace) -> int:
    from OutputWriter.metrics_summary import write_metrics_comparison_summary

    write_metrics_comparison_summary(
        original_metrics_dir=args.original_metrics,
        generated_metrics_dir=args.generated_metrics,
        output_dir=args.output,
        depth_check_path=args.depth_check,
    )
    print(f"Wrote {Path(args.output) / 'summary.json'}")
    return 0


def _blender_script_command(*, blender: str, script: Path, script_args: list[str], blend: Path | None = None) -> list[str]:
    command = [blender, "--background"]
    if blend is not None:
        command.append(str(blend))
    command.extend(["--python", str(script), "--"])
    command.extend(script_args)
    return command


def _run_subprocess(command: list[str]) -> int:
    try:
        completed = subprocess.run(command, text=True, check=False)
    except FileNotFoundError as exc:
        raise CliError(f"Executable not found: {command[0]}") from exc
    if completed.returncode != 0:
        raise CliError(f"Command failed with exit code {completed.returncode}: {' '.join(command)}")
    return 0


def _require_file(value: str | Path | None, label: str) -> Path:
    if value is None or str(value) == "":
        raise CliError(f"{label} is required")
    path = Path(value)
    if not path.is_file():
        raise CliError(f"{label} does not exist: {path}")
    return path


def _require_dir(value: str | Path | None, label: str) -> Path:
    if value is None or str(value) == "":
        raise CliError(f"{label} is required")
    path = Path(value)
    if not path.is_dir():
        raise CliError(f"{label} does not exist: {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    if argv == [] or (argv is None and len(sys.argv) == 1):
        from Runtime.guided_cli import guided_scene_main

        return guided_scene_main(main)
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    try:
        return int(args.func(args) or 0)
    except CliError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"SceneForge error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
