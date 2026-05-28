from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ObjectEnrichment.geometry_classifier import classify_geometry as _geometry_classifier_startup_import
from PrimitiveFitting.pipeline import run_primitive_fitting


ROOT = Path(__file__).resolve().parent
PUBLIC_DETECTOR_BACKENDS = ("depth-edge", "depth-edge-object", "sam3", "groundingdino-sam3", "rgb-yolo", "rgbd-yolo", "real")
PUBLIC_EDGE_BACKENDS = ("none", "simple", "dexined")
PUBLIC_MESH_BACKENDS = ("none", "triposr")
PUBLIC_WIREFRAME_BACKENDS = ("none", "hawp")
PRIMITIVE_SOURCES = ("none", "detector-label", "clip")


class CliError(RuntimeError):
    """User-facing CLI failure with an exit code of 2."""


class NoEdgeProvider:
    backend = "none"
    model_dir = None

    def detect_edges(self, image):
        from EdgeDetection.types import EdgeResult
        from PIL import Image

        return EdgeResult(image=Image.new("L", image.size, 0), backend=self.backend, model_dir=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SceneForge modular image/depth to primitive scene pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)


    preflight = subparsers.add_parser("check-open-vocab-integration", help="Preflight local GroundingDINO/SAM3 repo and model paths.")
    preflight.add_argument("--backend", choices=("sam3", "groundingdino-sam3"), default="groundingdino-sam3")
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
    preflight.add_argument("--text-prompt", default="chair . table . box . sphere . cylinder . cone . plane . foreground object .")
    preflight.add_argument("--output", default="Output/Latest/open_vocab_preflight.json")
    preflight.set_defaults(func=cmd_check_open_vocab_integration)


    import_probe = subparsers.add_parser("probe-open-vocab-imports", help="Probe local GroundingDINO/SAM3 imports without loading checkpoints.")
    import_probe.add_argument("--backend", choices=("sam3", "groundingdino-sam3"), default="groundingdino-sam3")
    import_probe.add_argument("--groundingdino-repo-dir", default="Models/OpenVocabulary/GroundingDINO/repo")
    import_probe.add_argument("--sam3-repo-dir", default="Models/OpenVocabulary/SAM3/repo")
    import_probe.add_argument("--output", default="Output/Latest/open_vocab_import_probe.json")
    import_probe.set_defaults(func=cmd_probe_open_vocab_imports)


    prepare_open_vocab = subparsers.add_parser("prepare-open-vocab-layout", help="Create local GroundingDINO/SAM3 layout and setup manifest.")
    prepare_open_vocab.add_argument("--root", default="Models/OpenVocabulary")
    prepare_open_vocab.add_argument("--no-create-dirs", action="store_true")
    prepare_open_vocab.add_argument("--no-script", action="store_true")
    prepare_open_vocab.set_defaults(func=cmd_prepare_open_vocab_layout)


    readiness = subparsers.add_parser("audit-open-vocab-readiness", help="Run non-inference readiness checks for GroundingDINO/SAM3 integration.")
    readiness.add_argument("--root", default="Models/OpenVocabulary")
    readiness.add_argument("--backend", choices=("sam3", "groundingdino-sam3"), default="groundingdino-sam3")
    readiness.add_argument("--text-prompt", default="chair . table . box . sphere . cylinder . cone . plane . foreground object .")
    readiness.add_argument("--skip-import-probe", action="store_true")
    readiness.add_argument("--output", default="Output/Latest/open_vocab_readiness.json")
    readiness.set_defaults(func=cmd_audit_open_vocab_readiness)


    smoke = subparsers.add_parser("run-open-vocab-smoke", help="Run guarded GroundingDINO/SAM3 detect-shapes smoke test.")
    smoke.add_argument("--root", default="Models/OpenVocabulary")
    smoke.add_argument("--backend", choices=("sam3", "groundingdino-sam3"), default="groundingdino-sam3")
    smoke.add_argument("--text-prompt", default="chair . table . box . sphere . cylinder . cone . plane . foreground object .")
    smoke.add_argument("--output", default="Output/Latest/open_vocab_smoke.json")
    smoke.set_defaults(func=cmd_run_open_vocab_smoke)

    detect = subparsers.add_parser("detect-shapes", help="Write detections.json and overlay.png.")
    detect.add_argument("--image", required=True)
    detect.add_argument("--depth")
    detect.add_argument("--edge-map")
    detect.add_argument("--output", required=True)
    detect.add_argument("--backend", choices=PUBLIC_DETECTOR_BACKENDS, default="depth-edge-object")
    detect.add_argument("--detector-model")
    detect.add_argument("--detector-weights")
    detect.add_argument("--clip-model-dir")
    detect.add_argument("--device", default="auto")
    detect.add_argument("--primitive-source", choices=PRIMITIVE_SOURCES, default="none")
    detect.add_argument("--confidence", type=float, default=0.25)
    detect.add_argument("--overlap-iou-threshold", type=float, default=0.50)
    detect.add_argument("--rgbd-channel-weights", default="0.25,0.25,0.25,0.25")
    add_open_vocabulary_detector_args(detect)
    detect.set_defaults(func=cmd_detect_shapes)

    enrich = subparsers.add_parser("enrich-objects", help="Fuse depth, edge, wireframe, and mesh evidence.")
    enrich.add_argument("--image", required=True)
    enrich.add_argument("--depth", required=True)
    enrich.add_argument("--detections", required=True)
    enrich.add_argument("--output", required=True)
    add_provider_args(enrich)
    add_enrichment_tuning_args(enrich)
    enrich.set_defaults(func=cmd_enrich_objects)

    fit = subparsers.add_parser("fit-primitives", help="Fit detections to geometric 3D primitive proxies.")
    fit.add_argument("--image", required=True)
    fit.add_argument("--depth", required=True)
    fit.add_argument("--detections", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--enrichment")
    fit.add_argument("--fov-degrees", type=float, default=70.0)
    fit.add_argument("--sensor-fit", default="horizontal")
    fit.add_argument("--camera-shift-x", type=float, default=0.0)
    fit.add_argument("--camera-shift-y", type=float, default=0.0)
    fit.add_argument("--near-depth", type=float, default=1.0)
    fit.add_argument("--far-depth", type=float, default=6.0)
    fit.add_argument("--blender", default="blender")
    fit.add_argument("--reference-blend")
    fit.add_argument("--final-layout", choices=("camera", "ground", "original-camera"), default="camera")
    fit.add_argument("--no-depth-refinement", action="store_true")
    fit.add_argument("--require-quality-gate", action="store_true")
    fit.set_defaults(func=cmd_fit_primitives)

    reconstruct = subparsers.add_parser(
        "reconstruct-scene",
        help="Render a reference blend, detect objects, enrich evidence, and fit primitives.",
    )
    reconstruct.add_argument("--reference-blend", required=True)
    reconstruct.add_argument("--output", default="Output/Latest")
    reconstruct.add_argument("--camera-name")
    reconstruct.add_argument("--detector-backend", choices=PUBLIC_DETECTOR_BACKENDS, default="depth-edge-object")
    reconstruct.add_argument("--detector-model")
    reconstruct.add_argument("--detector-weights")
    reconstruct.add_argument("--primitive-source", choices=PRIMITIVE_SOURCES, default="none")
    reconstruct.add_argument("--detector-confidence", type=float, default=0.20)
    reconstruct.add_argument("--detector-overlap-iou-threshold", type=float, default=0.50)
    reconstruct.add_argument("--rgbd-channel-weights", default="0.25,0.25,0.25,0.25")
    add_open_vocabulary_detector_args(reconstruct)
    add_provider_args(reconstruct)
    add_enrichment_tuning_args(reconstruct)
    reconstruct.add_argument("--final-layout", choices=("camera", "ground", "original-camera"), default="camera")
    reconstruct.add_argument("--blender", default="blender")
    reconstruct.add_argument("--width", type=int, default=640)
    reconstruct.add_argument("--height", type=int, default=640)
    reconstruct.add_argument("--render-samples", type=int, default=16)
    reconstruct.add_argument("--near-depth", type=float, default=1.0)
    reconstruct.add_argument("--far-depth", type=float, default=8.0)
    reconstruct.add_argument("--fov-degrees", type=float, default=70.0)
    reconstruct.add_argument("--resume", action="store_true")
    reconstruct.add_argument("--force", action="store_true")
    reconstruct.add_argument("--no-archive", action="store_true")
    reconstruct.add_argument("--no-depth-refinement", action="store_true")
    reconstruct.add_argument("--require-quality-gate", action="store_true")
    reconstruct.set_defaults(func=cmd_reconstruct_scene)

    overlay = subparsers.add_parser("render-evidence-overlay", help="Render a fused evidence audit image.")
    overlay.add_argument("--image", required=True)
    overlay.add_argument("--detections", required=True)
    overlay.add_argument("--enrichment", required=True)
    overlay.add_argument("--output", required=True)
    overlay.add_argument("--edge-map")
    overlay.set_defaults(func=cmd_render_evidence_overlay)

    metrics = subparsers.add_parser("compare-metrics", help="Compare original/generated metrics render folders.")
    metrics.add_argument("--original-metrics", required=True)
    metrics.add_argument("--generated-metrics", required=True)
    metrics.add_argument("--output", required=True)
    metrics.add_argument("--depth-check")
    metrics.set_defaults(func=cmd_compare_metrics)

    train = subparsers.add_parser("train-rgbd-yolo", help="Legacy RGBD YOLO comparison training.")
    train.add_argument("--data", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--model", default="Configs/YOLO/yolo26l_seg_rgbd.yaml")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--imgsz", type=int, default=640)
    train.add_argument("--batch", type=int, default=8)
    train.add_argument("--device")
    train.add_argument("--seed", type=int, default=20260525)
    train.add_argument("--patience", type=int, default=5)
    train.add_argument("--lr0", type=float)
    train.add_argument("--resume-from")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--rgbd-channel-weights", default="0.25,0.25,0.25,0.25")
    train.set_defaults(func=cmd_train_rgbd_yolo)

    evaluate = subparsers.add_parser("eval-rgbd-yolo", help="Legacy RGBD YOLO comparison evaluation.")
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--weights", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--imgsz", type=int, default=640)
    evaluate.add_argument("--batch", type=int, default=8)
    evaluate.add_argument("--device")
    evaluate.add_argument("--split", choices=("train", "val", "test"), default="test")
    evaluate.add_argument("--rgbd-channel-weights", default="0.25,0.25,0.25,0.25")
    evaluate.set_defaults(func=cmd_eval_rgbd_yolo)

    dataset = subparsers.add_parser("generate-rgbd-dataset", help="Generate synthetic detector-neutral RGBD data.")
    add_dataset_args(dataset)
    dataset.add_argument("--curriculum-stage", type=int, choices=tuple(range(1, 11)), required=True)
    dataset.add_argument("--images-per-class", type=int)
    dataset.add_argument("--shards", type=parse_shards, default="auto")
    dataset.set_defaults(func=cmd_generate_rgbd_dataset)

    target_dataset = subparsers.add_parser(
        "generate-target-rgbd-dataset",
        help="Generate target RGBD data from a labeled .blend file.",
    )
    target_dataset.add_argument("--reference-blend", required=True)
    add_dataset_args(target_dataset)
    target_dataset.add_argument("--camera-jitter", type=float, default=0.28)
    target_dataset.add_argument("--target-jitter", type=float, default=0.12)
    target_dataset.add_argument("--fov-jitter-degrees", type=float, default=3.0)
    target_dataset.add_argument("--object-rotation-degrees", type=float, default=0.0)
    target_dataset.add_argument("--random-object-rotation", action="store_true")
    target_dataset.add_argument("--eval-only", action="store_true")
    target_dataset.add_argument("--exact-first", action="store_true")
    target_dataset.add_argument("--shards", type=parse_shards, default="auto")
    target_dataset.set_defaults(func=cmd_generate_target_rgbd_dataset)

    train_instance = subparsers.add_parser("train-instance-detector", help="Train the Primitive3D instance detector.")
    train_instance.add_argument("--manifest", required=True)
    train_instance.add_argument("--config", required=True)
    train_instance.add_argument("--output", required=True)
    train_instance.add_argument("--epochs", type=int, default=8)
    train_instance.add_argument("--batch", type=int, default=8)
    train_instance.add_argument("--device", default="auto")
    train_instance.set_defaults(func=cmd_train_instance_detector)

    eval_instance = subparsers.add_parser("eval-instance-detector", help="Evaluate the Primitive3D instance detector.")
    eval_instance.add_argument("--manifest", required=True)
    eval_instance.add_argument("--model", required=True)
    eval_instance.add_argument("--config", required=True)
    eval_instance.add_argument("--output", required=True)
    eval_instance.add_argument("--split", choices=("train", "val", "test"), default="test")
    eval_instance.add_argument("--device", default="auto")
    eval_instance.set_defaults(func=cmd_eval_instance_detector)

    return parser


def add_open_vocabulary_detector_args(parser: argparse.ArgumentParser) -> None:
    from Tools.Integration.open_vocab_runtime import prompt_preset_names

    parser.add_argument("--open-vocab-root")
    parser.add_argument("--text-prompt-preset", choices=prompt_preset_names(), default="scene-primitives-v1")
    parser.add_argument("--text-prompt")
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--groundingdino-repo-dir")
    parser.add_argument("--groundingdino-config")
    parser.add_argument("--groundingdino-checkpoint")
    parser.add_argument("--sam3-repo-dir")
    parser.add_argument("--sam3-model-dir")


def _resolve_open_vocabulary_runtime_args(args: argparse.Namespace, *, enforce_readiness: bool) -> None:
    from Tools.Integration.open_vocab_runtime import resolve_open_vocab_options

    options = resolve_open_vocab_options(
        backend=getattr(args, "backend", getattr(args, "detector_backend", "")),
        open_vocab_root=getattr(args, "open_vocab_root", None),
        text_prompt=getattr(args, "text_prompt", None),
        text_prompt_preset=getattr(args, "text_prompt_preset", None),
        groundingdino_repo_dir=getattr(args, "groundingdino_repo_dir", None),
        groundingdino_config=getattr(args, "groundingdino_config", None),
        groundingdino_checkpoint=getattr(args, "groundingdino_checkpoint", None),
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


def add_provider_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--edge-backend", choices=PUBLIC_EDGE_BACKENDS, default="simple")
    parser.add_argument("--edge-model-dir")
    parser.add_argument("--mesh-backend", choices=PUBLIC_MESH_BACKENDS, default="none")
    parser.add_argument("--mesh-model-dir")
    parser.add_argument("--wireframe-backend", choices=PUBLIC_WIREFRAME_BACKENDS, default="none")
    parser.add_argument("--wireframe-model-dir")


def add_enrichment_tuning_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--max-objects", type=int, default=32)
    parser.add_argument("--max-mesh-objects", type=int, default=16)
    parser.add_argument("--min-edge-mask-pixels", type=int, default=64)
    parser.add_argument("--min-mesh-mask-pixels", type=int, default=256)
    parser.add_argument("--min-wireframe-mask-pixels", type=int, default=64)
    parser.add_argument("--edge-timeout-seconds", type=int, default=120)
    parser.add_argument("--mesh-timeout-seconds", type=int, default=180)
    parser.add_argument("--wireframe-timeout-seconds", type=int, default=120)


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", default="Datasets/PrimitiveShapes")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--render-samples", type=int, default=16)
    parser.add_argument("--train-split", type=float, default=0.70)
    parser.add_argument("--val-split", type=float, default=0.20)
    parser.add_argument("--depth-near", type=float, default=1.0)
    parser.add_argument("--depth-far", type=float, default=8.0)
    parser.add_argument("--dark-background-ratio", type=float, default=0.35)
    parser.add_argument("--material-variation", type=float, default=0.85)
    parser.add_argument("--finish", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--blender", default="blender")


def parse_shards(value: str) -> str | int:
    if value == "auto":
        return value
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("--shards must be auto or a positive integer")
    return parsed


def cmd_check_open_vocab_integration(args: argparse.Namespace) -> int:
    from Tools.Integration.open_vocab_preflight import build_report, print_summary, write_report

    report = build_report(
        backend=args.backend,
        groundingdino_repo_dir=args.groundingdino_repo_dir,
        groundingdino_config=args.groundingdino_config,
        groundingdino_checkpoint=args.groundingdino_checkpoint,
        sam3_repo_dir=args.sam3_repo_dir,
        sam3_model_dir=args.sam3_model_dir,
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
        text_prompt=args.text_prompt,
        output=args.output,
    )
    print_summary(report)
    if report["status"] != "passed":
        raise CliError(f"Open-vocabulary smoke test did not pass; wrote {args.output}")
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
    )
    print(f"Wrote {Path(args.output) / 'detections.json'}")
    print(f"Wrote {Path(args.output) / 'overlay.png'}")
    return 0


def cmd_enrich_objects(args: argparse.Namespace) -> int:
    from ObjectEnrichment.pipeline import run_object_enrichment

    edge_provider, mesh_provider, wireframe_provider = build_evidence_providers(args)
    run_object_enrichment(
        image_path=args.image,
        depth_path=args.depth,
        detections_path=args.detections,
        output_dir=args.output,
        edge_provider=edge_provider,
        mesh_provider=mesh_provider,
        wireframe_provider=wireframe_provider,
        device=args.device,
        seed=args.seed,
        max_objects=args.max_objects,
        max_mesh_objects=args.max_mesh_objects,
        min_edge_mask_pixels=args.min_edge_mask_pixels,
        min_mesh_mask_pixels=args.min_mesh_mask_pixels,
        min_wireframe_mask_pixels=args.min_wireframe_mask_pixels,
        edge_timeout_seconds=args.edge_timeout_seconds,
        mesh_timeout_seconds=args.mesh_timeout_seconds,
        wireframe_timeout_seconds=args.wireframe_timeout_seconds,
    )
    print(f"Wrote {Path(args.output) / 'object_enrichment.json'}")
    return 0


def cmd_fit_primitives(args: argparse.Namespace) -> int:
    run_primitive_fitting(
        image_path=args.image,
        depth_path=args.depth,
        detections_path=args.detections,
        output_dir=args.output,
        enrichment_path=args.enrichment,
        fov_degrees=args.fov_degrees,
        sensor_fit=args.sensor_fit,
        camera_shift_x=args.camera_shift_x,
        camera_shift_y=args.camera_shift_y,
        near_depth=args.near_depth,
        far_depth=args.far_depth,
        blender_executable=args.blender,
        reference_blend_path=args.reference_blend,
        final_layout=args.final_layout,
        depth_refinement_enabled=not args.no_depth_refinement,
    )
    if args.require_quality_gate:
        _require_fit_quality_gate(Path(args.output) / "primitive_fits.json")
    print(f"Wrote {Path(args.output) / 'primitive_fits.json'}")
    print(f"Wrote {Path(args.output) / 'fit_overlay.png'}")
    print(f"Wrote {Path(args.output) / 'fitted_scene.blend'}")
    return 0


def cmd_reconstruct_scene(args: argparse.Namespace) -> int:
    output_dir = Path(args.output)
    _preflight_reconstruct(args)
    build_evidence_providers(args)
    _prepare_latest_output(args, output_dir)
    _write_run_status(output_dir, "running", stage="prepare", open_vocab=getattr(args, "open_vocab_metadata", None))
    try:
        render_info = _run_reconstruct_render(args, output_dir)
        _write_run_status(output_dir, "running", stage="detect", render=render_info, open_vocab=getattr(args, "open_vocab_metadata", None))
        _run_reconstruct_detect(args, output_dir, render_info)
        _write_run_status(output_dir, "running", stage="enrich", render=render_info, open_vocab=getattr(args, "open_vocab_metadata", None))
        _run_reconstruct_enrich(args, output_dir, render_info)
        _write_run_status(output_dir, "running", stage="fit", render=render_info, open_vocab=getattr(args, "open_vocab_metadata", None))
        _run_reconstruct_fit(args, output_dir, fov_degrees=float(render_info.get("fov_degrees", args.fov_degrees)))
        if args.require_quality_gate:
            _require_fit_quality_gate(output_dir / "fit" / "primitive_fits.json")
        _write_run_status(output_dir, "complete", stage="complete", render=render_info, open_vocab=getattr(args, "open_vocab_metadata", None))
    except Exception as exc:
        _write_run_status(
            output_dir,
            "failed",
            stage="failed",
            error=str(exc),
            open_vocab=getattr(args, "open_vocab_metadata", None),
        )
        raise
    print(f"Wrote {output_dir / 'run_status.json'}")
    return 0


def cmd_render_evidence_overlay(args: argparse.Namespace) -> int:
    from OutputWriter.evidence_overlay import write_evidence_overlay

    write_evidence_overlay(
        image_path=args.image,
        detections_path=args.detections,
        enrichment_path=args.enrichment,
        output_path=args.output,
        edge_map_path=args.edge_map,
    )
    print(f"Wrote {Path(args.output)}")
    return 0


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


def cmd_train_rgbd_yolo(args: argparse.Namespace) -> int:
    from Tools.Training.rgbd_yolo import train_rgbd_yolo

    output = train_rgbd_yolo(
        data_yaml=args.data,
        model_yaml=args.model,
        output_weights=args.output,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        seed=args.seed,
        patience=args.patience,
        lr0=args.lr0,
        resume_from=args.resume_from,
        resume=args.resume,
        channel_weights=args.rgbd_channel_weights,
    )
    print(f"Wrote {output}")
    return 0


def cmd_eval_rgbd_yolo(args: argparse.Namespace) -> int:
    from Tools.Training.rgbd_yolo import evaluate_rgbd_yolo

    output = evaluate_rgbd_yolo(
        data_yaml=args.data,
        weights_path=args.weights,
        output_dir=args.output,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split=args.split,
        channel_weights=args.rgbd_channel_weights,
    )
    print(f"Wrote {output}")
    return 0


def cmd_generate_rgbd_dataset(args: argparse.Namespace) -> int:
    shard_count = _resolve_auto_shards(args.shards, args.count)
    command = _blender_script_command(
        blender=args.blender,
        script=ROOT / "Tools" / "Dataset" / "generate_primitives_dataset.py",
        script_args=_dataset_script_args(args, shard_count=shard_count, exclude={"material_variation"}),
    )
    return _run_subprocess(command)


def cmd_generate_target_rgbd_dataset(args: argparse.Namespace) -> int:
    shard_count = _resolve_auto_shards(args.shards, args.count)
    command = _blender_script_command(
        blender=args.blender,
        blend=Path(args.reference_blend),
        script=ROOT / "Tools" / "Dataset" / "generate_blend_target_dataset.py",
        script_args=_dataset_script_args(args, shard_count=shard_count, exclude={"reference_blend"}),
    )
    return _run_subprocess(command)


def cmd_train_instance_detector(args: argparse.Namespace) -> int:
    from Tools.Training.instance_detector import write_training_scaffold

    checkpoint = write_training_scaffold(
        manifest_path=args.manifest,
        config_path=args.config,
        output_dir=args.output,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
    )
    print(f"Wrote {checkpoint}")
    return 0


def cmd_eval_instance_detector(args: argparse.Namespace) -> int:
    from Tools.Training.instance_detector import write_eval_scaffold

    summary = write_eval_scaffold(
        manifest_path=args.manifest,
        model_path=args.model,
        config_path=args.config,
        output_dir=args.output,
        split=args.split,
        device=args.device,
    )
    print(f"Wrote {summary}")
    return 0


def build_evidence_providers(args: argparse.Namespace):
    edge_provider = build_edge_provider(args.edge_backend, args.edge_model_dir, args.device)
    mesh_provider = build_mesh_provider(args.mesh_backend, args.mesh_model_dir, args.device)
    wireframe_provider = build_wireframe_provider(
        args.wireframe_backend,
        args.wireframe_model_dir,
        args.device,
        args.wireframe_timeout_seconds,
    )
    return edge_provider, mesh_provider, wireframe_provider


def build_edge_provider(backend: str, model_dir: str | None, device: str | None):
    if backend == "none":
        return NoEdgeProvider()
    if backend == "simple":
        from EdgeDetection.simple_edge_provider import SimpleEdgeProvider

        return SimpleEdgeProvider()
    if backend == "dexined":
        model_path = _require_dir(model_dir, "--edge-model-dir")
        from EdgeDetection.dexined_provider import DexiNedEdgeProvider

        return DexiNedEdgeProvider(model_dir=model_path, device=device)
    raise CliError(f"Unsupported edge backend: {backend}")


def build_mesh_provider(backend: str, model_dir: str | None, device: str | None):
    if backend == "none":
        from MeshReconstruction.no_mesh_provider import NoMeshProvider

        return NoMeshProvider()
    if backend == "triposr":
        model_path = _require_dir(model_dir, "--mesh-model-dir")
        from MeshReconstruction.triposr_provider import TripoSRMeshProvider

        return TripoSRMeshProvider(model_dir=model_path, device=device)
    raise CliError(f"Unsupported mesh backend: {backend}")


def build_wireframe_provider(backend: str, model_dir: str | None, device: str | None, timeout_seconds: int):
    if backend == "none":
        from WireframeDetection.types import NoWireframeProvider

        return NoWireframeProvider()
    if backend == "hawp":
        model_path = _require_dir(model_dir, "--wireframe-model-dir")
        from WireframeDetection.hawp_provider import HawpWireframeProvider

        return HawpWireframeProvider(model_dir=model_path, device=device, timeout_seconds=timeout_seconds)
    raise CliError(f"Unsupported wireframe backend: {backend}")


def _preflight_reconstruct(args: argparse.Namespace) -> None:
    _require_file(args.reference_blend, "--reference-blend")
    _resolve_open_vocabulary_runtime_args(
        args,
        enforce_readiness=args.detector_backend in {"sam3", "groundingdino-sam3"},
    )
    if args.detector_backend in {"rgb-yolo", "rgbd-yolo", "real"} and not args.detector_model:
        _require_file(args.detector_weights, "--detector-weights")
    if args.detector_backend == "sam3":
        _require_dir(args.sam3_repo_dir, "--sam3-repo-dir")
        _require_dir(args.sam3_model_dir, "--sam3-model-dir")
    if args.detector_backend == "groundingdino-sam3":
        _require_dir(args.groundingdino_repo_dir, "--groundingdino-repo-dir")
        _require_file(args.groundingdino_config, "--groundingdino-config")
        _require_file(args.groundingdino_checkpoint, "--groundingdino-checkpoint")
        _require_dir(args.sam3_repo_dir, "--sam3-repo-dir")
        _require_dir(args.sam3_model_dir, "--sam3-model-dir")
    if args.detector_model:
        _require_file(args.detector_model, "--detector-model")
    if args.final_layout == "original-camera":
        _require_file(args.reference_blend, "--reference-blend")


def _run_reconstruct_render(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    render_dir = output_dir / "render"
    render_dir.mkdir(parents=True, exist_ok=True)
    image_path = render_dir / "image.png"
    depth_path = render_dir / "depth.png"
    metadata_path = render_dir / "camera.json"
    script_path = ROOT / "Tools" / "Scripts" / "render_reference_rgbd.py"
    if not script_path.is_file():
        raise CliError(
            "Reference rendering is not wired yet: expected Tools/Scripts/render_reference_rgbd.py. "
            "This is the open integration point for Blender or another renderer."
        )
    command = [
        args.blender,
        "--background",
        str(args.reference_blend),
        "--python",
        str(script_path),
        "--",
        "--image-output",
        str(image_path),
        "--depth-output",
        str(depth_path),
        "--camera-output",
        str(metadata_path),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--render-samples",
        str(args.render_samples),
        "--near-depth",
        str(args.near_depth),
        "--far-depth",
        str(args.far_depth),
    ]
    if args.camera_name:
        command.extend(["--camera-name", args.camera_name])
    _run_subprocess(command)
    return {
        "image_path": str(image_path),
        "depth_path": str(depth_path),
        "camera_metadata_path": str(metadata_path),
        "fov_degrees": args.fov_degrees,
    }


def _run_reconstruct_detect(args: argparse.Namespace, output_dir: Path, render_info: dict[str, Any]) -> None:
    from Segmentation.factory import ReconstructDetectionBackendConfig, build_reconstruct_detection_runtime
    from ShapeDetection.pipeline import run_shape_detection

    edge_provider = build_edge_provider(args.edge_backend, args.edge_model_dir, args.device)
    runtime = build_reconstruct_detection_runtime(
        ReconstructDetectionBackendConfig(
            detector_backend=args.detector_backend,
            detector_model=args.detector_model,
            detector_weights=args.detector_weights,
            requested_device=args.device,
            device=args.device,
            primitive_source=args.primitive_source,
            detector_confidence=args.detector_confidence,
            detector_overlap_iou_threshold=args.detector_overlap_iou_threshold,
            rgbd_channel_weights=args.rgbd_channel_weights,
            max_objects=args.max_objects,
            text_prompt=args.text_prompt,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            text_prompt_preset=args.text_prompt_preset,
            open_vocab_metadata=getattr(args, "open_vocab_metadata", None),
            groundingdino_repo_dir=args.groundingdino_repo_dir,
            groundingdino_config=args.groundingdino_config,
            groundingdino_checkpoint=args.groundingdino_checkpoint,
            sam3_repo_dir=args.sam3_repo_dir,
            sam3_model_dir=args.sam3_model_dir,
        ),
        image_path=Path(render_info["image_path"]),
        depth_path=Path(render_info["depth_path"]),
        edge_provider=edge_provider,
        require_file=_require_file,
    )
    temp_dir = output_dir / ".tmp_detect"
    run_shape_detection(
        image_path=render_info["image_path"],
        output_dir=temp_dir,
        segmenter=runtime.segmenter,
        classifier=runtime.classifier,
        model_info=runtime.model_info,
    )
    _replace_stage_output(temp_dir, output_dir / "detect")


def _run_reconstruct_enrich(args: argparse.Namespace, output_dir: Path, render_info: dict[str, Any]) -> None:
    from ObjectEnrichment.pipeline import run_object_enrichment
    from OutputWriter.evidence_overlay import write_evidence_overlay

    edge_provider, mesh_provider, wireframe_provider = build_evidence_providers(args)
    temp_dir = output_dir / ".tmp_enrich"
    run_object_enrichment(
        image_path=render_info["image_path"],
        depth_path=render_info["depth_path"],
        detections_path=output_dir / "detect" / "detections.json",
        output_dir=temp_dir,
        edge_provider=edge_provider,
        mesh_provider=mesh_provider,
        wireframe_provider=wireframe_provider,
        device=args.device,
        seed=args.seed,
        max_objects=args.max_objects,
        max_mesh_objects=args.max_mesh_objects,
        min_edge_mask_pixels=args.min_edge_mask_pixels,
        min_mesh_mask_pixels=args.min_mesh_mask_pixels,
        min_wireframe_mask_pixels=args.min_wireframe_mask_pixels,
        edge_timeout_seconds=args.edge_timeout_seconds,
        mesh_timeout_seconds=args.mesh_timeout_seconds,
        wireframe_timeout_seconds=args.wireframe_timeout_seconds,
    )
    _replace_stage_output(temp_dir, output_dir / "enrich")
    write_evidence_overlay(
        image_path=render_info["image_path"],
        detections_path=output_dir / "detect" / "detections.json",
        enrichment_path=output_dir / "enrich" / "object_enrichment.json",
        output_path=output_dir / "enrich" / "evidence_overlay.png",
    )


def _run_reconstruct_fit(args: argparse.Namespace, output_dir: Path, fov_degrees: float = 70.0) -> None:
    render_dir = output_dir / "render"
    fit_dir = output_dir / "fit"
    run_primitive_fitting(
        image_path=render_dir / "image.png",
        depth_path=render_dir / "depth.png",
        detections_path=output_dir / "detect" / "detections.json",
        output_dir=fit_dir,
        enrichment_path=output_dir / "enrich" / "object_enrichment.json",
        fov_degrees=fov_degrees,
        near_depth=args.near_depth,
        far_depth=args.far_depth,
        blender_executable=args.blender,
        reference_blend_path=args.reference_blend,
        final_layout=args.final_layout,
        depth_refinement_enabled=not getattr(args, "no_depth_refinement", False),
    )
    nested_blend = fit_dir / "fitted_scene.blend"
    top_blend = output_dir / "fitted_scene.blend"
    if nested_blend.exists():
        top_blend.unlink(missing_ok=True)
        shutil.move(str(nested_blend), str(top_blend))


def _prepare_latest_output(args: argparse.Namespace, output_dir: Path) -> None:
    manifest = _run_manifest(args)
    if args.resume:
        manifest_path = output_dir / "run_manifest.json"
        if not manifest_path.is_file():
            raise CliError("Cannot resume: run_manifest.json is missing.")
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if current != manifest:
            raise CliError("Cannot resume: run_manifest.json does not match requested run.")
        return

    if output_dir.exists() and any(output_dir.iterdir()):
        if args.no_archive:
            raise CliError(f"Output directory is not empty: {output_dir}")
        archive_root = output_dir.parent / "Archive"
        archive_root.mkdir(parents=True, exist_ok=True)
        archive_dir = archive_root / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        shutil.move(str(output_dir), str(archive_dir))

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _replace_stage_output(temp_dir: Path, final_dir: Path) -> None:
    if final_dir.exists():
        if final_dir.is_dir():
            shutil.rmtree(final_dir)
        else:
            final_dir.unlink()
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(temp_dir), str(final_dir))


def _run_manifest(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "reference_blend": str(getattr(args, "reference_blend", "")),
        "camera_name": getattr(args, "camera_name", None),
        "detector_backend": getattr(args, "detector_backend", "depth-edge-object"),
        "detector_model": getattr(args, "detector_model", None),
        "detector_weights": getattr(args, "detector_weights", None),
        "edge_backend": getattr(args, "edge_backend", "simple"),
        "mesh_backend": getattr(args, "mesh_backend", "none"),
        "wireframe_backend": getattr(args, "wireframe_backend", "none"),
        "final_layout": getattr(args, "final_layout", "camera"),
        "seed": getattr(args, "seed", 20260525),
        "width": getattr(args, "width", 640),
        "height": getattr(args, "height", 640),
        "near_depth": getattr(args, "near_depth", 1.0),
        "far_depth": getattr(args, "far_depth", 8.0),
    }


def _write_run_status(output_dir: Path, status: str, **extra: Any) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": status,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    (output_dir / "run_status.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_auto_shards(value: str | int, count: int) -> int:
    if value != "auto":
        return int(value)
    if count <= 1:
        return 1
    return max(1, min(32, round(count / 125)))


def _dataset_script_args(args: argparse.Namespace, *, shard_count: int, exclude: set[str] | None = None) -> list[str]:
    exclude = set(exclude or ())
    exclude.update({"command", "func", "blender", "shards"})
    values = vars(args)
    script_args: list[str] = []
    for key, value in sorted(values.items()):
        if key in exclude or value is None or value is False:
            continue
        option = f"--{key.replace('_', '-')}"
        if value is True:
            script_args.append(option)
        else:
            script_args.extend([option, str(value)])
    script_args.extend(["--shard-count", str(shard_count)])
    return script_args


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


def _require_fit_quality_gate(report_path: Path) -> None:
    if not report_path.is_file():
        raise CliError(f"Fit quality report does not exist: {report_path}")
    data = json.loads(report_path.read_text(encoding="utf-8"))
    summary = data.get("model_info", {}).get("fit_quality_summary", {})
    if summary and summary.get("quality_gate_passed") is False:
        raise CliError("Fit quality gate failed.")


def main(argv: list[str] | None = None) -> int:
    if argv == [] or (argv is None and len(sys.argv) == 1):
        from Runtime.guided_cli import guided_scene_main

        return guided_scene_main(main)
    parser = build_parser()
    args = parser.parse_args(argv)
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
