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
    reconstruct_objects.add_argument("--backend", choices=("hunyuan3d", "triposr"), default="hunyuan3d")
    reconstruct_objects.add_argument("--model-dir", default="Models/Mesh/TripoSR")
    reconstruct_objects.add_argument("--model", default="tencent/Hunyuan3D-2.1")
    reconstruct_objects.add_argument("--device", default="auto")
    reconstruct_objects.add_argument("--source", choices=("auto", "completed", "masked"), default="completed")
    reconstruct_objects.add_argument("--max-objects", type=int, default=0)
    reconstruct_objects.add_argument("--with-texture", action="store_true", help="Run Hunyuan3D paint after shape reconstruction.")
    reconstruct_objects.add_argument("--texture-resolution", type=int, default=512)
    reconstruct_objects.add_argument("--texture-views", type=int, default=6)
    reconstruct_objects.add_argument("--no-texture-remesh", action="store_true")
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
            texture_use_remesh=not args.no_texture_remesh,
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
        )
        print(f"Wrote {Path(args.objects) / 'triposr_manifest.json'}")
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
