from __future__ import annotations

import gc
import json
from datetime import datetime, timezone
from pathlib import Path

from Input.Image.image_loader import load_rgb_image
from OutputWriter.object_masks import write_object_masks
from OutputWriter.overlay import write_overlay
from OutputWriter.report_writer import write_report
from SceneGeometry.coordinate_contract import camera_fusion_contract, load_fusion_contract_from_camera_metadata
from Segmentation.proposal_quality import (
    filter_open_vocab_segments,
    is_open_vocab_model_info,
    summarize_open_vocab_proposals,
)
from ShapeDetection.report import DetectionReport, ObjectShapeDetection


def run_shape_detection(
    image_path: str | Path,
    output_dir: str | Path,
    segmenter,
    classifier,
    model_info: dict,
    completion_backend: str = "none",
    completion_model: str | Path | None = None,
    completion_device: str | None = "auto",
    completion_steps: int = 24,
    completion_guidance_scale: float = 6.5,
    completion_strength: float = 0.55,
    completion_canvas_size: int = 1024,
    completion_seed: int = 20260528,
    completion_max_objects: int = 16,
    completion_quantization: str = "4bit",
) -> DetectionReport:
    resolved_image_path = Path(image_path)
    image = load_rgb_image(resolved_image_path)
    segments = segmenter.detect(image)
    filtering_stats = None
    if is_open_vocab_model_info(model_info):
        segments, filtering_stats = filter_open_vocab_segments(
            segments,
            image_width=image.width,
            image_height=image.height,
        )
        proposal_quality = summarize_open_vocab_proposals(
            segments,
            image_width=image.width,
            image_height=image.height,
        )
        proposal_quality["filtering"] = filtering_stats
    else:
        proposal_quality = None

    runtime_debug: dict = {}
    ram_tags = getattr(segmenter, "last_ram_tags", None)
    if ram_tags is not None:
        runtime_debug["ram_tags"] = list(ram_tags)
    grounding_prompt = getattr(segmenter, "last_grounding_prompt", None)
    if grounding_prompt is not None:
        runtime_debug["grounding_prompt"] = grounding_prompt
    ram_error = getattr(segmenter, "last_ram_error", None)
    if ram_error:
        runtime_debug["ram_error"] = ram_error

    objects: list[ObjectShapeDetection] = []
    for index, segment in enumerate(segments, start=1):
        prediction = classifier.classify(image, segment)
        objects.append(
            ObjectShapeDetection(
                id=index,
                bbox_xyxy=segment.bbox_xyxy,
                mask_polygon=segment.mask_polygon,
                detector_label=segment.detector_label,
                detector_confidence=segment.detector_confidence,
                primitive_label=prediction.label,
                primitive_confidence=prediction.confidence,
                primitive_label_source=prediction.source,
            )
        )

    model_info_with_time = dict(model_info)
    if proposal_quality is not None:
        model_info_with_time["proposal_quality"] = proposal_quality
    if runtime_debug:
        model_info_with_time["runtime_debug"] = runtime_debug
    model_info_with_time["fusion_contract"] = load_source_fusion_contract(resolved_image_path, image.width, image.height)
    model_info_with_time["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    report = DetectionReport(
        image_path=str(resolved_image_path),
        image_width=image.width,
        image_height=image.height,
        objects=objects,
        model_info=model_info_with_time,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    write_report(report, output_path / "detections.json")
    if proposal_quality is not None:
        (output_path / "proposal_quality.json").write_text(
            json.dumps(proposal_quality, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    write_overlay(image, objects, output_path / "overlay.png")
    objects_dir = object_masks_output_dir(output_path)
    write_object_masks(image, objects, objects_dir)
    release_detection_runtime(segmenter, classifier)
    run_object_completion(
        objects_dir=objects_dir,
        backend=completion_backend,
        model_dir=completion_model,
        device=completion_device,
        steps=completion_steps,
        guidance_scale=completion_guidance_scale,
        strength=completion_strength,
        canvas_size=completion_canvas_size,
        seed=completion_seed,
        max_objects=completion_max_objects,
        quantization=completion_quantization,
    )
    return report


def release_detection_runtime(segmenter, classifier) -> None:
    del segmenter
    del classifier
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def run_object_completion(
    *,
    objects_dir: Path,
    backend: str,
    model_dir: str | Path | None,
    device: str | None,
    steps: int,
    guidance_scale: float,
    strength: float,
    canvas_size: int,
    seed: int,
    max_objects: int,
    quantization: str,
) -> None:
    if backend == "none":
        return
    if backend == "flux-fill":
        if model_dir is None:
            raise ValueError("--completion-model is required for --completion-backend flux-fill")
        model_path = Path(model_dir)
        if not model_path.is_dir():
            raise ValueError(f"--completion-model does not exist or is not a directory: {model_path}")
        from ObjectCompletion.flux_fill import run_flux_object_completion

        run_flux_object_completion(
            objects_dir=objects_dir,
            model_dir=model_path,
            device=device,
            steps=steps,
            guidance_scale=guidance_scale,
            strength=strength,
            canvas_size=canvas_size,
            seed=seed,
            max_objects=max_objects,
            quantization=quantization,
        )
        return
    if backend != "sdxl-inpaint":
        raise ValueError(f"Unsupported object completion backend: {backend}")
    if model_dir is None:
        raise ValueError("--completion-model is required for --completion-backend sdxl-inpaint")
    model_path = Path(model_dir)
    if not model_path.is_dir():
        raise ValueError(f"--completion-model does not exist or is not a directory: {model_path}")
    from ObjectCompletion.sdxl_inpaint import run_sdxl_object_completion

    run_sdxl_object_completion(
        objects_dir=objects_dir,
        model_dir=model_path,
        device=device,
        steps=steps,
        guidance_scale=guidance_scale,
        strength=strength,
        canvas_size=canvas_size,
        seed=seed,
        max_objects=max_objects,
    )


def object_masks_output_dir(output_path: Path) -> Path:
    if output_path.name == "detect":
        return output_path.parent / "objects"
    return output_path / "objects"


def load_source_fusion_contract(image_path: Path, width: int, height: int) -> dict:
    camera_path = image_path.parent / "camera.json"
    if camera_path.is_file():
        try:
            return load_fusion_contract_from_camera_metadata(json.loads(camera_path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            pass
    return camera_fusion_contract(image_width=width, image_height=height)
