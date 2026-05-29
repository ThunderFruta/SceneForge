from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from Runtime.device import resolve_torch_device
from ShapeDetection.unassigned_classifier import UnassignedPrimitiveClassifier


RequireFile = Callable[[str | Path | None, str], Path]
RequireDir = Callable[[str | Path | None, str], Path]


@dataclass(frozen=True)
class DetectionRuntime:
    segmenter: object
    classifier: object
    model_info: dict


@dataclass(frozen=True)
class DetectShapesBackendConfig:
    backend: str
    depth: str | None
    edge_map: str | None
    detector_model: str | None
    detector_weights: str | None
    clip_model_dir: str | None
    device: str | None
    primitive_source: str
    confidence: float
    overlap_iou_threshold: float
    rgbd_channel_weights: str
    text_prompt: str = "object . plane . foreground object ."
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    text_prompt_preset: str | None = None
    open_vocab_metadata: dict | None = None
    ram_repo_dir: str | None = None
    ram_checkpoint: str | None = None
    groundingdino_repo_dir: str | None = None
    groundingdino_config: str | None = None
    groundingdino_checkpoint: str | None = None
    sam3_repo_dir: str | None = None
    sam3_model_dir: str | None = None


def build_detect_shapes_runtime(
    config: DetectShapesBackendConfig,
    *,
    require_file: RequireFile,
    require_dir: RequireDir,
) -> DetectionRuntime:
    if config.backend == "sam3":
        return sam3_runtime(config, require_dir=require_dir)

    if config.backend == "groundingdino-sam3":
        return groundingdino_sam3_runtime(config, require_file=require_file, require_dir=require_dir)

    if config.backend == "ram-groundingdino-sam3":
        return ram_groundingdino_sam3_runtime(config, require_file=require_file, require_dir=require_dir)

    raise ValueError(
        f"Unsupported detector backend: {config.backend}. "
        "The depth-edge, Primitive3D, RGB YOLO, and RGBD YOLO detector paths are retired; "
        "use sam3, groundingdino-sam3, or ram-groundingdino-sam3."
    )


def open_vocabulary_runtime(segmenter: object, config: DetectShapesBackendConfig) -> DetectionRuntime:
    backend_info = segmenter.backend_info
    open_vocab_metadata = dict(config.open_vocab_metadata or {})
    model_info = {
        "detector_backend": backend_info.name,
        "detector_architecture": backend_info.architecture,
        "detector_input_channels": list(backend_info.input_channels),
        "detector_backend_info": backend_info.to_dict(),
        "classifier_backend": "unassigned",
        "primitive_label_policy": backend_info.primitive_label_policy,
        "legacy_yolo": backend_info.legacy,
        "text_prompt": getattr(segmenter, "text_prompt", None),
        "text_prompt_preset": config.text_prompt_preset,
        "open_vocab_metadata": open_vocab_metadata,
        "open_vocab_sources": {
            "groundingdino_repo_dir": config.groundingdino_repo_dir,
            "groundingdino_config": config.groundingdino_config,
            "groundingdino_checkpoint": config.groundingdino_checkpoint,
            "ram_repo_dir": config.ram_repo_dir,
            "ram_checkpoint": config.ram_checkpoint,
            "sam3_repo_dir": config.sam3_repo_dir,
            "sam3_model_dir": config.sam3_model_dir,
        },
        "groundingdino_box_threshold": config.box_threshold,
        "groundingdino_text_threshold": config.text_threshold,
        "sam_mask_mode": "box_prompt_with_rectangle_fallback" if "groundingdino" in backend_info.name else "text_prompt",
    }
    return DetectionRuntime(
        segmenter=segmenter,
        classifier=UnassignedPrimitiveClassifier(),
        model_info=model_info,
    )


def sam3_runtime(config: DetectShapesBackendConfig, *, require_dir: RequireDir) -> DetectionRuntime:
    from Segmentation.sam3_segmenter import Sam3Segmenter

    segmenter = Sam3Segmenter(
        repo_dir=require_dir(config.sam3_repo_dir, "--sam3-repo-dir"),
        model_dir=require_dir(config.sam3_model_dir, "--sam3-model-dir"),
        text_prompt=config.text_prompt,
        score_threshold=config.confidence,
        device=resolve_torch_device(config.device),
    )
    return open_vocabulary_runtime(segmenter, config)


def groundingdino_sam3_runtime(
    config: DetectShapesBackendConfig,
    *,
    require_file: RequireFile,
    require_dir: RequireDir,
) -> DetectionRuntime:
    from Segmentation.groundingdino_sam3_segmenter import GroundingDinoSam3Segmenter

    segmenter = GroundingDinoSam3Segmenter(
        groundingdino_repo_dir=require_dir(config.groundingdino_repo_dir, "--groundingdino-repo-dir"),
        groundingdino_config=require_file(config.groundingdino_config, "--groundingdino-config"),
        groundingdino_checkpoint=require_file(config.groundingdino_checkpoint, "--groundingdino-checkpoint"),
        sam3_repo_dir=require_dir(config.sam3_repo_dir, "--sam3-repo-dir"),
        sam3_model_dir=require_dir(config.sam3_model_dir, "--sam3-model-dir"),
        text_prompt=config.text_prompt,
        box_threshold=config.box_threshold,
        text_threshold=config.text_threshold,
        score_threshold=config.confidence,
        device=resolve_torch_device(config.device),
    )
    return open_vocabulary_runtime(segmenter, config)


def ram_groundingdino_sam3_runtime(
    config: DetectShapesBackendConfig,
    *,
    require_file: RequireFile,
    require_dir: RequireDir,
) -> DetectionRuntime:
    from Segmentation.ram_groundingdino_sam3_segmenter import RamGroundingDinoSam3Segmenter

    segmenter = RamGroundingDinoSam3Segmenter(
        ram_repo_dir=require_dir(config.ram_repo_dir, "--ram-repo-dir"),
        ram_checkpoint=require_file(config.ram_checkpoint, "--ram-checkpoint"),
        groundingdino_repo_dir=require_dir(config.groundingdino_repo_dir, "--groundingdino-repo-dir"),
        groundingdino_config=require_file(config.groundingdino_config, "--groundingdino-config"),
        groundingdino_checkpoint=require_file(config.groundingdino_checkpoint, "--groundingdino-checkpoint"),
        sam3_repo_dir=require_dir(config.sam3_repo_dir, "--sam3-repo-dir"),
        sam3_model_dir=require_dir(config.sam3_model_dir, "--sam3-model-dir"),
        text_prompt=config.text_prompt,
        box_threshold=config.box_threshold,
        text_threshold=config.text_threshold,
        score_threshold=config.confidence,
        device=resolve_torch_device(config.device),
    )
    return open_vocabulary_runtime(segmenter, config)
