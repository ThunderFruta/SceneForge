from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from EdgeDetection.types import EdgeProvider
from Runtime.device import resolve_torch_device
from Segmentation.depth_edge_segmenter import DepthEdgeSegmenter, EdgeReasonedDepthSegmenter
from ShapeDetection.depth_geometry_classifier import DepthGeometryPrimitiveClassifier
from Segmentation.learned_depth_edge_segmenter import LearnedDepthEdgeSegmenter
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
    groundingdino_repo_dir: str | None = None
    groundingdino_config: str | None = None
    groundingdino_checkpoint: str | None = None
    sam3_repo_dir: str | None = None
    sam3_model_dir: str | None = None


@dataclass(frozen=True)
class ReconstructDetectionBackendConfig:
    detector_backend: str
    detector_model: str | None
    detector_weights: str | None
    requested_device: str | None
    device: str | None
    primitive_source: str
    detector_confidence: float
    detector_overlap_iou_threshold: float
    rgbd_channel_weights: str
    max_objects: int
    text_prompt: str = "object . plane . foreground object ."
    box_threshold: float = 0.35
    text_threshold: float = 0.25
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
    if config.detector_model:
        depth_path = require_file(config.depth, "--depth")
        model_path = require_file(config.detector_model, "--detector-model")
        edge_map_path = require_file(config.edge_map, "--edge-map") if config.edge_map else None
        segmenter = LearnedDepthEdgeSegmenter(
            model_path=model_path,
            depth_path=depth_path,
            edge_path=edge_map_path,
            device=resolve_torch_device(config.device),
        )
        return learned_depth_edge_runtime(
            segmenter=segmenter,
            depth_path=depth_path,
            edge_map_path=edge_map_path,
            edge_backend=None,
        )

    if config.backend in {"depth-edge", "depth-edge-object"}:
        depth_path = require_file(config.depth, "--depth")
        edge_map_path = require_file(config.edge_map, "--edge-map") if config.edge_map else None
        segmenter_class = EdgeReasonedDepthSegmenter if config.backend == "depth-edge-object" else DepthEdgeSegmenter
        segmenter = segmenter_class(depth_path=depth_path, edge_path=edge_map_path)
        return depth_edge_runtime(
            segmenter=segmenter,
            depth_path=depth_path,
            edge_map_path=edge_map_path,
            edge_backend=None,
            classifier=DepthGeometryPrimitiveClassifier(depth_path) if config.backend == "depth-edge-object" else None,
            classifier_backend="depth-geometry-weak" if config.backend == "depth-edge-object" else "unassigned",
        )

    if config.backend == "sam3":
        return sam3_runtime(config, require_dir=require_dir)

    if config.backend == "groundingdino-sam3":
        return groundingdino_sam3_runtime(config, require_file=require_file, require_dir=require_dir)

    if config.backend in {"real", "rgb-yolo"}:
        return legacy_rgb_yolo_runtime(config, require_file=require_file, require_dir=require_dir)

    if config.backend == "rgbd-yolo":
        return legacy_rgbd_yolo_runtime(
            detector_weights=config.detector_weights,
            depth_path=require_file(config.depth, "--depth"),
            confidence=config.confidence,
            requested_device=config.device,
            device=config.device,
            overlap_iou_threshold=config.overlap_iou_threshold,
            primitive_source=config.primitive_source,
            rgbd_channel_weights=config.rgbd_channel_weights,
            require_file=require_file,
        )

    raise ValueError(f"Unsupported detector backend: {config.backend}")


def build_reconstruct_detection_runtime(
    config: ReconstructDetectionBackendConfig,
    *,
    image_path: Path,
    depth_path: Path,
    edge_provider: EdgeProvider | None,
    require_file: RequireFile,
) -> DetectionRuntime:
    del image_path
    if config.detector_model:
        model_path = require_file(config.detector_model, "--detector-model")
        segmenter = LearnedDepthEdgeSegmenter(
            model_path=model_path,
            depth_path=depth_path,
            edge_provider=edge_provider,
            device=resolve_torch_device(config.device),
            max_components=config.max_objects,
        )
        return learned_depth_edge_runtime(
            segmenter=segmenter,
            depth_path=depth_path,
            edge_map_path=None,
            edge_backend=getattr(edge_provider, "backend", None),
        )

    if config.detector_backend in {"depth-edge", "depth-edge-object"}:
        segmenter_class = EdgeReasonedDepthSegmenter if config.detector_backend == "depth-edge-object" else DepthEdgeSegmenter
        segmenter = segmenter_class(
            depth_path=depth_path,
            edge_provider=edge_provider,
            max_components=config.max_objects,
        )
        return depth_edge_runtime(
            segmenter=segmenter,
            depth_path=depth_path,
            edge_map_path=None,
            edge_backend=getattr(edge_provider, "backend", None),
            classifier=DepthGeometryPrimitiveClassifier(depth_path) if config.detector_backend == "depth-edge-object" else None,
            classifier_backend="depth-geometry-weak" if config.detector_backend == "depth-edge-object" else "unassigned",
        )

    if config.detector_backend == "sam3":
        detect_config = DetectShapesBackendConfig(
            backend=config.detector_backend,
            depth=str(depth_path),
            edge_map=None,
            detector_model=config.detector_model,
            detector_weights=config.detector_weights,
            clip_model_dir=None,
            device=config.device,
            primitive_source=config.primitive_source,
            confidence=config.detector_confidence,
            overlap_iou_threshold=config.detector_overlap_iou_threshold,
            rgbd_channel_weights=config.rgbd_channel_weights,
            text_prompt=config.text_prompt,
            box_threshold=config.box_threshold,
            text_threshold=config.text_threshold,
            sam3_repo_dir=config.sam3_repo_dir,
            sam3_model_dir=config.sam3_model_dir,
        )
        return sam3_runtime(detect_config, require_dir=lambda value, label: Path(value) if Path(value or "").is_dir() else require_file(value, label).parent)

    if config.detector_backend == "groundingdino-sam3":
        detect_config = DetectShapesBackendConfig(
            backend=config.detector_backend,
            depth=str(depth_path),
            edge_map=None,
            detector_model=config.detector_model,
            detector_weights=config.detector_weights,
            clip_model_dir=None,
            device=config.device,
            primitive_source=config.primitive_source,
            confidence=config.detector_confidence,
            overlap_iou_threshold=config.detector_overlap_iou_threshold,
            rgbd_channel_weights=config.rgbd_channel_weights,
            text_prompt=config.text_prompt,
            box_threshold=config.box_threshold,
            text_threshold=config.text_threshold,
            groundingdino_repo_dir=config.groundingdino_repo_dir,
            groundingdino_config=config.groundingdino_config,
            groundingdino_checkpoint=config.groundingdino_checkpoint,
            sam3_repo_dir=config.sam3_repo_dir,
            sam3_model_dir=config.sam3_model_dir,
        )
        return groundingdino_sam3_runtime(
            detect_config,
            require_file=require_file,
            require_dir=lambda value, label: Path(value) if Path(value or "").is_dir() else require_file(value, label).parent,
        )

    if config.detector_backend == "rgbd-yolo":
        return legacy_rgbd_yolo_runtime(
            detector_weights=config.detector_weights,
            depth_path=depth_path,
            confidence=config.detector_confidence,
            requested_device=config.requested_device,
            device=config.device,
            overlap_iou_threshold=config.detector_overlap_iou_threshold,
            primitive_source=config.primitive_source,
            rgbd_channel_weights=config.rgbd_channel_weights,
            require_file=require_file,
        )

    raise ValueError(f"Unsupported reconstruct detector backend: {config.detector_backend}")


def depth_edge_runtime(
    *,
    segmenter: DepthEdgeSegmenter,
    depth_path: Path,
    edge_map_path: Path | None,
    edge_backend: str | None,
    classifier: object | None = None,
    classifier_backend: str = "unassigned",
) -> DetectionRuntime:
    model_info = {
        "detector_backend": segmenter.backend_info.name,
        "detector_architecture": segmenter.backend_info.architecture,
        "detector_input_channels": list(segmenter.backend_info.input_channels),
        "detector_backend_info": segmenter.backend_info.to_dict(),
        "depth_path": str(depth_path),
        "edge_map_path": str(edge_map_path) if edge_map_path else None,
        "edge_backend": edge_backend,
        "classifier_backend": classifier_backend,
        "primitive_label_policy": segmenter.backend_info.primitive_label_policy,
        "legacy_yolo": segmenter.backend_info.legacy,
    }
    return DetectionRuntime(
        segmenter=segmenter,
        classifier=classifier or UnassignedPrimitiveClassifier(),
        model_info=model_info,
    )


def learned_depth_edge_runtime(
    *,
    segmenter: LearnedDepthEdgeSegmenter,
    depth_path: Path,
    edge_map_path: Path | None,
    edge_backend: str | None,
) -> DetectionRuntime:
    model_info = {
        "detector_backend": segmenter.backend_info.name,
        "detector_architecture": segmenter.backend_info.architecture,
        "detector_input_channels": list(segmenter.backend_info.input_channels),
        "detector_backend_info": segmenter.backend_info.to_dict(),
        "detector_model": str(segmenter.model_path),
        "detector_device": segmenter.device,
        "detector_checkpoint_metadata": segmenter.checkpoint_metadata,
        "depth_path": str(depth_path),
        "edge_map_path": str(edge_map_path) if edge_map_path else None,
        "edge_backend": edge_backend,
        "classifier_backend": "unassigned",
        "primitive_label_policy": segmenter.backend_info.primitive_label_policy,
        "legacy_yolo": segmenter.backend_info.legacy,
    }
    return DetectionRuntime(
        segmenter=segmenter,
        classifier=UnassignedPrimitiveClassifier(),
        model_info=model_info,
    )


def open_vocabulary_runtime(segmenter: object) -> DetectionRuntime:
    backend_info = segmenter.backend_info
    model_info = {
        "detector_backend": backend_info.name,
        "detector_architecture": backend_info.architecture,
        "detector_input_channels": list(backend_info.input_channels),
        "detector_backend_info": backend_info.to_dict(),
        "classifier_backend": "unassigned",
        "primitive_label_policy": backend_info.primitive_label_policy,
        "legacy_yolo": backend_info.legacy,
        "text_prompt": getattr(segmenter, "text_prompt", None),
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
    return open_vocabulary_runtime(segmenter)


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
    return open_vocabulary_runtime(segmenter)


def legacy_rgb_yolo_runtime(
    config: DetectShapesBackendConfig,
    *,
    require_file: RequireFile,
    require_dir: RequireDir,
) -> DetectionRuntime:
    inference_device = resolve_torch_device(config.device)
    detector_weights = require_file(config.detector_weights, "--detector-weights")
    from Segmentation.yolo_segmenter import YoloSegmenter

    segmenter = YoloSegmenter(
        weights_path=detector_weights,
        confidence=config.confidence,
        device=inference_device,
        overlap_iou_threshold=config.overlap_iou_threshold,
        retina_masks=True,
    )
    classifier, classifier_backend, clip_model_dir_value = legacy_classifier(
        primitive_source=config.primitive_source,
        clip_model_dir=config.clip_model_dir,
        device=inference_device,
        require_dir=require_dir,
        allow_clip=True,
    )
    return DetectionRuntime(
        segmenter=segmenter,
        classifier=classifier,
        model_info={
            "detector_backend": "ultralytics-yolo-seg",
            "detector_backend_alias": config.backend,
            "detector_weights": str(detector_weights),
            "detector_requested_device": config.device,
            "detector_device": segmenter.device,
            "detector_confidence": float(config.confidence),
            "detector_overlap_iou_threshold": float(config.overlap_iou_threshold),
            "retina_masks": True,
            "classifier_backend": classifier_backend,
            "clip_model_dir": clip_model_dir_value,
            "legacy_yolo": True,
        },
    )


def legacy_rgbd_yolo_runtime(
    *,
    detector_weights: str | None,
    depth_path: Path,
    confidence: float,
    requested_device: str | None,
    device: str | None,
    overlap_iou_threshold: float,
    primitive_source: str,
    rgbd_channel_weights: str,
    require_file: RequireFile,
) -> DetectionRuntime:
    inference_device = resolve_torch_device(device)
    weights_path = require_file(detector_weights, "--detector-weights")
    from Segmentation.rgbd_yolo_segmenter import RgbdYoloSegmenter

    segmenter = RgbdYoloSegmenter(
        weights_path=weights_path,
        depth_path=depth_path,
        confidence=confidence,
        device=inference_device,
        overlap_iou_threshold=overlap_iou_threshold,
        retina_masks=True,
        channel_weights=rgbd_channel_weights,
    )
    classifier, classifier_backend, _ = legacy_classifier(
        primitive_source=primitive_source,
        clip_model_dir=None,
        device=inference_device,
        require_dir=None,
        allow_clip=False,
    )
    return DetectionRuntime(
        segmenter=segmenter,
        classifier=classifier,
        model_info={
            "detector_backend": "ultralytics-yolo26l-rgbd-seg",
            "detector_weights": str(weights_path),
            "detector_requested_device": requested_device,
            "detector_device": segmenter.device,
            "detector_confidence": float(confidence),
            "detector_overlap_iou_threshold": float(overlap_iou_threshold),
            "retina_masks": True,
            "detector_input_channels": 4,
            "depth_path": str(depth_path),
            "depth_convention": "white_close_black_far",
            "rgbd_channel_weights": rgbd_channel_weights,
            "classifier_backend": classifier_backend,
            "clip_model_dir": None,
            "legacy_yolo": True,
        },
    )


def legacy_classifier(
    *,
    primitive_source: str,
    clip_model_dir: str | None,
    device: str | None,
    require_dir: RequireDir | None,
    allow_clip: bool,
) -> tuple[object, str, str | None]:
    if primitive_source == "none":
        return UnassignedPrimitiveClassifier(), "unassigned", None
    if primitive_source == "detector-label":
        from ShapeDetection.detector_label_classifier import DetectorLabelPrimitiveClassifier

        return DetectorLabelPrimitiveClassifier(), "detector-label-legacy", None
    if primitive_source == "clip" and allow_clip and require_dir is not None:
        clip_dir = require_dir(clip_model_dir, "--clip-model-dir")
        from ShapeDetection.clip_classifier import ClipPrimitiveClassifier

        return ClipPrimitiveClassifier(model_dir=clip_dir, device=device), "transformers-clip", str(clip_dir)
    if primitive_source == "clip":
        raise ValueError("This detector backend supports --primitive-source none or detector-label.")
    raise ValueError(f"Unsupported primitive source: {primitive_source}")
