from __future__ import annotations

from pathlib import Path

from PIL import Image

from Segmentation.primitive_3d import Primitive3DConfig, Primitive3DSegNet, default_checkpoint_metadata
from Segmentation.factory import DetectShapesBackendConfig, build_detect_shapes_runtime


def require_file(value: str | Path | None, label: str) -> Path:
    if value is None:
        raise ValueError(f"{label} is required.")
    path = Path(value)
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    return path


def require_dir(value: str | Path | None, label: str) -> Path:
    raise AssertionError(f"{label} should not be required for depth-edge runtime")


def write_tiny_checkpoint(path: Path) -> None:
    import torch

    config = Primitive3DConfig(hidden_dim=8, embedding_dim=4, max_points=64, min_cluster_points=4)
    model = Primitive3DSegNet(input_dim=config.input_dim, hidden_dim=config.hidden_dim, embedding_dim=config.embedding_dim)
    torch.nn.init.zeros_(model.objectness_head.weight)
    torch.nn.init.constant_(model.objectness_head.bias, -10.0)
    torch.save(
        {
            "schema_version": 1,
            "metadata": default_checkpoint_metadata(config),
            "model_state": model.state_dict(),
        },
        path,
    )


def test_factory_builds_depth_edge_runtime_without_yolo_weights(tmp_path: Path) -> None:
    depth_path = tmp_path / "depth.png"
    Image.new("L", (16, 16), 180).save(depth_path)

    runtime = build_detect_shapes_runtime(
        DetectShapesBackendConfig(
            backend="depth-edge",
            depth=str(depth_path),
            edge_map=None,
            detector_model=None,
            detector_weights=None,
            clip_model_dir=None,
            device="auto",
            primitive_source="none",
            confidence=0.25,
            overlap_iou_threshold=0.7,
            rgbd_channel_weights="0.20,0.20,0.20,0.40",
        ),
        require_file=require_file,
        require_dir=require_dir,
    )

    assert runtime.model_info["detector_backend"] == "depth-edge-instance-scaffold"
    assert runtime.model_info["detector_backend_info"]["legacy"] is False
    assert runtime.model_info["detector_backend_info"]["output_contract"] == "instance_masks_only"
    assert runtime.model_info["primitive_label_policy"] == "geometry_fitting_downstream"


def test_factory_builds_depth_edge_object_runtime(tmp_path: Path) -> None:
    depth_path = tmp_path / "depth.png"
    Image.new("L", (16, 16), 180).save(depth_path)

    runtime = build_detect_shapes_runtime(
        DetectShapesBackendConfig(
            backend="depth-edge-object",
            depth=str(depth_path),
            edge_map=None,
            detector_model=None,
            detector_weights=None,
            clip_model_dir=None,
            device="auto",
            primitive_source="none",
            confidence=0.25,
            overlap_iou_threshold=0.7,
            rgbd_channel_weights="0.20,0.20,0.20,0.40",
        ),
        require_file=require_file,
        require_dir=require_dir,
    )

    assert runtime.model_info["detector_backend"] == "depth-edge-object-detector"
    assert runtime.model_info["detector_architecture"] == "rgb_depth_edge_object_detector"
    assert runtime.model_info["detector_input_channels"] == ["rgb", "depth", "edge"]
    assert runtime.model_info["classifier_backend"] == "depth-geometry-weak"


def test_factory_builds_primitive_3d_detector_model(tmp_path: Path) -> None:
    depth_path = tmp_path / "depth.png"
    model_path = tmp_path / "model.pt"
    Image.new("L", (16, 16), 180).save(depth_path)
    write_tiny_checkpoint(model_path)

    runtime = build_detect_shapes_runtime(
        DetectShapesBackendConfig(
            backend="depth-edge",
            depth=str(depth_path),
            edge_map=None,
            detector_model=str(model_path),
            detector_weights=None,
            clip_model_dir=None,
            device="auto",
            primitive_source="none",
            confidence=0.25,
            overlap_iou_threshold=0.7,
            rgbd_channel_weights="0.20,0.20,0.20,0.40",
        ),
        require_file=require_file,
        require_dir=require_dir,
    )

    assert runtime.model_info["detector_backend"] == "primitive-3d-segmenter"
    assert runtime.model_info["detector_model"] == str(model_path)
    assert runtime.model_info["detector_backend_info"]["output_contract"] == "class_agnostic_instance_masks"
    assert runtime.model_info["classifier_backend"] == "unassigned"
    assert runtime.segmenter.detect(Image.new("RGB", (16, 16), "black")) == []


def test_factory_builds_sam3_runtime_without_importing_external_repo(tmp_path: Path) -> None:
    import sys

    repo_dir = tmp_path / "sam3_repo"
    model_dir = tmp_path / "sam3_model"
    repo_dir.mkdir()
    model_dir.mkdir()

    runtime = build_detect_shapes_runtime(
        DetectShapesBackendConfig(
            backend="sam3",
            depth=None,
            edge_map=None,
            detector_model=None,
            detector_weights=None,
            clip_model_dir=None,
            device="cpu",
            primitive_source="none",
            confidence=0.25,
            overlap_iou_threshold=0.7,
            rgbd_channel_weights="0.20,0.20,0.20,0.40",
            sam3_repo_dir=str(repo_dir),
            sam3_model_dir=str(model_dir),
            text_prompt="chair . table .",
        ),
        require_file=require_file,
        require_dir=lambda value, label: Path(value),
    )

    assert runtime.model_info["detector_backend"] == "sam3-open-vocabulary"
    assert runtime.model_info["detector_backend_info"]["proposal_only"] is True
    assert runtime.model_info["primitive_label_policy"] == "geometry_fitting_downstream"
    assert "sam3.model_builder" not in sys.modules


def test_factory_builds_groundingdino_sam3_runtime_without_importing_external_repos(tmp_path: Path) -> None:
    import sys

    gdino_repo = tmp_path / "GroundingDINO"
    gdino_repo.mkdir()
    gdino_config = tmp_path / "GroundingDINO_SwinT_OGC.py"
    gdino_checkpoint = tmp_path / "groundingdino_swint_ogc.pth"
    gdino_config.write_text("# test config", encoding="utf-8")
    gdino_checkpoint.write_bytes(b"checkpoint")
    sam3_repo = tmp_path / "sam3_repo"
    sam3_model = tmp_path / "sam3_model"
    sam3_repo.mkdir()
    sam3_model.mkdir()

    runtime = build_detect_shapes_runtime(
        DetectShapesBackendConfig(
            backend="groundingdino-sam3",
            depth=None,
            edge_map=None,
            detector_model=None,
            detector_weights=None,
            clip_model_dir=None,
            device="cpu",
            primitive_source="none",
            confidence=0.25,
            overlap_iou_threshold=0.7,
            rgbd_channel_weights="0.20,0.20,0.20,0.40",
            groundingdino_repo_dir=str(gdino_repo),
            groundingdino_config=str(gdino_config),
            groundingdino_checkpoint=str(gdino_checkpoint),
            sam3_repo_dir=str(sam3_repo),
            sam3_model_dir=str(sam3_model),
            text_prompt="chair . table .",
        ),
        require_file=require_file,
        require_dir=lambda value, label: Path(value),
    )

    assert runtime.model_info["detector_backend"] == "groundingdino-sam3-open-vocabulary"
    assert runtime.model_info["detector_backend_info"]["output_contract"] == "open_vocab_box_guided_instance_masks"
    assert runtime.model_info["legacy_yolo"] is False
    assert "groundingdino.util.inference" not in sys.modules
    assert "sam3.model_builder" not in sys.modules
