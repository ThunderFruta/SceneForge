from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from Segmentation.primitive_3d import FEATURE_NAMES
from Tools.Dataset.instance_manifest import write_instance_manifest
from Tools.Dataset.rgbd_curriculum import DATASET_SPLITS, split_path
from Tools.Training.instance_detector import write_eval_scaffold, write_training_scaffold


def make_split(root: Path, split: str) -> None:
    split_path(root, split, "images").mkdir(parents=True)
    split_path(root, split, "depth").mkdir(parents=True)
    split_path(root, split, "masks").mkdir(parents=True)
    Image.new("RGB", (16, 16), "black").save(split_path(root, split, "images") / "sample_0000.png")
    Image.new("L", (16, 16), 180).save(split_path(root, split, "depth") / "sample_0000.png")
    mask = Image.new("L", (16, 16), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle((4, 4, 12, 12), fill=255)
    mask.save(split_path(root, split, "masks") / "sample_0000_00_box.png")


def make_manifest(root: Path) -> Path:
    for split in DATASET_SPLITS:
        make_split(root, split)
    return write_instance_manifest(root)


def make_config(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "test_primitive_3d",
                "architecture": "primitive_3d_point_embedding_v1",
                "input_contract": "rgb_depth_camera_to_visible_point_cloud",
                "input_channels": list(FEATURE_NAMES),
                "output_contract": "class_agnostic_instance_masks",
                "primitive_label_policy": "geometry_fitting_downstream",
                "hidden_dim": 16,
                "embedding_dim": 4,
                "max_points": 128,
                "min_cluster_points": 4,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_training_writes_primitive_3d_checkpoint_and_summaries(tmp_path: Path) -> None:
    manifest_path = make_manifest(tmp_path / "dataset")
    config_path = make_config(tmp_path / "config.json")

    checkpoint_path = write_training_scaffold(
        manifest_path=manifest_path,
        config_path=config_path,
        output_dir=tmp_path / "train",
        epochs=1,
        batch=4,
        device="cpu",
    )

    summary_path = tmp_path / "train" / "training_summary.json"
    eval_path = tmp_path / "train" / "eval_summary.json"
    assert checkpoint_path.name == "primitive_3d_segmenter.pt"
    assert checkpoint_path.is_file()
    assert eval_path.is_file()
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert data["status"] == "trained"
    assert data["trained"] is True
    assert data["config_path"] == str(config_path)
    assert data["architecture"] == "primitive_3d_point_embedding_v1"
    assert data["input_channels"] == list(FEATURE_NAMES)
    assert data["output_contract"] == "class_agnostic_instance_masks"
    assert data["split_counts"]["train"] == {"objects": 1, "samples": 1}
    assert (tmp_path / "dataset" / "train" / "point_cloud" / "sample_0000.npz").is_file()


def test_eval_writes_primitive_3d_summary(tmp_path: Path) -> None:
    manifest_path = make_manifest(tmp_path / "dataset")
    config_path = make_config(tmp_path / "config.json")
    model_path = write_training_scaffold(
        manifest_path=manifest_path,
        config_path=config_path,
        output_dir=tmp_path / "train",
        epochs=1,
        batch=4,
        device="cpu",
    )

    summary_path = write_eval_scaffold(
        manifest_path=manifest_path,
        model_path=model_path,
        config_path=config_path,
        output_dir=tmp_path / "eval",
        split="test",
        device="cpu",
    )

    data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert data["status"] == "evaluated"
    assert data["evaluated"] is True
    assert data["model_path"] == str(model_path)
    assert data["config_path"] == str(config_path)
    assert data["config_name"] == "test_primitive_3d"
    assert data["architecture"] == "primitive_3d_point_embedding_v1"
    assert data["sample_count"] == 1
    assert data["object_count"] == 1
    assert "object_recall" in data
