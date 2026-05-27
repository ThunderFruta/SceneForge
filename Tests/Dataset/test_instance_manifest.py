from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from Segmentation.primitive_3d import FEATURE_NAMES
from Tools.Dataset.instance_manifest import write_instance_manifest
from Tools.Dataset.rgbd_curriculum import DATASET_SPLITS, split_path


def make_split(root: Path, split: str) -> None:
    split_path(root, split, "images").mkdir(parents=True)
    split_path(root, split, "depth").mkdir(parents=True)
    split_path(root, split, "masks").mkdir(parents=True)
    Image.new("RGB", (32, 32), (80, 80, 80)).save(split_path(root, split, "images") / "primitive_0000.png")
    Image.new("L", (32, 32), 180).save(split_path(root, split, "depth") / "primitive_0000.png")

    mask = Image.new("L", (32, 32), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle((8, 8, 24, 24), fill=255)
    mask.save(split_path(root, split, "masks") / "primitive_0000_00_box.png")


def test_write_instance_manifest_describes_primitive_3d_mask_contract(tmp_path: Path) -> None:
    for split in DATASET_SPLITS:
        make_split(tmp_path, split)

    output_path = write_instance_manifest(tmp_path)

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["detector_training_contract"]["input_contract"] == "rgb_depth_camera_to_visible_point_cloud"
    assert data["detector_training_contract"]["input_channels"] == list(FEATURE_NAMES)
    assert data["detector_training_contract"]["output_contract"] == "class_agnostic_instance_masks"
    assert data["detector_training_contract"]["legacy_yolo_labels"] == "compatibility_only"
    train_sample = data["splits"]["train"]["samples"][0]
    assert train_sample["rgb"] == "train/images/primitive_0000.png"
    assert train_sample["depth"] == "train/depth/primitive_0000.png"
    assert train_sample["camera"] is None
    assert train_sample["point_cloud"] == "train/point_cloud/primitive_0000.npz"
    assert train_sample["objects"][0] == {
        "class_name": "box",
        "object_index": 0,
        "point_label": 1,
        "visible_mask": "train/masks/primitive_0000_00_box.png",
    }
