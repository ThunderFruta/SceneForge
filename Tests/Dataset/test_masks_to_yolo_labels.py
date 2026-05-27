from pathlib import Path

from PIL import Image, ImageDraw

from Tools.Dataset.rgbd_curriculum import DATASET_SPLITS, split_path
from Tools.Dataset.masks_to_yolo_labels import convert_dataset


def make_split(root: Path, split: str) -> None:
    split_path(root, split, "images").mkdir(parents=True)
    split_path(root, split, "masks").mkdir(parents=True)
    Image.new("RGB", (100, 100), (80, 80, 80)).save(split_path(root, split, "images") / "primitive_0000.png")

    mask = Image.new("L", (100, 100), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle((20, 30, 70, 80), fill=255)
    mask.save(split_path(root, split, "masks") / "primitive_0000_00_box.png")


def test_convert_dataset_writes_yolo_segmentation_labels(tmp_path: Path) -> None:
    make_split(tmp_path, "train")
    make_split(tmp_path, "val")
    make_split(tmp_path, "test")

    convert_dataset(tmp_path, mask_subdir="masks", min_area=4.0, min_object_area=None, epsilon_ratio=0.01)

    label_text = (split_path(tmp_path, "train", "labels") / "primitive_0000.txt").read_text(encoding="utf-8")
    parts = label_text.split()
    assert parts[0] == "1"
    assert len(parts[1:]) >= 6
    for split in DATASET_SPLITS:
        assert (split_path(tmp_path, split, "labels") / "primitive_0000.txt").is_file()


def test_convert_dataset_skips_tiny_object_masks(tmp_path: Path) -> None:
    make_split(tmp_path, "train")
    make_split(tmp_path, "val")
    make_split(tmp_path, "test")

    convert_dataset(tmp_path, mask_subdir="masks", min_area=1.0, min_object_area=10000.0, epsilon_ratio=0.01)

    label_text = (split_path(tmp_path, "train", "labels") / "primitive_0000.txt").read_text(encoding="utf-8")
    assert label_text == ""
