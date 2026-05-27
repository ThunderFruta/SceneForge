from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from Tools.Dataset.rgbd_curriculum import (
    BASE_CLASSES,
    DATASET_SPLITS,
    EXTENDED_CLASSES,
    compose_rgbd_image,
    split_name_for_index,
    split_path,
    stage_for_id,
    validate_rgbd_dataset,
    validate_rgbd_sample,
    write_rgbd_data_yaml,
)


def test_stage_presets_have_expected_object_counts() -> None:
    assert stage_for_id(1).min_objects == 1
    assert stage_for_id(1).max_objects == 1
    assert stage_for_id(2).min_objects == 3
    assert stage_for_id(2).max_objects == 4
    assert stage_for_id(3).min_objects == 5
    assert stage_for_id(3).max_objects == 7
    assert stage_for_id(5).allow_occlusion is True
    assert stage_for_id(4).allow_occlusion is True
    assert stage_for_id(4).max_objects == 7
    assert stage_for_id(7).classes == EXTENDED_CLASSES
    assert stage_for_id(8).name == "furnishing_combinations"
    assert stage_for_id(9).allow_occlusion is False
    assert stage_for_id(10).name == "low_poly_scene"
    assert stage_for_id(10).classes == EXTENDED_CLASSES
    assert stage_for_id(1).scale_min > 1.0
    assert stage_for_id(1).min_screen_area_ratio > stage_for_id(5).min_screen_area_ratio
    assert stage_for_id(2).max_screen_overlap_ratio < stage_for_id(4).max_screen_overlap_ratio
    assert stage_for_id(4).max_screen_overlap_ratio < stage_for_id(5).max_screen_overlap_ratio
    assert stage_for_id(3).max_screen_overlap_ratio < stage_for_id(5).max_screen_overlap_ratio


def test_stage_rejects_unknown_id() -> None:
    with pytest.raises(ValueError, match="Unknown curriculum stage"):
        stage_for_id(99)


def test_write_rgbd_data_yaml_declares_four_channels(tmp_path: Path) -> None:
    write_rgbd_data_yaml(tmp_path, BASE_CLASSES)

    text = (tmp_path / "data_rgbd.yaml").read_text(encoding="utf-8")
    assert "channels: 4" in text
    assert "train: train/images" in text
    assert "val: val/images" in text
    assert "test: test/images" in text
    assert "1: box" in text


def test_default_split_names_are_70_20_10() -> None:
    names = [split_name_for_index(index, 10) for index in range(10)]

    assert names.count("train") == 7
    assert names.count("val") == 2
    assert names.count("test") == 1
    assert names == ["train"] * 7 + ["val"] * 2 + ["test"]


def test_split_path_uses_split_first_layout(tmp_path: Path) -> None:
    assert split_path(tmp_path, "train", "images") == tmp_path / "train" / "images"
    assert split_path(tmp_path, "val", "images_rgb") == tmp_path / "val" / "rgb"
    assert split_path(tmp_path, "test", "images_rgbd") == tmp_path / "test" / "rgbd"
    assert split_path(tmp_path, "test", "depth") == tmp_path / "test" / "depth"
    assert split_path(tmp_path, "test", "labels") == tmp_path / "test" / "labels"
    assert split_path(tmp_path, "test", "custom_masks") == tmp_path / "test" / "custom_masks"


def test_validate_rgbd_sample_requires_alpha_to_match_depth(tmp_path: Path) -> None:
    rgbd_path = tmp_path / "sample.png"
    depth_path = tmp_path / "sample_depth.png"
    Image.new("RGBA", (2, 2), (10, 20, 30, 128)).save(rgbd_path)
    Image.new("L", (2, 2), 128).save(depth_path)

    assert validate_rgbd_sample(rgbd_path, depth_path) == (2, 2)

    Image.new("L", (2, 2), 64).save(depth_path)
    with pytest.raises(ValueError, match="alpha does not match"):
        validate_rgbd_sample(rgbd_path, depth_path)


def test_compose_rgbd_image_uses_depth_as_alpha(tmp_path: Path) -> None:
    rgb_path = tmp_path / "rgb.png"
    depth_path = tmp_path / "depth.png"
    output_path = tmp_path / "rgbd.png"
    Image.new("RGB", (1, 1), (10, 20, 30)).save(rgb_path)
    Image.new("L", (1, 1), 77).save(depth_path)

    compose_rgbd_image(rgb_path, depth_path, output_path)

    with Image.open(output_path) as image:
        assert image.mode == "RGBA"
        assert image.getpixel((0, 0)) == (10, 20, 30, 77)


def test_validate_rgbd_dataset_counts_labels(tmp_path: Path) -> None:
    for split in DATASET_SPLITS:
        split_path(tmp_path, split, "images_rgbd").mkdir(parents=True)
        split_path(tmp_path, split, "depth").mkdir(parents=True)
        split_path(tmp_path, split, "labels").mkdir(parents=True)
        Image.new("RGBA", (4, 4), (1, 2, 3, 200)).save(split_path(tmp_path, split, "images_rgbd") / "primitive_0000.png")
        Image.new("L", (4, 4), 200).save(split_path(tmp_path, split, "depth") / "primitive_0000.png")
        (split_path(tmp_path, split, "labels") / "primitive_0000.txt").write_text(
            "1 0.1 0.1 0.9 0.1 0.9 0.9\n",
            encoding="utf-8",
        )

    summary = validate_rgbd_dataset(tmp_path, BASE_CLASSES)

    assert summary["class_counts"]["box"] == 3
    assert set(summary["split_counts"]) == set(DATASET_SPLITS)
    assert summary["split_counts"]["test"]["images"] == 1
    assert summary["object_count_min"] == 1
    assert summary["object_count_max"] == 1
    assert summary["depth_min"] == 200
    assert summary["depth_max"] == 200
    assert (tmp_path / "validation_report.json").is_file()


def test_compose_rgbd_dataset_copies_rgbd_to_yolo_images_dir(tmp_path: Path) -> None:
    for split in DATASET_SPLITS:
        split_path(tmp_path, split, "images_rgb").mkdir(parents=True)
        split_path(tmp_path, split, "depth").mkdir(parents=True)
        Image.new("RGB", (2, 2), (10, 20, 30)).save(split_path(tmp_path, split, "images_rgb") / "primitive_0000.png")
        Image.new("L", (2, 2), 77).save(split_path(tmp_path, split, "depth") / "primitive_0000.png")

    from Tools.Dataset.rgbd_curriculum import compose_rgbd_dataset

    compose_rgbd_dataset(tmp_path)

    for path in (
        split_path(tmp_path, "train", "images_rgbd") / "primitive_0000.png",
        split_path(tmp_path, "train", "images") / "primitive_0000.png",
        split_path(tmp_path, "val", "images_rgbd") / "primitive_0000.png",
        split_path(tmp_path, "val", "images") / "primitive_0000.png",
        split_path(tmp_path, "test", "images_rgbd") / "primitive_0000.png",
        split_path(tmp_path, "test", "images") / "primitive_0000.png",
    ):
        with Image.open(path) as image:
            assert image.mode == "RGBA"
            assert image.getpixel((0, 0)) == (10, 20, 30, 77)
