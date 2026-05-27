from pathlib import Path

from PIL import Image

from Tools.Dataset.render_label_previews import render_dataset
from Tools.Dataset.rgbd_curriculum import DATASET_SPLITS, split_path


def make_split(root: Path, split: str) -> None:
    split_path(root, split, "images").mkdir(parents=True)
    split_path(root, split, "labels").mkdir(parents=True)
    Image.new("RGBA", (100, 100), (20, 30, 40, 128)).save(split_path(root, split, "images") / "primitive_0000.png")
    (split_path(root, split, "labels") / "primitive_0000.txt").write_text(
        "1 0.2 0.3 0.7 0.3 0.7 0.8 0.2 0.8\n",
        encoding="utf-8",
    )


def test_render_dataset_writes_annotation_previews_for_all_splits(tmp_path: Path) -> None:
    for split in DATASET_SPLITS:
        make_split(tmp_path, split)

    render_dataset(tmp_path, "annotations")

    for split in DATASET_SPLITS:
        output_path = split_path(tmp_path, split, "annotations") / "primitive_0000.png"
        assert output_path.is_file()
        with Image.open(output_path) as image:
            assert image.mode == "RGB"
            assert image.size == (100, 100)
            assert image.getpixel((20, 30)) != (20, 30, 40)
