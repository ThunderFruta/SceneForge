from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image


BASE_CLASSES = ("sphere", "box", "cylinder", "cone", "plane")
EXTENDED_CLASSES = (*BASE_CLASSES, "torus", "tube", "arch")
DATASET_SPLITS = ("train", "val", "test")
SPLIT_SUBDIRS = {
    "images": "images",
    "images_rgb": "rgb",
    "images_rgbd": "rgbd",
    "depth": "depth",
    "labels": "labels",
    "masks": "masks",
    "labeled_images": "labeled_images",
    "annotations": "annotations",
}


@dataclass(frozen=True)
class CurriculumStage:
    id: int
    name: str
    classes: tuple[str, ...]
    min_objects: int
    max_objects: int
    allow_occlusion: bool
    extended_geometry: bool
    scale_min: float
    scale_max: float
    min_screen_area_ratio: float
    max_screen_overlap_ratio: float
    description: str


CURRICULUM_STAGES: dict[int, CurriculumStage] = {
    1: CurriculumStage(
        id=1,
        name="single_clean",
        classes=BASE_CLASSES,
        min_objects=1,
        max_objects=1,
        allow_occlusion=False,
        extended_geometry=False,
        scale_min=1.25,
        scale_max=1.75,
        min_screen_area_ratio=0.035,
        max_screen_overlap_ratio=0.00,
        description="One clean base primitive per image.",
    ),
    2: CurriculumStage(
        id=2,
        name="handful_unobstructed",
        classes=BASE_CLASSES,
        min_objects=3,
        max_objects=4,
        allow_occlusion=False,
        extended_geometry=False,
        scale_min=1.15,
        scale_max=1.55,
        min_screen_area_ratio=0.025,
        max_screen_overlap_ratio=0.01,
        description="A handful of mostly separated base primitives.",
    ),
    3: CurriculumStage(
        id=3,
        name="many_unobstructed",
        classes=BASE_CLASSES,
        min_objects=5,
        max_objects=7,
        allow_occlusion=False,
        extended_geometry=False,
        scale_min=1.05,
        scale_max=1.40,
        min_screen_area_ratio=0.018,
        max_screen_overlap_ratio=0.03,
        description="Many base primitives with little intentional touching.",
    ),
    4: CurriculumStage(
        id=4,
        name="light_occlusion",
        classes=BASE_CLASSES,
        min_objects=4,
        max_objects=7,
        allow_occlusion=True,
        extended_geometry=False,
        scale_min=1.00,
        scale_max=1.35,
        min_screen_area_ratio=0.012,
        max_screen_overlap_ratio=0.14,
        description="Base primitives with mild touching, light overlap, and occasional border crops.",
    ),
    5: CurriculumStage(
        id=5,
        name="heavy_occlusion",
        classes=BASE_CLASSES,
        min_objects=8,
        max_objects=16,
        allow_occlusion=True,
        extended_geometry=False,
        scale_min=1.00,
        scale_max=1.45,
        min_screen_area_ratio=0.010,
        max_screen_overlap_ratio=0.70,
        description="Busy base primitive scenes with frequent occlusion.",
    ),
    6: CurriculumStage(
        id=6,
        name="extended_clean",
        classes=EXTENDED_CLASSES,
        min_objects=1,
        max_objects=5,
        allow_occlusion=False,
        extended_geometry=True,
        scale_min=1.15,
        scale_max=1.55,
        min_screen_area_ratio=0.022,
        max_screen_overlap_ratio=0.03,
        description="Clean base plus torus, tube, and arch geometry.",
    ),
    7: CurriculumStage(
        id=7,
        name="extended_occluded",
        classes=EXTENDED_CLASSES,
        min_objects=4,
        max_objects=12,
        allow_occlusion=True,
        extended_geometry=True,
        scale_min=1.05,
        scale_max=1.45,
        min_screen_area_ratio=0.010,
        max_screen_overlap_ratio=0.70,
        description="Occluded base and extended passthrough geometry.",
    ),
    8: CurriculumStage(
        id=8,
        name="furnishing_combinations",
        classes=EXTENDED_CLASSES,
        min_objects=8,
        max_objects=18,
        allow_occlusion=True,
        extended_geometry=True,
        scale_min=0.90,
        scale_max=1.55,
        min_screen_area_ratio=0.008,
        max_screen_overlap_ratio=0.60,
        description="Indoor/outdoor furnishing-like primitive combinations.",
    ),
    9: CurriculumStage(
        id=9,
        name="minimal_occlusion_scene",
        classes=EXTENDED_CLASSES,
        min_objects=12,
        max_objects=24,
        allow_occlusion=False,
        extended_geometry=True,
        scale_min=0.80,
        scale_max=1.35,
        min_screen_area_ratio=0.006,
        max_screen_overlap_ratio=0.06,
        description="Larger mostly visible scenes with minimal occlusion.",
    ),
    10: CurriculumStage(
        id=10,
        name="low_poly_scene",
        classes=EXTENDED_CLASSES,
        min_objects=18,
        max_objects=36,
        allow_occlusion=True,
        extended_geometry=True,
        scale_min=0.70,
        scale_max=1.30,
        min_screen_area_ratio=0.004,
        max_screen_overlap_ratio=0.80,
        description="Dense low-poly primitive scenes for broad scene layout practice.",
    ),
}


def stage_for_id(stage_id: int) -> CurriculumStage:
    try:
        return CURRICULUM_STAGES[stage_id]
    except KeyError as exc:
        valid = ", ".join(str(value) for value in sorted(CURRICULUM_STAGES))
        raise ValueError(f"Unknown curriculum stage {stage_id}. Valid stages: {valid}.") from exc


def stage_root(dataset_root: str | Path, stage: CurriculumStage) -> Path:
    return Path(dataset_root) / f"stage{stage.id}_{stage.name}"


def split_path(dataset_root: str | Path, split: str, artifact: str) -> Path:
    subdir = SPLIT_SUBDIRS.get(artifact, artifact)
    if split not in DATASET_SPLITS:
        valid = ", ".join(DATASET_SPLITS)
        raise ValueError(f"Unknown dataset split {split!r}. Valid splits: {valid}.")
    return Path(dataset_root) / split / subdir


def split_name_for_index(index: int, count: int, train_split: float = 0.70, val_split: float = 0.20) -> str:
    train_count = int(count * train_split)
    val_count = int(count * val_split)
    if index < train_count:
        return "train"
    if index < train_count + val_count:
        return "val"
    return "test"


def write_rgbd_data_yaml(root: str | Path, classes: tuple[str, ...]) -> None:
    dataset_root = Path(root)
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(classes))
    (dataset_root / "data_rgbd.yaml").write_text(
        "\n".join(
            [
                f"path: {dataset_root.resolve()}",
                "train: train/images",
                "val: val/images",
                "test: test/images",
                "channels: 4",
                f"nc: {len(classes)}",
                "names:",
                names,
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_stage_manifest(root: str | Path, stage: CurriculumStage) -> None:
    dataset_root = Path(root)
    (dataset_root / "stage_manifest.json").write_text(
        json.dumps(asdict(stage), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_rgbd_sample(rgbd_path: Path, depth_path: Path) -> tuple[int, int]:
    with Image.open(rgbd_path) as rgbd:
        if rgbd.mode != "RGBA":
            raise ValueError(f"RGBD image must be RGBA: {rgbd_path}")
        size = rgbd.size
        alpha = rgbd.getchannel("A")
    with Image.open(depth_path) as depth:
        depth_l = depth.convert("L")
        if depth_l.size != size:
            raise ValueError(f"Depth size does not match RGBD size for {rgbd_path.name}")
        if alpha.tobytes() != depth_l.tobytes():
            raise ValueError(f"RGBD alpha does not match depth image for {rgbd_path.name}")
    return size


def compose_rgbd_image(rgb_path: str | Path, depth_path: str | Path, output_path: str | Path) -> None:
    with Image.open(rgb_path) as rgb_image:
        rgb = rgb_image.convert("RGB")
    with Image.open(depth_path) as depth_image:
        depth = depth_image.convert("L")
    if rgb.size != depth.size:
        raise ValueError(f"RGB and depth sizes do not match: {rgb_path}, {depth_path}")
    rgba = rgb.convert("RGBA")
    rgba.putalpha(depth)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    rgba.save(output_path)


def compose_rgbd_dataset(root: str | Path) -> None:
    dataset_root = Path(root)
    for split in DATASET_SPLITS:
        rgb_dir = split_path(dataset_root, split, "images_rgb")
        depth_dir = split_path(dataset_root, split, "depth")
        rgbd_dir = split_path(dataset_root, split, "images_rgbd")
        image_dir = split_path(dataset_root, split, "images")
        if not rgb_dir.is_dir():
            continue
        rgbd_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)
        for rgb_path in sorted(rgb_dir.glob("*.png")):
            rgbd_path = rgbd_dir / rgb_path.name
            compose_rgbd_image(rgb_path, depth_dir / rgb_path.name, rgbd_path)
            shutil.copyfile(rgbd_path, image_dir / rgb_path.name)


def validate_rgbd_dataset(root: str | Path, classes: tuple[str, ...] | None = None) -> dict:
    dataset_root = Path(root)
    class_names = classes or BASE_CLASSES
    class_counts = {name: 0 for name in class_names}
    split_counts: dict[str, dict[str, int]] = {}
    object_counts: list[int] = []
    depth_ranges: list[tuple[int, int]] = []

    for split in DATASET_SPLITS:
        rgbd_dir = split_path(dataset_root, split, "images_rgbd")
        depth_dir = split_path(dataset_root, split, "depth")
        label_dir = split_path(dataset_root, split, "labels")
        split_counts[split] = {"images": 0, "labels": 0, "objects": 0}
        if not rgbd_dir.exists():
            continue
        for rgbd_path in sorted(rgbd_dir.glob("*.png")):
            depth_path = depth_dir / rgbd_path.name
            validate_rgbd_sample(rgbd_path, depth_path)
            with Image.open(depth_path) as depth_image:
                values = depth_image.convert("L").tobytes()
                depth_ranges.append((min(values), max(values)))

            label_path = label_dir / f"{rgbd_path.stem}.txt"
            lines = []
            if label_path.exists():
                lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            split_counts[split]["images"] += 1
            split_counts[split]["labels"] += int(label_path.exists())
            split_counts[split]["objects"] += len(lines)
            object_counts.append(len(lines))
            for line in lines:
                class_id = int(line.split()[0])
                if 0 <= class_id < len(class_names):
                    class_counts[class_names[class_id]] += 1

    summary = {
        "root": str(dataset_root),
        "classes": list(class_names),
        "class_counts": class_counts,
        "split_counts": split_counts,
        "object_count_min": min(object_counts) if object_counts else 0,
        "object_count_max": max(object_counts) if object_counts else 0,
        "depth_min": min((item[0] for item in depth_ranges), default=0),
        "depth_max": max((item[1] for item in depth_ranges), default=0),
    }
    (dataset_root / "validation_report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
