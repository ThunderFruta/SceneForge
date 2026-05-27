from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

from Tools.Dataset.rgbd_curriculum import DATASET_SPLITS, split_path


CLASS_NAMES = ("sphere", "box", "cylinder", "cone", "plane", "torus", "tube", "arch")
MASK_NAME_RE = re.compile(r"^(?P<stem>.+)_(?P<object_index>\d+)_(?P<class_name>[a-z_]+)\.png$")


def _get_cv2():
    try:
        import cv2 as cv2_module
    except ModuleNotFoundError:
        return None
    return cv2_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert rendered primitive instance masks into YOLO segmentation labels."
    )
    parser.add_argument("--dataset", default="Datasets/PrimitiveShapes")
    parser.add_argument("--mask-subdir", default="masks")
    parser.add_argument("--min-area", type=float, default=64.0)
    parser.add_argument(
        "--min-object-area",
        type=float,
        default=None,
        help="Minimum total visible mask area for an object. Defaults to --min-area.",
    )
    parser.add_argument("--epsilon-ratio", type=float, default=0.0015)
    return parser.parse_args()


def grouped_masks(mask_dir: Path) -> dict[str, list[tuple[str, Path]]]:
    grouped: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for path in sorted(mask_dir.glob("*.png")):
        match = MASK_NAME_RE.match(path.name)
        if not match:
            continue
        class_name = match.group("class_name")
        if class_name not in CLASS_NAMES:
            continue
        grouped[match.group("stem")].append((class_name, path))
    return grouped


def mask_to_segments(
    mask_path: Path,
    class_name: str,
    min_area: float,
    min_object_area: float,
    epsilon_ratio: float,
) -> list[str]:
    cv2 = _get_cv2()
    if cv2 is None:
        raise ModuleNotFoundError("cv2 is required for mask_to_segments()")

    image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read mask image: {mask_path}")

    height, width = image.shape[:2]
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = sum(float(cv2.contourArea(contour)) for contour in contours)
    if total_area < min_object_area:
        return []

    lines: list[str] = []
    class_id = CLASS_NAMES.index(class_name)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        epsilon = max(1.0, epsilon_ratio * cv2.arcLength(contour, True))
        polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(polygon) < 3:
            continue

        values: list[str] = []
        for x, y in polygon:
            values.append(f"{max(0.0, min(1.0, float(x) / width)):.6f}")
            values.append(f"{max(0.0, min(1.0, float(y) / height)):.6f}")
        lines.append(f"{class_id} {' '.join(values)}")
    return lines


def convert_split(
    dataset_root: Path,
    split: str,
    mask_subdir: str,
    min_area: float,
    min_object_area: float,
    epsilon_ratio: float,
) -> None:
    image_dir = split_path(dataset_root, split, "images")
    label_dir = split_path(dataset_root, split, "labels")
    mask_dir = split_path(dataset_root, split, mask_subdir)
    if not image_dir.is_dir():
        raise ValueError(f"Image split directory does not exist: {image_dir}")
    if not mask_dir.is_dir():
        raise ValueError(f"Mask split directory does not exist: {mask_dir}")

    label_dir.mkdir(parents=True, exist_ok=True)
    masks_by_stem = grouped_masks(mask_dir)
    for image_path in sorted(image_dir.glob("*.png")):
        lines: list[str] = []
        for class_name, mask_path in masks_by_stem.get(image_path.stem, []):
            lines.extend(mask_to_segments(mask_path, class_name, min_area, min_object_area, epsilon_ratio))
        label_path = label_dir / f"{image_path.stem}.txt"
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def convert_dataset(
    dataset_root: Path,
    mask_subdir: str,
    min_area: float,
    min_object_area: float | None,
    epsilon_ratio: float,
) -> None:
    object_area = min_area if min_object_area is None else min_object_area
    for split in DATASET_SPLITS:
        convert_split(dataset_root, split, mask_subdir, min_area, object_area, epsilon_ratio)


if __name__ == "__main__":
    args = parse_args()
    convert_dataset(
        dataset_root=Path(args.dataset),
        mask_subdir=args.mask_subdir,
        min_area=args.min_area,
        min_object_area=args.min_object_area,
        epsilon_ratio=args.epsilon_ratio,
    )
