from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from Tools.Dataset.rgbd_curriculum import DATASET_SPLITS, split_path


CLASS_NAMES = ("sphere", "box", "cylinder", "cone", "plane", "torus", "tube", "arch")
CLASS_COLORS = {
    "sphere": (42, 157, 143),
    "box": (244, 162, 97),
    "cylinder": (38, 70, 83),
    "cone": (230, 57, 70),
    "plane": (69, 123, 157),
    "torus": (131, 56, 236),
    "tube": (255, 183, 3),
    "arch": (142, 202, 230),
}
MASK_NAME_RE = re.compile(r"^(?P<stem>.+)_(?P<object_index>\d+)_(?P<class_name>[a-z_]+)\.png$")


def _get_cv2():
    try:
        import cv2 as cv2_module
    except ModuleNotFoundError:
        return None
    return cv2_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render labeled preview images from a YOLO segmentation dataset.")
    parser.add_argument("--dataset", default="Datasets/PrimitiveShapes")
    parser.add_argument("--output-subdir", default="annotations")
    return parser.parse_args()


def parse_label_file(path: Path, width: int, height: int) -> list[tuple[str, list[tuple[float, float]]]]:
    items: list[tuple[str, list[tuple[float, float]]]] = []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return items

    for line in text.splitlines():
        parts = line.split()
        class_id = int(parts[0])
        values = [float(value) for value in parts[1:]]
        points = [
            (values[index] * width, values[index + 1] * height)
            for index in range(0, len(values), 2)
        ]
        items.append((CLASS_NAMES[class_id], points))
    return items


def grouped_masks(mask_dir: Path) -> dict[str, list[tuple[int, str, Path]]]:
    grouped: dict[str, list[tuple[int, str, Path]]] = defaultdict(list)
    if not mask_dir.is_dir():
        return grouped
    for path in sorted(mask_dir.glob("*.png")):
        match = MASK_NAME_RE.match(path.name)
        if not match:
            continue
        class_name = match.group("class_name")
        if class_name not in CLASS_NAMES:
            continue
        grouped[match.group("stem")].append((int(match.group("object_index")), class_name, path))
    for masks in grouped.values():
        masks.sort(key=lambda item: item[0])
    return grouped


def mask_contours(mask_path: Path) -> list[list[tuple[int, int]]]:
    cv2 = _get_cv2()
    if cv2 is None:
        with Image.open(mask_path) as mask_image:
            alpha = mask_image.convert("L")
            bbox = alpha.getbbox()
            if bbox is None:
                return []
            left, top, right, bottom = bbox
            right = max(left + 1, right)
            bottom = max(top + 1, bottom)
            return [[(left, top), (right, top), (right, bottom), (left, bottom)]]

    image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return []
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points_by_contour: list[list[tuple[int, int]]] = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < 4:
            continue
        epsilon = max(0.5, 0.0015 * cv2.arcLength(contour, True))
        polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(polygon) >= 3:
            points_by_contour.append([(int(x), int(y)) for x, y in polygon])
    return points_by_contour


def draw_label(
    draw: ImageDraw.ImageDraw,
    label: str,
    point: tuple[float, float],
    font: ImageFont.ImageFont,
    color: tuple[int, int, int],
) -> None:
    x = max(0, int(point[0]))
    y = max(0, int(point[1]) - 18)
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    draw.rectangle((x, y, x + text_width + 8, y + text_height + 6), fill=(*color, 230))
    draw.text((x + 4, y + 3), label, fill=(255, 255, 255, 255), font=font)


def polygon_bbox(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def render_preview(image_path: Path, label_path: Path, output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    output = image.copy()
    draw = ImageDraw.Draw(output, "RGBA")
    font = ImageFont.load_default()

    for object_index, (label, points) in enumerate(parse_label_file(label_path, image.width, image.height), start=1):
        if len(points) < 3:
            continue
        color = CLASS_COLORS[label]
        draw.polygon(points, fill=(*color, 56), outline=(*color, 255))
        draw.line(points + [points[0]], fill=(*color, 255), width=3)
        left, top, right, bottom = polygon_bbox(points)
        draw.rectangle((left, top, right, bottom), outline=(*color, 255), width=3)
        draw_label(draw, f"{object_index:02d} {label}", (left, top), font, color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_path)


def render_mask_preview(
    image_path: Path,
    masks: list[tuple[int, str, Path]],
    output_path: Path,
) -> None:
    image = Image.open(image_path).convert("RGB")
    output = image.convert("RGBA")
    draw = ImageDraw.Draw(output, "RGBA")
    font = ImageFont.load_default()

    for object_index, (_mask_index, label, mask_path) in enumerate(masks, start=1):
        color = CLASS_COLORS[label]
        with Image.open(mask_path) as mask_image:
            mask = mask_image.convert("L")
            bbox = mask.getbbox()
            if bbox is None:
                continue
            overlay = Image.new("RGBA", image.size, (*color, 72))
            output.alpha_composite(Image.composite(overlay, Image.new("RGBA", image.size, (0, 0, 0, 0)), mask))

        for points in mask_contours(mask_path):
            draw.line(points + [points[0]], fill=(*color, 255), width=3)

        left, top, right, bottom = bbox
        draw.rectangle((left, top, right, bottom), outline=(*color, 255), width=3)
        draw_label(draw, f"{object_index:02d} {label}", (left, top), font, color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.convert("RGB").save(output_path)


def render_dataset(dataset_root: Path, output_subdir: str) -> None:
    for split in DATASET_SPLITS:
        image_dir = split_path(dataset_root, split, "images")
        label_dir = split_path(dataset_root, split, "labels")
        masks_by_stem = grouped_masks(split_path(dataset_root, split, "masks"))
        output_dir = split_path(dataset_root, split, output_subdir)
        for image_path in sorted(image_dir.glob("*.png")):
            masks = masks_by_stem.get(image_path.stem, [])
            if masks:
                render_mask_preview(image_path, masks, output_dir / image_path.name)
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                continue
            render_preview(image_path, label_path, output_dir / image_path.name)


if __name__ == "__main__":
    args = parse_args()
    render_dataset(Path(args.dataset), args.output_subdir)
