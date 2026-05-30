from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from Segmentation.masks import polygon_to_mask
from ShapeDetection.report import ObjectShapeDetection


def write_object_masks(
    image: Image.Image,
    objects: list[ObjectShapeDetection],
    output_dir: str | Path,
) -> None:
    root = Path(output_dir)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    for item in sorted(objects, key=lambda object_item: object_item.id):
        object_dir = root / object_folder_name(item)
        object_dir.mkdir(parents=True, exist_ok=True)
        segmentation_dir = object_dir / "artifacts" / "segmentation"
        segmentation_dir.mkdir(parents=True, exist_ok=True)
        box = crop_box(item, image.width, image.height)
        context_box = expanded_crop_box(box, image.width, image.height)
        mask = polygon_to_mask(item.mask_polygon, image.width, image.height)
        full_mask = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        rgb_crop = image.crop(box)
        mask_crop = full_mask.crop(box)
        masked_crop = rgb_crop.convert("RGBA")
        masked_crop.putalpha(mask_crop)
        context_crop = image.crop(context_box)
        context_mask = full_mask.crop(context_box)
        context_masked_crop = context_crop.convert("RGBA")
        context_masked_crop.putalpha(context_mask)
        context_focus_crop = dim_context_outside_mask(context_crop, context_mask)

        full_mask.save(segmentation_dir / "full_mask.png")
        rgb_crop.save(segmentation_dir / "rgb_crop.png")
        mask_crop.save(segmentation_dir / "mask.png")
        masked_crop.save(object_dir / "masked_crop.png")
        context_crop.save(segmentation_dir / "context_crop.png")
        context_mask.save(segmentation_dir / "context_mask.png")
        context_masked_crop.save(segmentation_dir / "context_masked_crop.png")
        context_focus_crop.save(segmentation_dir / "context_focus_crop.png")
        (object_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "id": item.id,
                    "detector_label": item.detector_label,
                    "detector_confidence": item.detector_confidence,
                    "primitive_label": item.primitive_label,
                    "primitive_confidence": item.primitive_confidence,
                    "primitive_label_source": item.primitive_label_source,
                    "bbox_xyxy": [round(value, 3) for value in item.bbox_xyxy],
                    "crop_box_xyxy": list(box),
                    "context_box_xyxy": list(context_box),
                    "image_width": image.width,
                    "image_height": image.height,
                    "mask_polygon_points": len(item.mask_polygon),
                    "mask_pixels": int(mask.sum()),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


def object_folder_name(item: ObjectShapeDetection) -> str:
    label = safe_label(item.detector_label or item.primitive_label or "object")
    return f"{item.id:02d}_{label}"


def safe_label(value: str) -> str:
    label = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    label = label.strip("_")
    return label[:48] or "object"


def dim_context_outside_mask(context_crop: Image.Image, context_mask: Image.Image) -> Image.Image:
    base = context_crop.convert("RGB")
    mask = context_mask.convert("L").filter(ImageFilter.MaxFilter(25)).filter(ImageFilter.GaussianBlur(4.0))
    gray = base.convert("L").convert("RGB")
    dim = Image.blend(gray, Image.new("RGB", base.size, (20, 20, 20)), 0.58)
    focused = dim.copy()
    focused.paste(base, (0, 0), mask)
    return focused


def crop_box(detection: ObjectShapeDetection, width: int, height: int) -> tuple[int, int, int, int]:
    left, top, right, bottom = detection.bbox_xyxy
    x0 = max(0, min(width - 1, int(np.floor(left))))
    y0 = max(0, min(height - 1, int(np.floor(top))))
    x1 = max(x0 + 1, min(width, int(np.ceil(right))))
    y1 = max(y0 + 1, min(height, int(np.ceil(bottom))))
    return x0, y0, x1, y1


def expanded_crop_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
    *,
    padding_ratio: float = 0.55,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    box_width = max(1, x1 - x0)
    box_height = max(1, y1 - y0)
    pad_x = int(np.ceil(box_width * padding_ratio))
    pad_y = int(np.ceil(box_height * padding_ratio))
    return (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(width, x1 + pad_x),
        min(height, y1 + pad_y),
    )
