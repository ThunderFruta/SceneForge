from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from PrimitiveFitting.masks import polygon_to_mask
from ShapeDetection.report import ObjectShapeDetection


def write_object_masks(
    image: Image.Image,
    objects: list[ObjectShapeDetection],
    output_dir: str | Path,
) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for item in sorted(objects, key=lambda object_item: object_item.id):
        object_dir = root / object_folder_name(item)
        object_dir.mkdir(parents=True, exist_ok=True)
        box = crop_box(item, image.width, image.height)
        mask = polygon_to_mask(item.mask_polygon, image.width, image.height)
        full_mask = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        rgb_crop = image.crop(box)
        mask_crop = full_mask.crop(box)
        masked_crop = rgb_crop.convert("RGBA")
        masked_crop.putalpha(mask_crop)

        full_mask.save(object_dir / "full_mask.png")
        rgb_crop.save(object_dir / "rgb_crop.png")
        mask_crop.save(object_dir / "mask.png")
        masked_crop.save(object_dir / "masked_crop.png")
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


def crop_box(detection: ObjectShapeDetection, width: int, height: int) -> tuple[int, int, int, int]:
    left, top, right, bottom = detection.bbox_xyxy
    x0 = max(0, min(width - 1, int(np.floor(left))))
    y0 = max(0, min(height - 1, int(np.floor(top))))
    x1 = max(x0 + 1, min(width, int(np.ceil(right))))
    y1 = max(y0 + 1, min(height, int(np.ceil(bottom))))
    return x0, y0, x1, y1
