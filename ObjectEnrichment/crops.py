from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from PrimitiveFitting.masks import polygon_to_mask
from SceneGeometry.coordinate_contract import crop_coordinate_contract
from ShapeDetection.report import ObjectShapeDetection


def crop_box(detection: ObjectShapeDetection, width: int, height: int) -> tuple[int, int, int, int]:
    left, top, right, bottom = detection.bbox_xyxy
    x0 = max(0, min(width - 1, int(np.floor(left))))
    y0 = max(0, min(height - 1, int(np.floor(top))))
    x1 = max(x0 + 1, min(width, int(np.ceil(right))))
    y1 = max(y0 + 1, min(height, int(np.ceil(bottom))))
    return x0, y0, x1, y1


def write_object_crops(
    image: Image.Image,
    depth: np.ndarray,
    edge_image: Image.Image,
    detection: ObjectShapeDetection,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    box = crop_box(detection, image.width, image.height)
    mask = polygon_to_mask(detection.mask_polygon, image.width, image.height)

    rgb_crop = image.crop(box)
    mask_crop = Image.fromarray((mask.astype(np.uint8) * 255), mode="L").crop(box)
    depth_crop = Image.fromarray(np.clip(depth * 255.0, 0, 255).astype(np.uint8), mode="L").crop(box)
    edge_crop = edge_image.convert("L").crop(box)

    masked = rgb_crop.convert("RGBA")
    masked.putalpha(mask_crop)

    paths = {
        "rgb_crop": output_dir / "rgb_crop.png",
        "mask": output_dir / "mask.png",
        "depth_crop": output_dir / "depth_crop.png",
        "edge_crop": output_dir / "edge_crop.png",
        "masked_crop": output_dir / "masked_crop.png",
        "crop_metadata": output_dir / "crop_metadata.json",
    }
    rgb_crop.save(paths["rgb_crop"])
    mask_crop.save(paths["mask"])
    depth_crop.save(paths["depth_crop"])
    edge_crop.save(paths["edge_crop"])
    masked.save(paths["masked_crop"])
    paths["crop_metadata"].write_text(
        json.dumps(
            crop_coordinate_contract(
                parent_image_width=image.width,
                parent_image_height=image.height,
                crop_box_xyxy=box,
                detection_id=detection.id,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    evidence_path = output_dir / "evidence_stack.npz"
    np.savez_compressed(
        evidence_path,
        rgb=np.asarray(rgb_crop.convert("RGB"), dtype=np.uint8),
        mask=np.asarray(mask_crop, dtype=np.uint8),
        depth=np.asarray(depth_crop, dtype=np.uint8),
        edge=np.asarray(edge_crop, dtype=np.uint8),
    )
    paths["evidence_stack"] = evidence_path
    return paths
