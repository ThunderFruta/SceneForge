from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from Tools.Dataset.rgbd_curriculum import DATASET_SPLITS, split_path
from Segmentation.primitive_3d import FEATURE_NAMES


MASK_NAME_RE = re.compile(r"^(?P<stem>.+)_(?P<object_index>\d+)_(?P<class_name>[a-z_]+)\.png$")


def write_instance_manifest(
    dataset_root: Path,
    *,
    mask_subdir: str = "masks",
    output_name: str = "instance_dataset_manifest.json",
) -> Path:
    """Write a detector-neutral instance-mask manifest for future non-YOLO models.

    The manifest describes RGB/depth inputs and per-object visible masks. YOLO
    labels can still be generated beside it, but this file is the scaffolded
    contract for depth/edge/3D instance segmentation training.
    """
    dataset_root = Path(dataset_root)
    manifest = {
        "schema_version": 1,
        "dataset_root": str(dataset_root),
        "detector_training_contract": {
            "input_contract": "rgb_depth_camera_to_visible_point_cloud",
            "input_channels": list(FEATURE_NAMES),
            "output_contract": "class_agnostic_instance_masks",
            "primitive_label_policy": "geometry_fitting_downstream",
            "legacy_yolo_labels": "compatibility_only",
        },
        "splits": {},
    }
    for split in DATASET_SPLITS:
        manifest["splits"][split] = split_manifest(dataset_root, split, mask_subdir)

    output_path = dataset_root / output_name
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def split_manifest(dataset_root: Path, split: str, mask_subdir: str) -> dict:
    image_dir = split_path(dataset_root, split, "images")
    depth_dir = split_path(dataset_root, split, "depth")
    mask_dir = split_path(dataset_root, split, mask_subdir)
    point_cloud_dir = split_path(dataset_root, split, "point_cloud")
    masks_by_stem = grouped_masks(mask_dir) if mask_dir.is_dir() else {}
    samples: list[dict] = []

    if image_dir.is_dir():
        for image_path in sorted(image_dir.glob("*.png")):
            stem = image_path.stem
            depth_path = depth_dir / image_path.name
            camera_path = image_path.parent / "camera.json"
            samples.append(
                {
                    "id": stem,
                    "rgb": relative_path(image_path, dataset_root),
                    "depth": relative_path(depth_path, dataset_root) if depth_path.is_file() else None,
                    "camera": relative_path(camera_path, dataset_root) if camera_path.is_file() else None,
                    "point_cloud": relative_path(point_cloud_dir / f"{stem}.npz", dataset_root),
                    "objects": [
                        {
                            "class_name": class_name,
                            "object_index": object_index,
                            "visible_mask": relative_path(mask_path, dataset_root),
                            "point_label": object_index + 1,
                        }
                        for object_index, class_name, mask_path in masks_by_stem.get(stem, [])
                    ],
                }
            )

    object_count = sum(len(sample["objects"]) for sample in samples)
    return {
        "sample_count": len(samples),
        "object_count": object_count,
        "samples": samples,
    }


def grouped_masks(mask_dir: Path) -> dict[str, list[tuple[int, str, Path]]]:
    grouped: dict[str, list[tuple[int, str, Path]]] = defaultdict(list)
    for path in sorted(mask_dir.glob("*.png")):
        match = MASK_NAME_RE.match(path.name)
        if not match:
            continue
        grouped[match.group("stem")].append(
            (
                int(match.group("object_index")),
                match.group("class_name"),
                path,
            )
        )
    return grouped


def relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
