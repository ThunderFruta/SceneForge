from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageChops, ImageFilter

from MeshReconstruction.triposr_provider import TripoSRMeshProvider


def run_triposr_object_reconstruction(
    objects_dir: str | Path,
    *,
    model_dir: str | Path,
    device: str | None = "auto",
    source: str = "auto",
    max_objects: int = 0,
) -> dict[str, Any]:
    root = Path(objects_dir)
    if not root.is_dir():
        return write_manifest(root, [], "missing_objects_dir", model_dir=model_dir, device=device, source=source)

    object_dirs = [path for path in sorted(root.iterdir()) if path.is_dir()]
    selected_dirs = object_dirs if max_objects <= 0 else object_dirs[:max_objects]
    if not selected_dirs:
        return write_manifest(root, [], "no_objects", model_dir=model_dir, device=device, source=source)

    provider = TripoSRMeshProvider(model_dir=model_dir, device=device)
    records: list[dict[str, Any]] = []
    print(f"Running TripoSR object reconstruction for {len(selected_dirs)} of {len(object_dirs)} objects.", flush=True)
    for index, object_dir in enumerate(selected_dirs, start=1):
        print(f"TripoSR reconstruction {index}/{len(selected_dirs)}: {object_dir.name}", flush=True)
        records.append(
            reconstruct_object_dir(
                object_dir,
                provider=provider,
                source=source,
                order_index=index,
            )
        )
    return write_manifest(root, records, "complete", model_dir=model_dir, device=device, source=source)


def reconstruct_object_dir(
    object_dir: Path,
    *,
    provider: TripoSRMeshProvider,
    source: str,
    order_index: int,
) -> dict[str, Any]:
    source_path, source_kind = select_source_image(object_dir, source)
    if source_path is None:
        return {
            "object_dir": str(object_dir),
            "status": "skipped",
            "reason": "missing_source_image",
            "order_index": order_index,
        }

    input_path = object_dir / "triposr_input.png"
    mask_path = object_dir / "triposr_mask.png"
    mesh_path = object_dir / "triposr_mesh.obj"
    prepared_image, prepared_mask = prepare_triposr_input(source_path, object_dir)
    prepared_image.save(input_path)
    prepared_mask.save(mask_path)

    result = provider.reconstruct(input_path, mask_path, mesh_path)
    record = {
        "object_dir": str(object_dir),
        "status": result.status,
        "reason": result.reason,
        "source": source_kind,
        "source_image": source_path.name,
        "triposr_input": input_path.name,
        "triposr_mask": mask_path.name,
        "mesh": mesh_path.name if result.path is not None else None,
        "order_index": order_index,
    }
    (object_dir / "triposr_metadata.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return record


def select_source_image(object_dir: Path, source: str) -> tuple[Path | None, str | None]:
    completed = object_dir / "completed_crop.png"
    masked = object_dir / "masked_crop.png"
    if source == "completed":
        return (completed, "completed") if completed.is_file() else (None, None)
    if source == "masked":
        return (masked, "masked") if masked.is_file() else (None, None)
    if completed.is_file():
        return completed, "completed"
    if masked.is_file():
        return masked, "masked"
    return None, None


def prepare_triposr_input(source_path: Path, object_dir: Path) -> tuple[Image.Image, Image.Image]:
    source = Image.open(source_path).convert("RGBA")
    image = source.convert("RGB")
    alpha = source.getchannel("A")
    if alpha_has_foreground(alpha):
        mask = alpha
    else:
        mask = mask_from_original_alpha(object_dir, image.size)
        if mask is None:
            mask = foreground_mask_from_neutral_background(image)
    mask = clean_mask(mask.resize(image.size, Image.Resampling.LANCZOS))
    return image, mask


def alpha_has_foreground(alpha: Image.Image) -> bool:
    values = np.asarray(alpha.convert("L"), dtype=np.uint8)
    return int(values.min()) < 250 and int((values > 8).sum()) >= 64


def mask_from_original_alpha(object_dir: Path, size: tuple[int, int]) -> Image.Image | None:
    source_path = object_dir / "masked_crop.png"
    if not source_path.is_file():
        return None
    source = Image.open(source_path).convert("RGBA")
    alpha = source.getchannel("A")
    if not alpha_has_foreground(alpha):
        return None
    fitted = source.copy()
    fitted.thumbnail((int(size[0] * 0.88), int(size[1] * 0.88)), Image.Resampling.LANCZOS)
    canvas = Image.new("L", size, 0)
    x = (size[0] - fitted.width) // 2
    y = (size[1] - fitted.height) // 2
    canvas.paste(fitted.getchannel("A"), (x, y))
    return canvas


def foreground_mask_from_neutral_background(image: Image.Image) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    border = np.concatenate([rgb[0, :, :], rgb[-1, :, :], rgb[:, 0, :], rgb[:, -1, :]], axis=0)
    background = np.median(border, axis=0)
    distance = np.linalg.norm(rgb - background.reshape(1, 1, 3), axis=2)
    threshold = max(18.0, float(np.percentile(distance, 82)))
    return Image.fromarray((distance > threshold).astype(np.uint8) * 255, mode="L")


def clean_mask(mask: Image.Image) -> Image.Image:
    cleaned = mask.convert("L")
    cleaned = cleaned.point(lambda value: 255 if value > 24 else 0)
    if cleaned.getbbox() is None:
        return Image.new("L", cleaned.size, 255)
    cleaned = cleaned.filter(ImageFilter.MaxFilter(7))
    cleaned = cleaned.filter(ImageFilter.MinFilter(3))
    cleaned = cleaned.filter(ImageFilter.GaussianBlur(1.0))
    return ImageChops.lighter(cleaned, cleaned.point(lambda value: 255 if value > 64 else 0))


def write_manifest(
    objects_dir: Path,
    records: list[dict[str, Any]],
    status: str,
    *,
    model_dir: str | Path,
    device: str | None,
    source: str,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "status": status,
        "backend": "triposr",
        "model_dir": str(model_dir),
        "device": device,
        "source": source,
        "object_count": len(records),
        "objects": records,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    objects_dir.mkdir(parents=True, exist_ok=True)
    (objects_dir / "triposr_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload
