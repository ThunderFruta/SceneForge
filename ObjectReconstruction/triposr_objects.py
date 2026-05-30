from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from ObjectReconstruction.triposr_provider import TripoSRMeshProvider

RECONSTRUCTION_ARTIFACTS_DIR = Path("artifacts") / "reconstruction"
COMPLETED_MASK_NAME = str(RECONSTRUCTION_ARTIFACTS_DIR / "completed_mask.png")
COMPLETED_MASK_METADATA_NAME = str(RECONSTRUCTION_ARTIFACTS_DIR / "completed_mask_metadata.json")
SHARED_RECONSTRUCTION_OUTPUTS = (
    COMPLETED_MASK_NAME,
    COMPLETED_MASK_METADATA_NAME,
)
TRIPOSR_RECONSTRUCTION_OUTPUTS = (
    "triposr_input.png",
    "triposr_mask.png",
    "triposr_mesh.obj",
    "triposr_metadata.json",
)


@dataclass
class PreparedReconstructionInput:
    image: Image.Image
    mask: Image.Image
    mask_source: str
    completed_mask_path: str | None = None


def run_triposr_object_reconstruction(
    objects_dir: str | Path,
    *,
    model_dir: str | Path,
    device: str | None = "auto",
    source: str = "auto",
    max_objects: int = 0,
    completed_mask_backend: str = "auto",
    completed_mask_segmenter: Any | None = None,
    completed_mask_prompt: str | None = None,
) -> dict[str, Any]:
    root = Path(objects_dir)
    if not root.is_dir():
        return write_manifest(root, [], "missing_objects_dir", model_dir=model_dir, device=device, source=source)

    object_dirs = [path for path in sorted(root.iterdir()) if path.is_dir()]
    selected_dirs = object_dirs if max_objects <= 0 else object_dirs[:max_objects]
    if not selected_dirs:
        return write_manifest(root, [], "no_objects", model_dir=model_dir, device=device, source=source)
    clean_reconstruction_outputs(root, selected_dirs, backend="triposr")

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
                completed_mask_backend=completed_mask_backend,
                completed_mask_segmenter=completed_mask_segmenter,
                completed_mask_prompt=completed_mask_prompt,
                order_index=index,
            )
        )
    return write_manifest(root, records, "complete", model_dir=model_dir, device=device, source=source)


def reconstruct_object_dir(
    object_dir: Path,
    *,
    provider: TripoSRMeshProvider,
    source: str,
    completed_mask_backend: str,
    completed_mask_segmenter: Any | None,
    completed_mask_prompt: str | None,
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

    artifacts_dir = reconstruction_artifacts_dir(object_dir)
    input_path = artifacts_dir / "triposr_input.png"
    mask_path = artifacts_dir / "triposr_mask.png"
    mesh_path = object_dir / "triposr_mesh.obj"
    prepared = prepare_reconstruction_input(
        source_path,
        object_dir,
        completed_mask_backend=completed_mask_backend,
        completed_mask_segmenter=completed_mask_segmenter,
        completed_mask_prompt=completed_mask_prompt,
    )
    prepared_image = prepared.image
    prepared_mask = prepared.mask
    prepared_image.save(input_path)
    prepared_mask.save(mask_path)

    result = provider.reconstruct(input_path, mask_path, mesh_path)
    record = {
        "object_dir": str(object_dir),
        "status": result.status,
        "reason": result.reason,
        "source": source_kind,
        "source_image": source_path.name,
        "mask_source": prepared.mask_source,
        "completed_mask": prepared.completed_mask_path,
        "triposr_input": relative_to_object_dir(object_dir, input_path),
        "triposr_mask": relative_to_object_dir(object_dir, mask_path),
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
    prepared = prepare_reconstruction_input(source_path, object_dir)
    return prepared.image, prepared.mask


def prepare_reconstruction_input(
    source_path: Path,
    object_dir: Path,
    *,
    completed_mask_backend: str = "auto",
    completed_mask_segmenter: Any | None = None,
    completed_mask_prompt: str | None = None,
) -> PreparedReconstructionInput:
    source = Image.open(source_path).convert("RGBA")
    image = source.convert("RGB")
    alpha = source.getchannel("A")
    if alpha_has_foreground(alpha):
        mask = alpha
        mask_source = "source_alpha"
        completed_mask_path = None
    elif source_path.name == "completed_crop.png" and completed_mask_backend != "original-alpha":
        mask, mask_source = completed_mask_for_source(
            image,
            object_dir,
            backend=completed_mask_backend,
            segmenter=completed_mask_segmenter,
            text_prompt=completed_mask_prompt,
        )
        completed_mask_path = COMPLETED_MASK_NAME
    else:
        mask = mask_from_original_alpha(object_dir, image.size)
        mask_source = "original_masked_crop_alpha" if mask is not None else "neutral_background_foreground"
        if mask is None:
            mask = foreground_mask_from_neutral_background(image)
    mask = clean_mask(mask.resize(image.size, Image.Resampling.LANCZOS))
    return PreparedReconstructionInput(
        image=image,
        mask=mask,
        mask_source=mask_source,
        completed_mask_path=completed_mask_path,
    )


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


def completed_mask_for_source(
    image: Image.Image,
    object_dir: Path,
    *,
    backend: str,
    segmenter: Any | None,
    text_prompt: str | None,
) -> tuple[Image.Image, str]:
    existing_path = object_dir / COMPLETED_MASK_NAME
    existing_path.parent.mkdir(parents=True, exist_ok=True)
    if existing_path.is_file() and (backend == "auto" or segmenter is None):
        return Image.open(existing_path).convert("L"), "completed_mask"

    mask: Image.Image | None = None
    mask_source = "completed_neutral_background_foreground"
    if backend in {"auto", "sam3"} and segmenter is not None:
        mask = completed_mask_from_sam3(image, segmenter, text_prompt)
        if mask is not None:
            mask_source = "completed_sam3"

    if mask is None:
        mask = foreground_mask_from_neutral_background(image)

    mask = clean_mask(mask.resize(image.size, Image.Resampling.LANCZOS))
    mask.save(existing_path)
    write_completed_mask_metadata(
        object_dir,
        backend=backend,
        mask_source=mask_source,
        text_prompt=text_prompt,
        image_size=image.size,
    )
    return mask, mask_source


def completed_mask_from_sam3(image: Image.Image, segmenter: Any, text_prompt: str | None) -> Image.Image | None:
    if text_prompt and hasattr(segmenter, "text_prompt"):
        segmenter.text_prompt = text_prompt
    detections = segmenter.detect(image)
    width, height = image.size
    min_area = max(64, int(width * height * 0.002))
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    kept = 0
    for detection in detections:
        polygon = getattr(detection, "mask_polygon", None) or []
        if len(polygon) < 3:
            continue
        candidate = Image.new("L", image.size, 0)
        ImageDraw.Draw(candidate).polygon([(float(x), float(y)) for x, y in polygon], fill=255)
        if int(np.asarray(candidate, dtype=np.uint8).sum() // 255) < min_area:
            continue
        draw.bitmap((0, 0), candidate, fill=255)
        kept += 1
    return mask if kept else None


def write_completed_mask_metadata(
    object_dir: Path,
    *,
    backend: str,
    mask_source: str,
    text_prompt: str | None,
    image_size: tuple[int, int],
) -> None:
    mask_path = object_dir / COMPLETED_MASK_NAME
    mask = Image.open(mask_path).convert("L")
    values = np.asarray(mask, dtype=np.uint8)
    payload = {
        "schema_version": 1,
        "mask": COMPLETED_MASK_NAME,
        "backend": backend,
        "mask_source": mask_source,
        "text_prompt": text_prompt,
        "image_width": image_size[0],
        "image_height": image_size[1],
        "foreground_pixels": int((values > 0).sum()),
        "coverage_ratio": float((values > 0).mean()),
    }
    metadata_path = object_dir / COMPLETED_MASK_METADATA_NAME
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def reconstruction_artifacts_dir(object_dir: Path) -> Path:
    path = object_dir / RECONSTRUCTION_ARTIFACTS_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def relative_to_object_dir(object_dir: Path, path: Path) -> str:
    return path.relative_to(object_dir).as_posix()


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


def clean_reconstruction_outputs(objects_dir: Path, object_dirs: list[Path], *, backend: str) -> int:
    names = list(SHARED_RECONSTRUCTION_OUTPUTS)
    root_names: list[str] = []
    if backend == "triposr":
        names.extend(TRIPOSR_RECONSTRUCTION_OUTPUTS)
        root_names.append("triposr_manifest.json")
    elif backend == "hunyuan3d":
        names.extend(hunyuan3d_reconstruction_outputs())
        root_names.append("hunyuan3d_manifest.json")
    else:
        raise ValueError(f"Unsupported reconstruction cleanup backend: {backend}")

    removed = 0
    for object_dir in object_dirs:
        for name in names:
            path = object_dir / name
            if path.is_file():
                path.unlink()
                removed += 1
        for subdir in (object_dir / RECONSTRUCTION_ARTIFACTS_DIR, object_dir / "artifacts" / "textures"):
            if subdir.is_dir():
                for child in sorted(subdir.rglob("*"), reverse=True):
                    if child.is_file():
                        child.unlink()
                        removed += 1
                    elif child.is_dir():
                        child.rmdir()
                subdir.rmdir()
    for name in root_names:
        path = objects_dir / name
        if path.is_file():
            path.unlink()
            removed += 1
    return removed


def hunyuan3d_reconstruction_outputs() -> tuple[str, ...]:
    return (
        "hunyuan3d_input.png",
        "hunyuan3d_mask.png",
        "hunyuan3d_mesh.obj",
        "hunyuan3d_mesh.glb",
        "hunyuan3d_metadata.json",
        "hunyuan3d_textured.obj",
        "hunyuan3d_textured.glb",
        "hunyuan3d_textured.mtl",
        "hunyuan3d_textured.jpg",
        "hunyuan3d_textured.png",
        "hunyuan3d_textured_metallic.jpg",
        "hunyuan3d_textured_metallic.png",
        "hunyuan3d_textured_roughness.jpg",
        "hunyuan3d_textured_roughness.png",
        "white_mesh_remesh.obj",
    )
