from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from Input.Image.image_loader import load_rgb_image
from Segmentation.masks import polygon_to_mask


SCHEMA_VERSION = 1
DEFAULT_EMPTY_ROOM_MODEL = os.environ.get("SCENEFORGE_EMPTY_ROOM_MODEL", "gpt-image-1.5")
DEFAULT_OPENAI_IMAGE_QUALITY = os.environ.get("SCENEFORGE_OPENAI_IMAGE_QUALITY", "medium")
DEFAULT_OPENAI_TIMEOUT_SECONDS = float(os.environ.get("SCENEFORGE_OPENAI_TIMEOUT_SECONDS", "180"))
STRUCTURAL_LABELS = {
    "background",
    "ceiling",
    "concrete floor",
    "door",
    "floor",
    "plane",
    "road",
    "room",
    "wall",
    "window",
}


@dataclass(frozen=True)
class EmptyRoomInput:
    image: Image.Image
    detections: dict[str, Any]
    objects: list[dict[str, Any]]
    image_path: Path
    detections_path: Path


def generate_empty_room(
    *,
    image_path: str | Path,
    detections_path: str | Path,
    objects_dir: str | Path,
    output_dir: str | Path,
    backend: str = "openai-image",
    model: str = DEFAULT_EMPTY_ROOM_MODEL,
    fill_mode: str = "transparent",
    mask_dilation_px: int = 10,
    mask_feather_px: int = 0,
    include_detection_ids: set[int] | None = None,
    exclude_detection_ids: set[int] | None = None,
    allow_rectangular_fallback_masks: bool = False,
    max_mask_coverage: float = 0.55,
) -> dict[str, Any]:
    loaded = load_empty_room_input(image_path=image_path, detections_path=detections_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fill_mode = normalize_fill_mode(fill_mode)
    selected, excluded = select_foreground_objects(
        loaded.objects,
        include_detection_ids=include_detection_ids or set(),
        exclude_detection_ids=exclude_detection_ids or set(),
        allow_rectangular_fallback_masks=allow_rectangular_fallback_masks,
    )
    binary_mask = build_foreground_mask(
        selected,
        image_width=loaded.image.width,
        image_height=loaded.image.height,
        dilation_px=mask_dilation_px,
    )
    edit_mask = apply_mask_feather(binary_mask, mask_feather_px)

    mask_path = output / "empty_room_mask.png"
    foreground_removal_mask_path = output / "foreground_removal_mask.png"
    openai_mask_path = output / "empty_room_openai_mask.png"
    input_path = output / "empty_room_openai_input.png"
    edit_input_path = output / "empty_room_edit_input.png"
    empty_room_path = output / "empty_room.png"
    metadata_path = output / "empty_room_metadata.json"
    removal_mask_image = Image.fromarray(binary_mask, mode="L")
    removal_mask_image.save(mask_path)
    removal_mask_image.save(foreground_removal_mask_path)
    build_openai_edit_mask(edit_mask).save(openai_mask_path)
    edit_input = build_edit_input(loaded.image, edit_mask, fill_mode=fill_mode)
    edit_input.save(input_path)
    edit_input.save(edit_input_path)

    prompt = build_empty_room_prompt()
    warnings = build_mask_warnings(
        selected=selected,
        excluded=excluded,
        mask=binary_mask,
        max_mask_coverage=max_mask_coverage,
    )
    if backend == "fake":
        result_image = fake_empty_room_image(loaded.image, edit_mask)
        backend_model = "fake-empty-room"
    elif backend == "openai-image":
        result_image = call_openai_empty_room_edit(
            input_path=input_path,
            mask_path=openai_mask_path,
            prompt=prompt,
            model=model,
        )
        backend_model = model
    else:
        raise ValueError(f"Unsupported empty-room backend: {backend}")

    resolution_preserved = result_image.size == loaded.image.size
    if not resolution_preserved:
        warnings.append(
            f"empty_room_resized_from_{result_image.width}x{result_image.height}_to_{loaded.image.width}x{loaded.image.height}"
        )
        result_image = result_image.resize(loaded.image.size, Image.Resampling.LANCZOS)
    result_image.convert("RGB").save(empty_room_path)

    mask_quality_counts = count_mask_quality(loaded.objects)
    coverage = float(np.asarray(binary_mask, dtype=np.uint8).mean() / 255.0) if binary_mask.size else 0.0
    review_required = sorted(
        {
            int(item["id"])
            for item in loaded.objects
            if str(item.get("mask_quality", "")).lower() == "rectangular_fallback"
        }
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "source_image_path": str(loaded.image_path),
        "source_detections_path": str(loaded.detections_path),
        "source_object_mask_dir": str(objects_dir),
        "selected_removed_detection_ids": [int(item["id"]) for item in selected],
        "excluded_detection_ids": [{"id": int(item["id"]), "reason": item["reason"]} for item in excluded],
        "protected_structural_labels": sorted(STRUCTURAL_LABELS),
        "mask_quality_counts": mask_quality_counts,
        "review_required_detections": review_required,
        "mask_coverage_ratio": coverage,
        "mask_expansion_settings": {
            "dilation_px": int(mask_dilation_px),
            "feather_px": int(mask_feather_px),
        },
        "openai_input_image_path": str(input_path),
        "empty_room_edit_input_path": str(edit_input_path),
        "openai_mask_image_path": str(openai_mask_path),
        "output_image_path": str(empty_room_path),
        "empty_room_mask_path": str(mask_path),
        "foreground_removal_mask_path": str(foreground_removal_mask_path),
        "image_edit_backend": backend,
        "model": backend_model,
        "prompt": prompt,
        "reference_context_used": False,
        "fill_mode": fill_mode,
        "resolution_framing_preserved": resolution_preserved,
        "warnings": warnings,
        "needs_review": bool(warnings),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def load_empty_room_input(*, image_path: str | Path, detections_path: str | Path) -> EmptyRoomInput:
    image_path = Path(image_path)
    detections_path = Path(detections_path)
    image = load_rgb_image(image_path)
    try:
        detections = json.loads(detections_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise FileNotFoundError(f"--detections does not exist: {detections_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"--detections is not valid JSON: {detections_path}") from exc
    objects = detections.get("objects", [])
    if not isinstance(objects, list):
        raise ValueError("detections.json must contain an objects list.")
    return EmptyRoomInput(
        image=image,
        detections=detections,
        objects=[item for item in objects if isinstance(item, dict)],
        image_path=image_path,
        detections_path=detections_path,
    )


def select_foreground_objects(
    objects: list[dict[str, Any]],
    *,
    include_detection_ids: set[int],
    exclude_detection_ids: set[int],
    allow_rectangular_fallback_masks: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for item in objects:
        detection_id = int(item.get("id", 0))
        label = object_label(item)
        mask_quality = str(item.get("mask_quality", "")).lower()
        reason = None
        if include_detection_ids and detection_id not in include_detection_ids:
            reason = "not_in_include_detection_ids"
        elif detection_id in exclude_detection_ids:
            reason = "explicitly_excluded"
        elif is_structural_label(label) and detection_id not in include_detection_ids:
            reason = "protected_structural_label"
        elif mask_quality == "rectangular_fallback" and not allow_rectangular_fallback_masks:
            reason = "rectangular_fallback_mask"
        elif len(item.get("mask_polygon") or []) < 3:
            reason = "missing_or_invalid_mask_polygon"
        if reason:
            copy = dict(item)
            copy["reason"] = reason
            excluded.append(copy)
        else:
            selected.append(item)
    return selected, excluded


def object_label(item: dict[str, Any]) -> str:
    return str(item.get("detector_label") or item.get("primitive_label") or "").strip().lower()


def is_structural_label(label: str) -> bool:
    normalized = label.replace("_", " ").strip().lower()
    return normalized in STRUCTURAL_LABELS


def build_foreground_mask(
    objects: list[dict[str, Any]],
    *,
    image_width: int,
    image_height: int,
    dilation_px: int,
) -> np.ndarray:
    combined = np.zeros((image_height, image_width), dtype=bool)
    for item in objects:
        polygon = [(float(x), float(y)) for x, y in item.get("mask_polygon", [])]
        combined |= polygon_to_mask(polygon, image_width, image_height)
    image = Image.fromarray((combined.astype(np.uint8) * 255), mode="L")
    if dilation_px > 0:
        size = max(3, int(dilation_px) * 2 + 1)
        if size % 2 == 0:
            size += 1
        image = image.filter(ImageFilter.MaxFilter(size))
    return np.asarray(image, dtype=np.uint8)


def apply_mask_feather(mask: np.ndarray, feather_px: int) -> Image.Image:
    image = Image.fromarray(mask, mode="L")
    if feather_px > 0:
        image = image.filter(ImageFilter.GaussianBlur(float(feather_px)))
    return image


def build_edit_input(image: Image.Image, mask: Image.Image, *, fill_mode: str) -> Image.Image:
    base = image.convert("RGBA")
    alpha = Image.eval(mask.convert("L"), lambda value: 255 - value)
    if fill_mode == "transparent":
        result = base.copy()
        result.putalpha(alpha)
        return result
    fill_color = (0, 0, 0) if fill_mode == "black" else (128, 128, 128)
    result = image.convert("RGB")
    fill = Image.new("RGB", image.size, fill_color)
    result.paste(fill, (0, 0), mask)
    return result


def build_openai_edit_mask(mask: Image.Image) -> Image.Image:
    edit_alpha = mask.convert("L")
    keep_alpha = Image.eval(edit_alpha, lambda value: 255 - value)
    mask_rgba = Image.new("RGBA", keep_alpha.size, (255, 255, 255, 255))
    mask_rgba.putalpha(keep_alpha)
    return mask_rgba


def fake_empty_room_image(image: Image.Image, mask: Image.Image) -> Image.Image:
    base = image.convert("RGB")
    softened = base.filter(ImageFilter.GaussianBlur(10))
    neutral = Image.new("RGB", base.size, (154, 154, 148))
    fill = Image.blend(softened, neutral, 0.35)
    result = base.copy()
    result.paste(fill, (0, 0), mask)
    return result


def build_empty_room_prompt() -> str:
    return (
        "Edit the image into the same empty room with the exact same camera framing, perspective, lens, lighting, "
        "wall layout, floor layout, ceiling, trim, windows, doors, and material style. Fill only the transparent or "
        "marked removed foreground regions as plausible empty room surfaces. Remove movable furniture, props, people, "
        "plants, clutter, shadows, and contact patches from the marked regions. Do not add replacement furniture, rugs, "
        "platforms, display stands, text, logos, or a new room layout. Return a full-frame opaque image at the same aspect ratio."
    )


def call_openai_empty_room_edit(*, input_path: Path, mask_path: Path, prompt: str, model: str) -> Image.Image:
    key_source = ensure_openai_api_key()
    if not key_source:
        raise RuntimeError("OPENAI_API_KEY is required for --empty-room-backend openai-image")
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("Install the openai package to use --empty-room-backend openai-image.") from exc

    client = OpenAI()
    with input_path.open("rb") as image_file, mask_path.open("rb") as mask_file:
        kwargs = {
            "model": model,
            "image": image_file,
            "mask": mask_file,
            "prompt": prompt,
            "size": "auto",
            "quality": DEFAULT_OPENAI_IMAGE_QUALITY,
            "output_format": "png",
            "background": "opaque",
            "timeout": DEFAULT_OPENAI_TIMEOUT_SECONDS,
        }
        try:
            result = client.images.edit(**kwargs)
        except TypeError:
            kwargs.pop("background", None)
            kwargs.pop("output_format", None)
            try:
                result = client.images.edit(**kwargs)
            except Exception as exc:
                raise RuntimeError(f"OpenAI empty-room image edit failed: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"OpenAI empty-room image edit failed: {exc}") from exc
    image_base64 = result.data[0].b64_json
    return Image.open(io.BytesIO(base64.b64decode(image_base64))).convert("RGB")


def ensure_openai_api_key() -> str | None:
    if os.environ.get("OPENAI_API_KEY"):
        return "environment"
    loaded = load_openai_api_key_from_shell_config()
    return str(loaded) if loaded is not None else None


def load_openai_api_key_from_shell_config() -> Path | None:
    home = Path.home()
    for path in (
        home / ".bashrc",
        home / ".profile",
        home / ".bash_profile",
        home / ".zshrc",
    ):
        value = read_openai_api_key_assignment(path)
        if value:
            os.environ["OPENAI_API_KEY"] = value
            return path
    return None


def read_openai_api_key_assignment(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].strip()
        if not stripped.startswith("OPENAI_API_KEY="):
            continue
        value = stripped.split("=", 1)[1].strip()
        if not value:
            continue
        if value[0:1] == value[-1:] and value[0:1] in {"'", '"'}:
            value = value[1:-1]
        return value.strip()
    return None


def build_mask_warnings(
    *,
    selected: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    mask: np.ndarray,
    max_mask_coverage: float,
) -> list[str]:
    warnings: list[str] = []
    if not selected:
        warnings.append("no_foreground_objects_selected")
    coverage = float(np.asarray(mask, dtype=np.uint8).mean() / 255.0) if mask.size else 0.0
    if coverage > max_mask_coverage:
        warnings.append(f"mask_coverage_exceeds_{max_mask_coverage:.2f}")
    if any(item.get("reason") == "protected_structural_label" for item in excluded):
        warnings.append("structural_labels_excluded")
    if any(item.get("reason") == "rectangular_fallback_mask" for item in excluded):
        warnings.append("rectangular_fallback_masks_excluded")
    return warnings


def count_mask_quality(objects: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in objects:
        key = str(item.get("mask_quality") or "unspecified")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def normalize_fill_mode(value: str) -> str:
    if value not in {"transparent", "neutral", "black"}:
        raise ValueError("--fill-mode must be transparent, neutral, or black")
    return value


def parse_detection_id_set(value: str | None) -> set[int]:
    if not value:
        return set()
    ids: set[int] = set()
    for piece in value.split(","):
        piece = piece.strip()
        if not piece:
            continue
        ids.add(int(piece))
    return ids
