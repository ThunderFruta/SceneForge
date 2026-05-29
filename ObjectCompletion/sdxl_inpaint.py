from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter
from skimage.filters import threshold_otsu


DEFAULT_PROMPT_TEMPLATE = (
    "complete the marked {label} only, remove occluders, preserve perspective and material, "
    "return the target object only"
)
DEFAULT_NEGATIVE_PROMPT = "full room, full scene, furniture set, background scene, text, watermark, duplicate object, extra object, distorted, low quality"


def run_sdxl_object_completion(
    objects_dir: str | Path,
    model_dir: str | Path,
    *,
    device: str | None = "auto",
    steps: int = 24,
    guidance_scale: float = 6.5,
    strength: float = 0.55,
    canvas_size: int = 1024,
    seed: int = 20260528,
    max_objects: int = 16,
) -> dict[str, Any]:
    root = Path(objects_dir)
    if not root.is_dir():
        return write_manifest(root, [], "missing_objects_dir")

    object_dirs = [path for path in sorted(root.iterdir()) if path.is_dir()]
    if not object_dirs:
        return write_manifest(root, [], "no_objects")

    pipe, torch, generator_device = load_pipeline(model_dir, device)
    generator = torch.Generator(device=generator_device).manual_seed(int(seed))
    records: list[dict[str, Any]] = []
    for index, object_dir in enumerate(object_dirs[:max_objects], start=1):
        record = complete_object_dir(
            object_dir,
            pipe=pipe,
            generator=generator,
            steps=steps,
            guidance_scale=guidance_scale,
            strength=strength,
            canvas_size=canvas_size,
            index=index,
        )
        records.append(record)

    return write_manifest(root, records, "complete")


def load_pipeline(model_dir: str | Path, device: str | None):
    try:
        import torch
        from diffusers import StableDiffusionXLInpaintPipeline
    except Exception as exc:
        raise RuntimeError(
            "SDXL inpainting requires diffusers. Install project requirements before using "
            "--completion-backend sdxl-inpaint."
        ) from exc

    requested_device = device if device not in (None, "auto") else ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if str(requested_device).startswith("cuda") else torch.float32
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        str(model_dir),
        torch_dtype=torch_dtype,
        use_safetensors=True,
        local_files_only=True,
    )
    if str(requested_device).startswith("cuda"):
        pipe = pipe.to(requested_device)
        generator_device = requested_device
    else:
        pipe = pipe.to("cpu")
        generator_device = "cpu"
    return pipe, torch, generator_device


def complete_object_dir(
    object_dir: Path,
    *,
    pipe,
    generator,
    steps: int,
    guidance_scale: float,
    strength: float,
    canvas_size: int,
    index: int,
) -> dict[str, Any]:
    metadata = read_metadata(object_dir / "metadata.json")
    label = str(metadata.get("detector_label") or metadata.get("primitive_label") or "object")
    prompt = DEFAULT_PROMPT_TEMPLATE.format(label=label)
    masked_crop_path = object_dir / "masked_crop.png"
    context_crop_path = object_dir / "context_crop.png"
    if not masked_crop_path.is_file():
        return {"object_dir": str(object_dir), "status": "skipped", "reason": "missing_masked_crop"}

    masked_crop = Image.open(masked_crop_path).convert("RGBA")
    context_crop = Image.open(context_crop_path).convert("RGB") if context_crop_path.is_file() else None
    image, mask, paste_box, output_alpha = build_inpaint_canvas(
        masked_crop=masked_crop,
        context_crop=context_crop,
        metadata=metadata,
        canvas_size=canvas_size,
    )

    result = pipe(
        prompt=prompt,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        image=image,
        mask_image=mask,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        strength=float(strength),
        generator=generator,
    ).images[0].convert("RGB")
    completed_rgb = result.crop(paste_box).convert("RGB")
    completed_object = completed_rgb.convert("RGBA")
    completed_object.putalpha(object_only_alpha(completed_rgb, output_alpha))
    completed_object.save(object_dir / "completed_crop.png")

    record = {
        "object_dir": str(object_dir),
        "status": "complete",
        "prompt": prompt,
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        "model": "sdxl-inpaint",
        "steps": int(steps),
        "guidance_scale": float(guidance_scale),
        "strength": float(strength),
        "canvas_size": int(canvas_size),
        "paste_box_xyxy": list(paste_box),
        "completed_crop": "completed_crop.png",
        "source_crop": "masked_crop.png",
        "context_crop": "context_crop.png" if context_crop_path.is_file() else None,
        "order_index": index,
    }
    (object_dir / "completion_metadata.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return record


def build_inpaint_canvas(
    masked_crop: Image.Image,
    context_crop: Image.Image | None,
    metadata: dict[str, Any],
    *,
    canvas_size: int,
) -> tuple[Image.Image, Image.Image, tuple[int, int, int, int], Image.Image]:
    source = masked_crop.convert("RGBA")
    if context_crop is None:
        object_rgb, object_alpha = resize_rgba_crop(source, max_size=int(canvas_size * 0.72))
        canvas = Image.new("RGB", (canvas_size, canvas_size), mean_background_color(object_rgb))
        x = (canvas_size - object_rgb.width) // 2
        y = (canvas_size - object_rgb.height) // 2
        paste_object_source(canvas, object_rgb, object_alpha, x, y)
        mask = Image.new("L", (canvas_size, canvas_size), 0)
        mask.paste(Image.eval(object_alpha.filter(ImageFilter.MaxFilter(5)), lambda value: 255 - value), (x, y))
        return canvas, mask, (x, y, x + object_rgb.width, y + object_rgb.height), object_alpha

    context = context_crop.convert("RGB")
    object_box = object_box_in_context(metadata, source.size, context_crop.size)
    canvas, crop_box = fit_context_on_canvas(context, canvas_size)
    draw_reference_marker(canvas, metadata, context_crop.size, (crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]), offset=crop_box[:2])

    scale_x = (crop_box[2] - crop_box[0]) / max(1, context_crop.width)
    scale_y = (crop_box[3] - crop_box[1]) / max(1, context_crop.height)
    object_x0 = crop_box[0] + int(round(object_box[0] * scale_x))
    object_y0 = crop_box[1] + int(round(object_box[1] * scale_y))
    object_x1 = crop_box[0] + int(round(object_box[2] * scale_x))
    object_y1 = crop_box[1] + int(round(object_box[3] * scale_y))
    paste_box = (
        max(0, object_x0),
        max(0, object_y0),
        min(canvas_size, object_x1),
        min(canvas_size, object_y1),
    )

    mask = Image.new("L", (canvas_size, canvas_size), 0)
    mask_region = Image.new("L", (max(1, paste_box[2] - paste_box[0]), max(1, paste_box[3] - paste_box[1])), 255)
    mask.paste(mask_region, paste_box[:2])
    source_alpha = source.getchannel("A").resize(mask_region.size, Image.Resampling.LANCZOS)
    output_alpha = source_alpha.filter(ImageFilter.MaxFilter(31))
    return canvas, mask, paste_box, output_alpha


def object_only_alpha(image: Image.Image, seed_alpha: Image.Image) -> Image.Image:
    seed = seed_alpha.convert("L")
    if seed.getbbox() is None:
        return seed
    array = np.asarray(image.convert("RGB"), dtype=np.int16)
    border = np.concatenate(
        [
            array[0, :, :],
            array[-1, :, :],
            array[:, 0, :],
            array[:, -1, :],
        ],
        axis=0,
    )
    background = np.median(border, axis=0)
    distance = np.linalg.norm(array - background.reshape(1, 1, 3), axis=2)
    try:
        threshold = max(12.0, float(threshold_otsu(distance)))
    except ValueError:
        threshold = 18.0
    color_alpha = Image.fromarray((distance > threshold).astype(np.uint8) * 255, mode="L")
    expanded_seed = seed.filter(ImageFilter.MaxFilter(51))
    combined = Image.composite(Image.new("L", seed.size, 255), Image.new("L", seed.size, 0), expanded_seed)
    combined = Image.composite(combined, Image.new("L", seed.size, 0), color_alpha.filter(ImageFilter.MaxFilter(9)))
    combined = Image.composite(Image.new("L", seed.size, 255), combined, seed.filter(ImageFilter.MaxFilter(5)))
    return combined.filter(ImageFilter.GaussianBlur(1.2))


def resize_rgba_crop(image: Image.Image, *, max_size: int) -> tuple[Image.Image, Image.Image]:
    resized = image.copy()
    resized.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return resized.convert("RGB"), resized.getchannel("A")


def paste_object_source(canvas: Image.Image, object_rgb: Image.Image, object_alpha: Image.Image, x: int, y: int) -> None:
    base = Image.new("RGB", object_rgb.size, mean_background_color(canvas))
    base.paste(object_rgb, mask=object_alpha)
    canvas.paste(base, (x, y))


def fit_on_white(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    fitted = image.copy().convert("RGB")
    fitted.thumbnail(size, Image.Resampling.LANCZOS)
    panel = Image.new("RGB", size, (245, 245, 240))
    x = (size[0] - fitted.width) // 2
    y = (size[1] - fitted.height) // 2
    panel.paste(fitted, (x, y))
    return panel


def fit_context_on_canvas(image: Image.Image, canvas_size: int) -> tuple[Image.Image, tuple[int, int, int, int]]:
    fitted = image.copy().convert("RGB")
    fitted.thumbnail((canvas_size, canvas_size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (canvas_size, canvas_size), mean_background_color(fitted))
    x = (canvas_size - fitted.width) // 2
    y = (canvas_size - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas, (x, y, x + fitted.width, y + fitted.height)


def fit_rgba_on_white(image: Image.Image, size: tuple[int, int]) -> tuple[Image.Image, Image.Image]:
    fitted = image.copy().convert("RGBA")
    fitted.thumbnail(size, Image.Resampling.LANCZOS)
    panel = Image.new("RGB", size, (245, 245, 240))
    alpha_panel = Image.new("L", size, 0)
    x = (size[0] - fitted.width) // 2
    y = (size[1] - fitted.height) // 2
    panel.paste(fitted.convert("RGB"), (x, y), fitted.getchannel("A"))
    alpha_panel.paste(fitted.getchannel("A"), (x, y))
    return panel, alpha_panel


def draw_reference_marker(
    panel: Image.Image,
    metadata: dict[str, Any],
    context_size: tuple[int, int],
    panel_size: tuple[int, int],
    offset: tuple[int, int] = (0, 0),
) -> None:
    from PIL import ImageDraw

    object_box = object_box_in_context(metadata, (1, 1), context_size)
    scale = min(panel_size[0] / max(1, context_size[0]), panel_size[1] / max(1, context_size[1]))
    offset_x = (panel_size[0] - context_size[0] * scale) / 2.0
    offset_y = (panel_size[1] - context_size[1] * scale) / 2.0
    box = (
        offset[0] + offset_x + object_box[0] * scale,
        offset[1] + offset_y + object_box[1] * scale,
        offset[0] + offset_x + object_box[2] * scale,
        offset[1] + offset_y + object_box[3] * scale,
    )
    draw = ImageDraw.Draw(panel, "RGBA")
    draw.rectangle(box, outline=(255, 38, 0, 255), width=4)
    draw.rectangle(box, fill=(255, 38, 0, 28))


def object_box_in_context(
    metadata: dict[str, Any],
    source_size: tuple[int, int],
    context_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    crop_box = metadata.get("crop_box_xyxy")
    context_box = metadata.get("context_box_xyxy")
    if (
        isinstance(crop_box, list)
        and isinstance(context_box, list)
        and len(crop_box) == 4
        and len(context_box) == 4
    ):
        return (
            float(crop_box[0]) - float(context_box[0]),
            float(crop_box[1]) - float(context_box[1]),
            float(crop_box[2]) - float(context_box[0]),
            float(crop_box[3]) - float(context_box[1]),
        )
    width, height = source_size
    context_width, context_height = context_size
    x = (context_width - width) / 2.0
    y = (context_height - height) / 2.0
    return (x, y, x + width, y + height)


def mean_background_color(image: Image.Image) -> tuple[int, int, int]:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if array.size == 0:
        return (245, 245, 240)
    color = array.reshape(-1, 3).mean(axis=0)
    return tuple(int(value) for value in color)


def read_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def write_manifest(root: Path, records: list[dict[str, Any]], status: str, backend: str = "sdxl-inpaint") -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "backend": backend,
        "status": status,
        "object_count": len(records),
        "objects": records,
    }
    (root / "completion_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
