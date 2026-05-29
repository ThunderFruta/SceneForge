from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageChops, ImageFilter
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
    if not masked_crop_path.is_file():
        return {"object_dir": str(object_dir), "status": "skipped", "reason": "missing_masked_crop"}

    masked_crop = Image.open(masked_crop_path).convert("RGBA")
    context_crop, context_reference_name = load_context_reference(object_dir)
    image, mask, paste_box, output_alpha = build_inpaint_canvas(
        masked_crop=masked_crop,
        context_crop=context_crop,
        metadata=metadata,
        canvas_size=canvas_size,
    )

    save_completion_debug_artifacts(object_dir, image, mask, output_alpha, paste_box)

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
    completed_object, warning = compose_completed_object(
        result=result,
        paste_box=paste_box,
        output_alpha=output_alpha,
        source_rgba=masked_crop,
    )
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
        "context_crop": context_reference_name,
        "order_index": index,
    }
    if warning:
        record["completion_warning"] = warning
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
    object_rgb, visible_alpha = resize_rgba_crop(source, max_size=int(canvas_size * 0.78))
    output_alpha = estimate_expected_alpha(visible_alpha, metadata)
    visible_alpha = ImageChops.darker(visible_alpha, output_alpha)
    repaint_alpha = repaint_alpha_from_visible(output_alpha, visible_alpha)
    canvas = Image.new("RGB", (canvas_size, canvas_size), (245, 245, 240))
    x = (canvas_size - object_rgb.width) // 2
    y = (canvas_size - object_rgb.height) // 2
    paste_box = (x, y, x + object_rgb.width, y + object_rgb.height)
    paste_reference_context(canvas, context_crop, max_size=int(canvas_size * 0.24), avoid_box=paste_box)
    paste_object_source(canvas, object_rgb, visible_alpha, x, y)
    mask = Image.new("L", (canvas_size, canvas_size), 0)
    mask.paste(repaint_alpha, (x, y))
    return canvas, mask, paste_box, output_alpha


def estimate_expected_alpha(visible_alpha: Image.Image, metadata: dict[str, Any]) -> Image.Image:
    label = str(metadata.get("detector_label") or metadata.get("primitive_label") or "").lower()
    alpha = visible_alpha.convert("L")
    if "vase" in label:
        return close_alpha(mirror_alpha(alpha), 35)
    if "chair" in label:
        return close_alpha(alpha, 51)
    if "flower" in label:
        return close_alpha(alpha, 23)
    return close_alpha(alpha, 35)


def mirror_alpha(alpha: Image.Image) -> Image.Image:
    return ImageChops.lighter(alpha, alpha.transpose(Image.Transpose.FLIP_LEFT_RIGHT))


def close_alpha(alpha: Image.Image, size: int) -> Image.Image:
    kernel_size = max(3, int(size) | 1)
    kernel_size = min(kernel_size, 99)
    closed = alpha.filter(ImageFilter.MaxFilter(kernel_size)).filter(ImageFilter.MinFilter(kernel_size))
    return ImageChops.lighter(alpha, closed)


def repaint_alpha_from_visible(output_alpha: Image.Image, visible_alpha: Image.Image) -> Image.Image:
    missing = ImageChops.subtract(output_alpha.convert("L"), visible_alpha.convert("L"))
    if missing.getbbox() is None:
        return missing
    return missing.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(1.1))


def paste_reference_context(
    canvas: Image.Image,
    context_crop: Image.Image | None,
    *,
    max_size: int,
    avoid_box: tuple[int, int, int, int],
) -> None:
    if context_crop is None:
        return
    reference = context_crop.copy().convert("RGB")
    reference.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    if reference.width < 8 or reference.height < 8:
        return
    margin = 24
    candidates = (
        (margin, margin),
        (canvas.width - reference.width - margin, margin),
        (margin, canvas.height - reference.height - margin),
        (canvas.width - reference.width - margin, canvas.height - reference.height - margin),
    )
    for x, y in candidates:
        box = (x, y, x + reference.width, y + reference.height)
        if not boxes_overlap(box, avoid_box):
            canvas.paste(reference, (x, y))
            return


def load_context_reference(object_dir: Path) -> tuple[Image.Image | None, str | None]:
    focus_path = object_dir / "context_focus_crop.png"
    if focus_path.is_file():
        return Image.open(focus_path).convert("RGB"), "context_focus_crop.png"
    context_path = object_dir / "context_crop.png"
    if not context_path.is_file():
        return None, None
    context = Image.open(context_path).convert("RGB")
    mask_path = object_dir / "context_mask.png"
    if mask_path.is_file():
        focused = dim_context_outside_mask(context, Image.open(mask_path).convert("L"))
        try:
            focused.save(focus_path)
        except OSError:
            pass
        return focused, "context_focus_crop.png"
    return context, "context_crop.png"


def dim_context_outside_mask(context_crop: Image.Image, context_mask: Image.Image) -> Image.Image:
    base = context_crop.convert("RGB")
    mask = context_mask.convert("L").filter(ImageFilter.MaxFilter(25)).filter(ImageFilter.GaussianBlur(4.0))
    gray = base.convert("L").convert("RGB")
    dim = Image.blend(gray, Image.new("RGB", base.size, (20, 20, 20)), 0.58)
    focused = dim.copy()
    focused.paste(base, (0, 0), mask)
    return focused


def boxes_overlap(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> bool:
    return first[0] < second[2] and first[2] > second[0] and first[1] < second[3] and first[3] > second[1]


def save_completion_debug_artifacts(
    object_dir: Path,
    image: Image.Image,
    mask: Image.Image,
    output_alpha: Image.Image,
    paste_box: tuple[int, int, int, int],
) -> None:
    image.save(object_dir / "completion_input.png")
    mask.save(object_dir / "completion_mask.png")
    output_alpha.save(object_dir / "completion_expected_alpha.png")
    image.crop(paste_box).save(object_dir / "completion_input_crop.png")


def compose_completed_object(
    *,
    result: Image.Image,
    paste_box: tuple[int, int, int, int],
    output_alpha: Image.Image,
    source_rgba: Image.Image,
) -> tuple[Image.Image, str | None]:
    source = source_rgba.convert("RGBA")
    completed_rgb = result.crop(paste_box).convert("RGB")
    if completed_rgb.size != source.size:
        completed_rgb = completed_rgb.resize(source.size, Image.Resampling.LANCZOS)
    final_alpha = output_alpha.convert("L")
    if final_alpha.size != source.size:
        final_alpha = final_alpha.resize(source.size, Image.Resampling.LANCZOS)
    visible_alpha = ImageChops.darker(source.getchannel("A"), final_alpha)
    fill_alpha = ImageChops.subtract(final_alpha, visible_alpha)
    if generated_fill_is_black(completed_rgb, fill_alpha):
        source.putalpha(visible_alpha)
        return clear_transparent_rgb(source), "black_generated_fill_preserved_source"
    completed_object = completed_rgb.convert("RGBA")
    completed_object.putalpha(final_alpha)
    source.putalpha(visible_alpha)
    completed_object.paste(source, (0, 0), visible_alpha)
    completed_object.putalpha(final_alpha)
    return clear_transparent_rgb(completed_object), None


def clear_transparent_rgb(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    array = np.asarray(rgba, dtype=np.uint8).copy()
    array[array[:, :, 3] == 0, :3] = 0
    return Image.fromarray(array, mode="RGBA")


def generated_fill_is_black(image: Image.Image, fill_alpha: Image.Image) -> bool:
    mask = np.asarray(fill_alpha.convert("L"), dtype=np.uint8) > 24
    if int(mask.sum()) < 32:
        return False
    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)[mask]
    return float(pixels.mean()) <= 4.0 and int(pixels.max()) <= 8


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
