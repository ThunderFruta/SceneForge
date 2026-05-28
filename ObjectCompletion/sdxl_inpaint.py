from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


DEFAULT_PROMPT_TEMPLATE = (
    "use the left reference scene for context, complete the marked {label} in the right panel, "
    "return a single isolated complete {label}, white background, no room, no extra objects"
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
    completed_object = result.crop(paste_box).convert("RGBA")
    completed_object.putalpha(output_alpha)
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
        return canvas, mask, (x, y, x + object_rgb.width, y + object_rgb.height), Image.new("L", object_rgb.size, 255)

    context = context_crop.convert("RGB")
    canvas = Image.new("RGB", (canvas_size, canvas_size), (245, 245, 240))
    mask = Image.new("L", (canvas_size, canvas_size), 0)

    panel_gap = max(16, canvas_size // 40)
    margin = max(24, canvas_size // 32)
    panel_width = (canvas_size - 2 * margin - panel_gap) // 2
    panel_height = canvas_size - 2 * margin
    left_box = (margin, margin, margin + panel_width, margin + panel_height)
    right_box = (margin + panel_width + panel_gap, margin, margin + 2 * panel_width + panel_gap, margin + panel_height)

    context_panel = fit_on_white(context, (panel_width, panel_height))
    draw_reference_marker(context_panel, metadata, context_crop.size, context_panel.size)
    canvas.paste(context_panel, left_box[:2])

    object_panel, object_alpha = fit_rgba_on_white(source, (panel_width, panel_height))
    canvas.paste(object_panel, right_box[:2])
    editable = Image.eval(object_alpha.filter(ImageFilter.MaxFilter(7)), lambda value: 255 - value)
    mask.paste(editable, right_box[:2])
    output_alpha = Image.new("L", object_panel.size, 255)
    return canvas, mask, right_box, output_alpha


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
) -> None:
    from PIL import ImageDraw

    object_box = object_box_in_context(metadata, (1, 1), context_size)
    scale = min(panel_size[0] / max(1, context_size[0]), panel_size[1] / max(1, context_size[1]))
    offset_x = (panel_size[0] - context_size[0] * scale) / 2.0
    offset_y = (panel_size[1] - context_size[1] * scale) / 2.0
    box = (
        offset_x + object_box[0] * scale,
        offset_y + object_box[1] * scale,
        offset_x + object_box[2] * scale,
        offset_y + object_box[3] * scale,
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
