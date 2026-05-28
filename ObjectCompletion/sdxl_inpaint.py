from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


DEFAULT_PROMPT_TEMPLATE = (
    "complete the full object, a {label}, same material, same perspective, "
    "single isolated object, no extra objects"
)
DEFAULT_NEGATIVE_PROMPT = "text, watermark, duplicate object, extra object, distorted, low quality"


def run_sdxl_object_completion(
    objects_dir: str | Path,
    model_dir: str | Path,
    *,
    device: str | None = "auto",
    steps: int = 24,
    guidance_scale: float = 6.5,
    strength: float = 0.92,
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
    image, mask, paste_box = build_inpaint_canvas(masked_crop, canvas_size=canvas_size)
    image.save(object_dir / "completion_input.png")
    mask.save(object_dir / "completion_mask.png")

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
    result.save(object_dir / "completed_canvas.png")
    completed_crop = result.crop(paste_box)
    completed_crop.save(object_dir / "completed_crop.png")

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
        "completed_canvas": "completed_canvas.png",
        "completed_crop": "completed_crop.png",
        "completion_input": "completion_input.png",
        "completion_mask": "completion_mask.png",
        "order_index": index,
    }
    (object_dir / "completion_metadata.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return record


def build_inpaint_canvas(masked_crop: Image.Image, *, canvas_size: int) -> tuple[Image.Image, Image.Image, tuple[int, int, int, int]]:
    crop = masked_crop.convert("RGBA")
    crop.thumbnail((int(canvas_size * 0.78), int(canvas_size * 0.78)), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (canvas_size, canvas_size), (245, 245, 240))
    alpha = crop.getchannel("A")
    x = (canvas_size - crop.width) // 2
    y = (canvas_size - crop.height) // 2
    visible_rgb = Image.new("RGB", crop.size, (245, 245, 240))
    visible_rgb.paste(crop.convert("RGB"), mask=alpha)
    canvas.paste(visible_rgb, (x, y))

    mask = Image.new("L", (canvas_size, canvas_size), 255)
    protected = alpha.filter(ImageFilter.MaxFilter(5))
    mask.paste(Image.eval(protected, lambda value: 255 - value), (x, y))
    return canvas, mask, (x, y, x + crop.width, y + crop.height)


def read_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def write_manifest(root: Path, records: list[dict[str, Any]], status: str) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "backend": "sdxl-inpaint",
        "status": status,
        "object_count": len(records),
        "objects": records,
    }
    (root / "completion_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
