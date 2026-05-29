from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from ObjectCompletion.sdxl_inpaint import (
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_PROMPT_TEMPLATE,
    build_inpaint_canvas,
    object_only_alpha,
    read_metadata,
    write_manifest,
)


def run_flux_object_completion(
    objects_dir: str | Path,
    model_dir: str | Path,
    *,
    device: str | None = "auto",
    steps: int = 28,
    guidance_scale: float = 30.0,
    strength: float = 1.0,
    canvas_size: int = 1024,
    seed: int = 20260528,
    max_objects: int = 16,
) -> dict[str, Any]:
    root = Path(objects_dir)
    if not root.is_dir():
        return write_manifest(root, [], "missing_objects_dir", backend="flux-fill")

    object_dirs = [path for path in sorted(root.iterdir()) if path.is_dir()]
    if not object_dirs:
        return write_manifest(root, [], "no_objects", backend="flux-fill")

    pipe, torch, generator_device = load_pipeline(model_dir, device)
    generator = torch.Generator(device=generator_device).manual_seed(int(seed))
    records: list[dict[str, Any]] = []
    for index, object_dir in enumerate(object_dirs[:max_objects], start=1):
        records.append(
            complete_object_dir(
                object_dir,
                pipe=pipe,
                generator=generator,
                steps=steps,
                guidance_scale=guidance_scale,
                strength=strength,
                canvas_size=canvas_size,
                index=index,
            )
        )
    return write_manifest(root, records, "complete", backend="flux-fill")


def load_pipeline(model_dir: str | Path, device: str | None):
    try:
        import torch
        from diffusers import FluxFillPipeline
    except Exception as exc:
        raise RuntimeError(
            "FLUX fill completion requires diffusers with FluxFillPipeline support."
        ) from exc

    requested_device = device if device not in (None, "auto") else ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.bfloat16 if str(requested_device).startswith("cuda") else torch.float32
    pipe = FluxFillPipeline.from_pretrained(
        str(model_dir),
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    if str(requested_device).startswith("cuda"):
        if hasattr(pipe, "vae"):
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
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
        image=image,
        mask_image=mask,
        height=canvas_size,
        width=canvas_size,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        generator=generator,
        max_sequence_length=512,
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
        "model": "flux-fill",
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
