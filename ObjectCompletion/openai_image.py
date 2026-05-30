from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter


DEFAULT_OPENAI_IMAGE_MODEL = "gpt-5.5"
DEFAULT_OPENAI_EDIT_MODEL = "gpt-image-2"
DEFAULT_OPENAI_IMAGE_QUALITY = os.environ.get("SCENEFORGE_OPENAI_IMAGE_QUALITY", "medium")
DEFAULT_OPENAI_TIMEOUT_SECONDS = float(os.environ.get("SCENEFORGE_OPENAI_TIMEOUT_SECONDS", "180"))
DEFAULT_OPENAI_COMPLETION_WORKERS = int(
    os.environ.get("SCENEFORGE_OPENAI_COMPLETION_WORKERS", os.environ.get("SCENEFORGE_COMPLETION_WORKERS", "2"))
)
DEFAULT_NEGATIVE_PROMPT = (
    "full room, full scene, furniture set, background scene, floor, ground plane, base slab, platform, display stand, "
    "text, watermark, duplicate object, extra object, distorted, low quality"
)


def run_openai_object_completion(
    objects_dir: str | Path,
    *,
    model: str = DEFAULT_OPENAI_IMAGE_MODEL,
    steps: int = 28,
    guidance_scale: float = 6.0,
    strength: float = 1.0,
    canvas_size: int = 1024,
    seed: int = 20260528,
    max_objects: int = 0,
) -> dict[str, Any]:
    root = Path(objects_dir)
    if not root.is_dir():
        return write_manifest(root, [], "missing_objects_dir", backend="openai-image")

    object_dirs = [path for path in sorted(root.iterdir()) if path.is_dir()]
    if not object_dirs:
        return write_manifest(root, [], "no_objects", backend="openai-image")

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for --completion-backend openai-image")

    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("Install the openai package to use --completion-backend openai-image.") from exc

    client = OpenAI()
    records: list[dict[str, Any]] = []
    selected_dirs = object_dirs if max_objects <= 0 else object_dirs[:max_objects]
    worker_count = max(1, min(int(DEFAULT_OPENAI_COMPLETION_WORKERS), len(selected_dirs)))
    print(
        f"Running OpenAI object completion for {len(selected_dirs)} of {len(object_dirs)} objects "
        f"with {model}, quality={DEFAULT_OPENAI_IMAGE_QUALITY}, workers={worker_count}.",
        flush=True,
    )
    if worker_count == 1:
        for index, object_dir in enumerate(selected_dirs, start=1):
            print(f"OpenAI completion {index}/{len(selected_dirs)}: {object_dir.name}", flush=True)
            records.append(
                complete_object_dir(
                    object_dir,
                    client=client,
                    model=model,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    strength=strength,
                    canvas_size=canvas_size,
                    seed=seed,
                    index=index,
                )
            )
            print(f"Wrote {object_dir / 'completed_crop.png'}", flush=True)
    else:
        records_by_index: dict[int, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {}
            for index, object_dir in enumerate(selected_dirs, start=1):
                print(f"OpenAI completion queued {index}/{len(selected_dirs)}: {object_dir.name}", flush=True)
                future = executor.submit(
                    complete_object_dir,
                    object_dir,
                    client=client,
                    model=model,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    strength=strength,
                    canvas_size=canvas_size,
                    seed=seed,
                    index=index,
                )
                futures[future] = (index, object_dir)
            for future in as_completed(futures):
                index, object_dir = futures[future]
                records_by_index[index] = future.result()
                print(f"Wrote {object_dir / 'completed_crop.png'}", flush=True)
        records = [records_by_index[index] for index in sorted(records_by_index)]
    return write_manifest(root, records, "complete", backend="openai-image")


def complete_object_dir(
    object_dir: Path,
    *,
    client,
    model: str,
    steps: int,
    guidance_scale: float,
    strength: float,
    canvas_size: int,
    seed: int,
    index: int,
) -> dict[str, Any]:
    metadata = read_metadata(object_dir / "metadata.json")
    label = str(metadata.get("detector_label") or metadata.get("primitive_label") or "object")
    masked_crop_path = object_dir / "masked_crop.png"
    if not masked_crop_path.is_file():
        return {"object_dir": str(object_dir), "status": "skipped", "reason": "missing_masked_crop"}

    masked_crop = Image.open(masked_crop_path).convert("RGBA")
    context_crop, context_reference_name = load_context_reference(object_dir)
    target_input = render_target_square(masked_crop, canvas_size=canvas_size)
    input_path = object_dir / "completion_openai_input.png"
    reference_path = object_dir / "completion_openai_reference.png"
    target_input.save(input_path)
    if context_crop is not None:
        render_reference_square(context_crop, canvas_size=canvas_size).save(reference_path)
    else:
        reference_path = None

    prompt = build_openai_prompt(label)
    result_image, backend_model, backend_warning = call_openai_image_completion(
        client=client,
        model=model,
        prompt=prompt,
        input_path=input_path,
        reference_path=reference_path,
        canvas_size=canvas_size,
    )
    result_image = ensure_transparent_completed_image(result_image)
    result_image.save(object_dir / "completed_crop.png")

    record = {
        "object_dir": str(object_dir),
        "status": "complete",
        "prompt": prompt,
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        "model": model,
        "backend_model": backend_model,
        "steps": int(steps),
        "guidance_scale": float(guidance_scale),
        "strength": float(strength),
        "canvas_size": int(canvas_size),
        "seed": int(seed),
        "completed_crop": "completed_crop.png",
        "source_crop": "masked_crop.png",
        "context_crop": context_reference_name,
        "openai_input": input_path.name,
        "openai_reference": reference_path.name if reference_path is not None else None,
        "background": "transparent",
        "order_index": index,
    }
    if backend_warning:
        record["backend_warning"] = backend_warning
    (object_dir / "completion_metadata.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return record


def build_openai_prompt(label: str) -> str:
    normalized = label.lower().strip()
    constraints = label_specific_constraints(normalized)
    return (
        f"Repair the target object crop in image 1. The target object is: {label}. "
        "Image 2, if present, is only reference context for shape, perspective, materials, and occlusion reasoning. "
        "Complete missing or cut-out parts of the target object and remove occluders from the target. "
        "Preserve the visible target-object pixels as closely as possible, including camera perspective, lighting, material, scale, and edges. "
        "Do not copy the reference background, other furniture, or reference layout into the final crop. "
        f"{constraints} "
        "Do not add a room, full scene, extra furniture, text, watermark, or new objects. "
        "No floor under the object. Do not add a floor, ground plane, base slab, platform, plinth, shadow catcher, or support surface under the object; keep only geometry that is part of the target object itself. "
        "Return one clean square product-style PNG of the single completed target object on a transparent background. "
        "Keep every non-object background pixel fully transparent."
    )


def label_specific_constraints(label: str) -> str:
    if "table" in label:
        return (
            "For a table, preserve the round/elliptical tabletop silhouette, continue the tabletop as one smooth uninterrupted surface, "
            "keep the rim thickness consistent, continue the wood grain naturally, and complete the pedestal as one vertical cylindrical support. "
            "Do not leave a vase stem, flower stem, black hole, cut-out notch, or extra wooden post on top of the tabletop. "
            "Do not create a rectangular floor sheet, ground plane, base slab, or platform under the table."
        )
    if "chair" in label:
        return (
            "For a chair, preserve the seat, legs, backrest angle, and black material; complete missing legs and rails with matching perspective and thickness."
        )
    if "vase" in label or "flower" in label or "plant" in label:
        return (
            "For a vase or plant, preserve the thin vertical glass/plant structure and complete only the missing continuation of that object."
        )
    return "Complete the hidden shape conservatively with matching material, contour, and perspective."


def write_openai_mask(mask: Image.Image, output_path: Path) -> None:
    edit_alpha = mask.convert("L")
    keep_alpha = Image.eval(edit_alpha, lambda value: 255 - value)
    mask_rgba = Image.new("RGBA", keep_alpha.size, (255, 255, 255, 255))
    mask_rgba.putalpha(keep_alpha)
    mask_rgba.save(output_path)


def call_openai_image_completion(
    *,
    client,
    model: str,
    prompt: str,
    input_path: Path,
    reference_path: Path | None,
    canvas_size: int,
) -> tuple[Image.Image, str, str | None]:
    if model.startswith("gpt-image"):
        print(f"Calling OpenAI Image API edit with {model}.", flush=True)
        return call_image_edit_api(
            client=client,
            model=model,
            prompt=prompt,
            input_path=input_path,
            reference_path=reference_path,
            canvas_size=canvas_size,
        ), model, None

    try:
        print(f"Calling OpenAI Responses image tool with {model} (timeout {DEFAULT_OPENAI_TIMEOUT_SECONDS:.0f}s).", flush=True)
        return call_responses_image_tool(
            client=client,
            model=model,
            prompt=prompt,
            input_path=input_path,
            reference_path=reference_path,
            canvas_size=canvas_size,
        ), model, None
    except Exception as exc:
        if not should_fallback_to_image_edit_api(exc):
            raise
        fallback = DEFAULT_OPENAI_EDIT_MODEL
        print(f"Responses image tool failed; retrying OpenAI Image API edit with {fallback}.", flush=True)
        return (
            call_image_edit_api(
                client=client,
                model=fallback,
                prompt=prompt,
                input_path=input_path,
                reference_path=reference_path,
                canvas_size=canvas_size,
            ),
            fallback,
            f"responses_image_tool_failed_fell_back_to_{fallback}: {exc}",
        )


def call_responses_image_tool(
    *,
    client,
    model: str,
    prompt: str,
    input_path: Path,
    reference_path: Path | None,
    canvas_size: int,
) -> Image.Image:
    image_file_id = create_openai_file(client, input_path)
    reference_file_id = create_openai_file(client, reference_path) if reference_path is not None else None
    try:
        content = [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "file_id": image_file_id},
        ]
        if reference_file_id is not None:
            content.append({"type": "input_image", "file_id": reference_file_id})
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
            tools=[
                {
                    "type": "image_generation",
                    "quality": DEFAULT_OPENAI_IMAGE_QUALITY,
                    "size": f"{canvas_size}x{canvas_size}",
                    "background": "transparent",
                }
            ],
            timeout=DEFAULT_OPENAI_TIMEOUT_SECONDS,
        )
    finally:
        delete_openai_file(client, image_file_id)
        delete_openai_file(client, reference_file_id)
    return decode_image_response(response)


def call_image_edit_api(
    *,
    client,
    model: str,
    prompt: str,
    input_path: Path,
    reference_path: Path | None,
    canvas_size: int,
) -> Image.Image:
    canvas_size = valid_openai_image_size(canvas_size)
    with input_path.open("rb") as image_file:
        image_files = [image_file]
        reference_file = None
        if reference_path is not None:
            reference_file = reference_path.open("rb")
            image_files.append(reference_file)
        print(
            f"Calling OpenAI Image API edit with {model}, quality={DEFAULT_OPENAI_IMAGE_QUALITY}, timeout {DEFAULT_OPENAI_TIMEOUT_SECONDS:.0f}s.",
            flush=True,
        )
        try:
            result = client.images.edit(
                model=model,
                image=image_files,
                prompt=prompt,
                size=f"{canvas_size}x{canvas_size}",
                quality=DEFAULT_OPENAI_IMAGE_QUALITY,
                output_format="png",
                background="transparent",
                timeout=DEFAULT_OPENAI_TIMEOUT_SECONDS,
            )
        finally:
            if reference_file is not None:
                reference_file.close()
    image_base64 = result.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)
    output_path = os.environ.get("SCENEFORGE_OPENAI_LAST_IMAGE")
    if output_path:
        Path(output_path).write_bytes(image_bytes)
    return Image.open(io.BytesIO(image_bytes)).convert("RGBA")


def valid_openai_image_size(canvas_size: int) -> int:
    if int(canvas_size) < 1024:
        return 1024
    return int(canvas_size)


def render_target_square(masked_crop: Image.Image, *, canvas_size: int) -> Image.Image:
    source = masked_crop.convert("RGBA")
    source.thumbnail((int(canvas_size * 0.88), int(canvas_size * 0.88)), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    x = (canvas_size - source.width) // 2
    y = (canvas_size - source.height) // 2
    canvas.paste(source, (x, y), source.getchannel("A"))
    return canvas


def render_reference_square(reference_crop: Image.Image, *, canvas_size: int) -> Image.Image:
    reference = reference_crop.convert("RGB")
    reference.thumbnail((int(canvas_size * 0.94), int(canvas_size * 0.94)), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (canvas_size, canvas_size), (245, 245, 240))
    x = (canvas_size - reference.width) // 2
    y = (canvas_size - reference.height) // 2
    canvas.paste(reference, (x, y))
    return canvas


def should_fallback_to_image_edit_api(exc: BaseException) -> bool:
    message = str(exc).lower()
    fallback_markers = (
        "input_image_mask",
        "image_generation",
        "unsupported",
        "not supported",
        "invalid tool",
        "unknown parameter",
        "timed out",
        "timeout",
        "readtimeout",
    )
    return any(marker in message for marker in fallback_markers)


def create_openai_file(client, path: Path) -> str:
    with path.open("rb") as file_content:
        result = client.files.create(file=file_content, purpose="vision")
    return str(result.id)


def delete_openai_file(client, file_id: str | None) -> None:
    if not file_id:
        return
    try:
        client.files.delete(file_id)
    except Exception:
        pass


def load_context_reference(object_dir: Path) -> tuple[Image.Image | None, str | None]:
    artifacts_dir = object_dir / "artifacts" / "segmentation"
    focus_path = object_dir / "context_focus_crop.png"
    if not focus_path.is_file():
        focus_path = artifacts_dir / "context_focus_crop.png"
    if focus_path.is_file():
        return Image.open(focus_path).convert("RGB"), focus_path.relative_to(object_dir).as_posix()
    context_path = object_dir / "context_crop.png"
    if not context_path.is_file():
        context_path = artifacts_dir / "context_crop.png"
    if not context_path.is_file():
        return None, None
    context = Image.open(context_path).convert("RGB")
    mask_path = object_dir / "context_mask.png"
    if not mask_path.is_file():
        mask_path = artifacts_dir / "context_mask.png"
    if mask_path.is_file():
        focused = dim_context_outside_mask(context, Image.open(mask_path).convert("L"))
        try:
            focus_path.parent.mkdir(parents=True, exist_ok=True)
            focused.save(focus_path)
        except OSError:
            pass
        return focused, focus_path.relative_to(object_dir).as_posix()
    return context, context_path.relative_to(object_dir).as_posix()


def dim_context_outside_mask(context_crop: Image.Image, context_mask: Image.Image) -> Image.Image:
    base = context_crop.convert("RGB")
    mask = context_mask.convert("L").filter(ImageFilter.MaxFilter(25)).filter(ImageFilter.GaussianBlur(4.0))
    gray = base.convert("L").convert("RGB")
    dim = Image.blend(gray, Image.new("RGB", base.size, (20, 20, 20)), 0.58)
    focused = dim.copy()
    focused.paste(base, (0, 0), mask)
    return focused


def read_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def write_manifest(root: Path, records: list[dict[str, Any]], status: str, backend: str = "openai-image") -> dict[str, Any]:
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


def decode_image_response(response) -> Image.Image:
    image_data = []
    for output in getattr(response, "output", []) or []:
        output_type = getattr(output, "type", None)
        if output_type == "image_generation_call":
            result = getattr(output, "result", None)
            if result:
                image_data.append(result)
        if isinstance(output, dict) and output.get("type") == "image_generation_call" and output.get("result"):
            image_data.append(output["result"])
    if not image_data:
        raise RuntimeError("OpenAI image completion returned no image_generation_call result.")

    image_bytes = base64.b64decode(image_data[0])
    output_path = os.environ.get("SCENEFORGE_OPENAI_LAST_IMAGE")
    if output_path:
        Path(output_path).write_bytes(image_bytes)
    return Image.open(io.BytesIO(image_bytes)).convert("RGBA")


def ensure_transparent_completed_image(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    if alpha.getextrema()[0] < 255:
        return rgba
    return remove_neutral_background_alpha(rgba)


def remove_neutral_background_alpha(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    rgb = rgba.convert("RGB")
    background = estimate_border_background(rgb)
    alpha = foreground_alpha_from_background(rgb, background)
    rgba.putalpha(alpha)
    return clear_transparent_rgb(rgba)


def estimate_border_background(image: Image.Image) -> tuple[int, int, int]:
    pixels = image.load()
    width, height = image.size
    samples: list[tuple[int, int, int]] = []
    for x in range(width):
        samples.append(pixels[x, 0])
        samples.append(pixels[x, height - 1])
    for y in range(height):
        samples.append(pixels[0, y])
        samples.append(pixels[width - 1, y])
    channels = list(zip(*samples))
    return tuple(sorted(channel)[len(channel) // 2] for channel in channels)


def foreground_alpha_from_background(image: Image.Image, background: tuple[int, int, int]) -> Image.Image:
    import numpy as np

    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    bg = np.asarray(background, dtype=np.int16).reshape(1, 1, 3)
    distance = np.linalg.norm(rgb - bg, axis=2)
    threshold = min(64.0, max(18.0, float(np.percentile(distance, 82))))
    alpha = (distance > threshold).astype(np.uint8) * 255
    return Image.fromarray(alpha, mode="L")


def clear_transparent_rgb(image: Image.Image) -> Image.Image:
    import numpy as np

    rgba = image.convert("RGBA")
    array = np.asarray(rgba, dtype=np.uint8).copy()
    array[array[..., 3] == 0, :3] = 0
    return Image.fromarray(array, mode="RGBA")
