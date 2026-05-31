from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter


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
    context_mode: str = "reference-square",
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
                    context_mode=context_mode,
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
                    context_mode=context_mode,
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
    context_mode: str,
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
    openai_target_input = flatten_transparency_on_white(target_input)
    openai_target_input.save(input_path)
    application_query_path: Path | None = None
    context_mode_requested = normalize_context_mode(context_mode)
    context_mode_effective = context_mode_requested
    context_mode_warning = None
    if context_mode_effective == "application-query" and context_crop is not None:
        application_query_path = object_dir / "application_query.png"
        render_application_query(
            context_crop,
            openai_target_input,
            label=label,
            canvas_size=canvas_size,
        ).save(application_query_path)
        input_path = application_query_path
        reference_path = None
        prompt = build_application_query_prompt(label)
    elif context_crop is not None:
        render_reference_square(context_crop, canvas_size=canvas_size).save(reference_path)
        prompt = build_openai_prompt(label)
    else:
        reference_path = None
        prompt = build_openai_prompt(label)
        if context_mode_effective == "application-query":
            context_mode_warning = "application_query_requested_without_context_reference"
            context_mode_effective = "reference-square"
    result_image, backend_model, backend_warning = call_openai_image_completion(
        client=client,
        model=model,
        prompt=prompt,
        input_path=input_path,
        reference_path=reference_path,
        canvas_size=canvas_size,
    )
    result_image = result_image.convert("RGB")
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
        "completion_context_mode_requested": context_mode_requested,
        "completion_context_mode": context_mode_effective,
        "application_query": application_query_path.name if application_query_path is not None else None,
        "openai_input": input_path.name,
        "openai_reference": reference_path.name if reference_path is not None else None,
        "background": "white",
        "order_index": index,
    }
    if context_mode_warning:
        record["context_mode_warning"] = context_mode_warning
    if backend_warning:
        record["backend_warning"] = backend_warning
    (object_dir / "completion_metadata.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return record


def normalize_context_mode(value: str) -> str:
    if value not in {"reference-square", "application-query"}:
        raise ValueError("--completion-context-mode must be reference-square or application-query")
    return value


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
        "Return one clean square product-style PNG of the single completed target object on a plain white background. "
        "Keep every non-object background pixel pure white."
    )


def build_application_query_prompt(label: str) -> str:
    normalized = label.lower().strip()
    constraints = label_specific_constraints(normalized)
    return (
        f"The input image is an Application-Querying layout for the target object: {label}. "
        "The left panel shows the source scene with the target object emphasized. "
        "The right panel is labeled Extracted Object and contains the current visible object crop. "
        "Use the left panel only for perspective, material, color, lighting, scale, and occlusion cues. "
        "Replace the right-panel object with one complete isolated render of only the marked target object. "
        "Preserve visible target-object pixels and complete missing or hidden parts conservatively. "
        f"{constraints} "
        "Do not include floor, walls, platforms, shadows, other objects, text, labels, frame borders, or the two-panel layout in the output. "
        "Return one clean square product-style PNG of the single completed object on a plain white background. "
        "Keep every non-object background pixel pure white."
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
                    "background": "opaque",
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
        kwargs = {
            "model": model,
            "image": image_files,
            "prompt": prompt,
            "size": f"{canvas_size}x{canvas_size}",
            "quality": DEFAULT_OPENAI_IMAGE_QUALITY,
            "output_format": "png",
            "background": "opaque",
            "timeout": DEFAULT_OPENAI_TIMEOUT_SECONDS,
        }
        try:
            result = client.images.edit(**kwargs)
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


def flatten_transparency_on_white(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    flattened = Image.new("RGB", rgba.size, (255, 255, 255))
    flattened.paste(rgba.convert("RGB"), (0, 0), rgba.getchannel("A"))
    return flattened


def render_reference_square(reference_crop: Image.Image, *, canvas_size: int) -> Image.Image:
    reference = reference_crop.convert("RGB")
    reference.thumbnail((int(canvas_size * 0.94), int(canvas_size * 0.94)), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (canvas_size, canvas_size), (245, 245, 240))
    x = (canvas_size - reference.width) // 2
    y = (canvas_size - reference.height) // 2
    canvas.paste(reference, (x, y))
    return canvas


def render_application_query(
    context_crop: Image.Image,
    target_input: Image.Image,
    *,
    label: str,
    canvas_size: int,
) -> Image.Image:
    panel_width = int(canvas_size)
    panel_height = int(canvas_size)
    gutter = max(12, panel_width // 48)
    header_height = max(48, panel_height // 16)
    canvas = Image.new("RGB", (panel_width * 2 + gutter, panel_height), (232, 230, 224))
    draw = ImageDraw.Draw(canvas)

    left = render_reference_square(context_crop, canvas_size=panel_width)
    right = Image.new("RGB", (panel_width, panel_height), (248, 247, 243))
    right_target = target_input.convert("RGBA")
    card_margin = max(24, panel_width // 24)
    target_max = panel_width - card_margin * 2
    right_target.thumbnail((target_max, target_max - header_height), Image.Resampling.LANCZOS)
    target_x = (panel_width - right_target.width) // 2
    target_y = header_height + max(8, (panel_height - header_height - right_target.height) // 2)
    right.paste(right_target.convert("RGB"), (target_x, target_y), right_target.getchannel("A"))

    canvas.paste(left, (0, 0))
    right_x = panel_width + gutter
    canvas.paste(right, (right_x, 0))

    border_width = max(3, panel_width // 180)
    draw.rectangle((0, 0, panel_width - 1, panel_height - 1), outline=(34, 91, 99), width=border_width)
    draw.rectangle((right_x, 0, right_x + panel_width - 1, panel_height - 1), outline=(74, 69, 59), width=border_width)
    draw.rectangle((right_x, 0, right_x + panel_width - 1, header_height), fill=(36, 34, 30))
    draw.text((right_x + card_margin, max(10, header_height // 4)), "Extracted Object", fill=(255, 252, 240))
    draw.text((card_margin, max(10, header_height // 4)), f"Source context: {label}", fill=(12, 43, 48))
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
