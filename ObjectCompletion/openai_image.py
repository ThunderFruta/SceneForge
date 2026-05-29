from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image

from ObjectCompletion.sdxl_inpaint import (
    DEFAULT_NEGATIVE_PROMPT,
    build_inpaint_canvas,
    compose_completed_object,
    load_context_reference,
    read_metadata,
    save_completion_debug_artifacts,
    write_manifest,
)


DEFAULT_OPENAI_IMAGE_MODEL = "gpt-5.5"
DEFAULT_OPENAI_EDIT_MODEL = "gpt-image-2"
DEFAULT_OPENAI_IMAGE_QUALITY = os.environ.get("SCENEFORGE_OPENAI_IMAGE_QUALITY", "medium")
DEFAULT_OPENAI_TIMEOUT_SECONDS = float(os.environ.get("SCENEFORGE_OPENAI_TIMEOUT_SECONDS", "180"))
DEFAULT_OPENAI_COMPLETION_WORKERS = int(
    os.environ.get("SCENEFORGE_OPENAI_COMPLETION_WORKERS", os.environ.get("SCENEFORGE_COMPLETION_WORKERS", "2"))
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
    image, mask, paste_box, output_alpha = build_inpaint_canvas(
        masked_crop=masked_crop,
        context_crop=context_crop,
        metadata=metadata,
        canvas_size=canvas_size,
    )
    save_completion_debug_artifacts(object_dir, image, mask, output_alpha, paste_box)

    input_path = object_dir / "completion_openai_input.png"
    mask_path = object_dir / "completion_openai_mask.png"
    image.save(input_path)
    write_openai_mask(mask, mask_path)

    prompt = build_openai_prompt(label)
    result_image, backend_model, backend_warning = call_openai_image_completion(
        client=client,
        model=model,
        prompt=prompt,
        input_path=input_path,
        mask_path=mask_path,
        canvas_size=canvas_size,
    )
    completed_object, warning = compose_completed_object(
        result=result_image,
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
        "model": model,
        "backend_model": backend_model,
        "steps": int(steps),
        "guidance_scale": float(guidance_scale),
        "strength": float(strength),
        "canvas_size": int(canvas_size),
        "seed": int(seed),
        "paste_box_xyxy": list(paste_box),
        "completed_crop": "completed_crop.png",
        "source_crop": "masked_crop.png",
        "context_crop": context_reference_name,
        "openai_input": input_path.name,
        "openai_mask": mask_path.name,
        "order_index": index,
    }
    if backend_warning:
        record["backend_warning"] = backend_warning
    if warning:
        record["completion_warning"] = warning
    (object_dir / "completion_metadata.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return record


def build_openai_prompt(label: str) -> str:
    normalized = label.lower().strip()
    constraints = label_specific_constraints(normalized)
    return (
        f"Edit only the masked pixels to complete the partially visible {label}. "
        "Treat any vase, plant, flower, chair, person, shadow object, or foreground obstruction as an occluder unless it is the named target object. "
        "Remove occluders from the target and infer the hidden target-object surface behind them. "
        "Preserve every unmasked pixel exactly, including camera perspective, lighting, material, scale, and edges. "
        f"{constraints} "
        "Do not add a room, full scene, extra furniture, text, watermark, or new objects. "
        "Return a clean product-style square crop of the single completed target object."
    )


def label_specific_constraints(label: str) -> str:
    if "table" in label:
        return (
            "For a table, preserve the round/elliptical tabletop silhouette, continue the tabletop as one smooth uninterrupted surface, "
            "keep the rim thickness consistent, continue the wood grain naturally, and complete the pedestal as one vertical cylindrical support. "
            "Do not leave a vase stem, flower stem, black hole, cut-out notch, or extra wooden post on top of the tabletop."
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
    alpha = mask.convert("L")
    mask_rgba = Image.new("RGBA", alpha.size, (255, 255, 255, 0))
    mask_rgba.putalpha(alpha)
    mask_rgba.save(output_path)


def call_openai_image_completion(
    *,
    client,
    model: str,
    prompt: str,
    input_path: Path,
    mask_path: Path,
    canvas_size: int,
) -> tuple[Image.Image, str, str | None]:
    if model.startswith("gpt-image"):
        print(f"Calling OpenAI Image API edit with {model}.", flush=True)
        return call_image_edit_api(
            client=client,
            model=model,
            prompt=prompt,
            input_path=input_path,
            mask_path=mask_path,
            canvas_size=canvas_size,
        ), model, None

    try:
        print(f"Calling OpenAI Responses image tool with {model} (timeout {DEFAULT_OPENAI_TIMEOUT_SECONDS:.0f}s).", flush=True)
        return call_responses_image_tool(
            client=client,
            model=model,
            prompt=prompt,
            input_path=input_path,
            mask_path=mask_path,
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
                mask_path=mask_path,
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
    mask_path: Path,
    canvas_size: int,
) -> Image.Image:
    image_file_id = create_openai_file(client, input_path)
    mask_file_id = create_openai_file(client, mask_path)
    try:
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "file_id": image_file_id},
                    ],
                }
            ],
            tools=[
                {
                    "type": "image_generation",
                    "quality": DEFAULT_OPENAI_IMAGE_QUALITY,
                    "size": f"{canvas_size}x{canvas_size}",
                    "input_image_mask": {"file_id": mask_file_id},
                }
            ],
            timeout=DEFAULT_OPENAI_TIMEOUT_SECONDS,
        )
    finally:
        delete_openai_file(client, image_file_id)
        delete_openai_file(client, mask_file_id)
    return decode_image_response(response)


def call_image_edit_api(
    *,
    client,
    model: str,
    prompt: str,
    input_path: Path,
    mask_path: Path,
    canvas_size: int,
) -> Image.Image:
    with input_path.open("rb") as image_file, mask_path.open("rb") as mask_file:
        print(
            f"Calling OpenAI Image API edit with {model}, quality={DEFAULT_OPENAI_IMAGE_QUALITY}, timeout {DEFAULT_OPENAI_TIMEOUT_SECONDS:.0f}s.",
            flush=True,
        )
        result = client.images.edit(
            model=model,
            image=image_file,
            mask=mask_file,
            prompt=prompt,
            size=f"{canvas_size}x{canvas_size}",
            quality=DEFAULT_OPENAI_IMAGE_QUALITY,
            output_format="png",
            timeout=DEFAULT_OPENAI_TIMEOUT_SECONDS,
        )
    image_base64 = result.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)
    output_path = os.environ.get("SCENEFORGE_OPENAI_LAST_IMAGE")
    if output_path:
        Path(output_path).write_bytes(image_bytes)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


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
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")
