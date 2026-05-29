from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from PIL import Image

from ObjectCompletion.sdxl_inpaint import (
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_PROMPT_TEMPLATE,
    build_inpaint_canvas,
    compose_completed_object,
    load_context_reference,
    read_metadata,
    save_completion_debug_artifacts,
    write_manifest,
)


def run_flux_object_completion(
    objects_dir: str | Path,
    model_dir: str | Path,
    *,
    device: str | None = "auto",
    steps: int = 28,
    guidance_scale: float = 6.0,
    strength: float = 1.0,
    canvas_size: int = 1024,
    seed: int = 20260528,
    max_objects: int = 0,
    quantization: str = "4bit",
) -> dict[str, Any]:
    root = Path(objects_dir)
    if not root.is_dir():
        return write_manifest(root, [], "missing_objects_dir", backend="flux-fill")

    object_dirs = [path for path in sorted(root.iterdir()) if path.is_dir()]
    if not object_dirs:
        return write_manifest(root, [], "no_objects", backend="flux-fill")

    pipe = None
    torch = None
    current_device: str = str(device) if device is not None else "auto"
    records: list[dict[str, Any]] = []

    def load_with(device_override: str, quantization_override: str):
        _unload_flux_pipeline(pipe, torch)
        _clear_cuda_memory(torch)
        return load_pipeline(model_dir, device_override, quantization=quantization_override)

    try:
        _clear_cuda_memory()
        try:
            pipe, torch, generator_device = load_pipeline(model_dir, device, quantization=quantization)
        except RuntimeError as exc:
            if _is_cuda_oom(exc) and str(device).strip().lower() in {"", "auto", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"}:
                print("FLUX GPU memory exceeded; retrying on CPU with non-quantized model load.")
                pipe, torch, generator_device = load_with("cpu", "none")
            else:
                raise
        except TypeError as exc:
            if _is_quantized_offload_bug(exc) and str(device).strip().lower() in {"", "auto", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"}:
                print("FLUX quantized offload hook failed; retrying on CPU non-quantized.")
                pipe, torch, generator_device = load_with("cpu", "none")
            else:
                raise
        current_device = str(generator_device)
        generator = torch.Generator(device=generator_device).manual_seed(int(seed))
        selected_dirs = object_dirs if max_objects <= 0 else object_dirs[:max_objects]
        for index, object_dir in enumerate(selected_dirs, start=1):
            while True:
                try:
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
                    break
                except RuntimeError as exc:
                    if _is_cuda_oom(exc) and current_device.startswith("cuda"):
                        print("FLUX GPU memory exceeded; switching to CPU completion.")
                        pipe, torch, generator_device = load_with("cpu", "none")
                        generator = torch.Generator(device=generator_device).manual_seed(int(seed))
                        current_device = str(generator_device)
                        continue
                    raise
            _clear_cuda_memory(torch)
        return write_manifest(root, records, "complete", backend="flux-fill")
    finally:
        _unload_flux_pipeline(pipe, torch)
        _clear_cuda_memory(torch)


def load_pipeline(model_dir: str | Path, device: str | None, *, quantization: str = "4bit"):
    try:
        import torch
        from diffusers import BitsAndBytesConfig, FluxFillPipeline, FluxTransformer2DModel
        from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
        from transformers import T5EncoderModel
    except Exception as exc:
        raise RuntimeError(
            "FLUX fill completion requires diffusers with FluxFillPipeline support."
        ) from exc

    requested_device = device if device not in (None, "auto") else ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if str(requested_device).startswith("cuda") else torch.float32
    quantization = quantization if str(requested_device).startswith("cuda") else "none"
    transformer = None
    text_encoder_2 = None
    _clear_cuda_memory(torch)
    is_quantized = quantization in {"4bit", "8bit"}
    if quantization in {"4bit", "8bit"}:
        torch.backends.cuda.matmul.allow_tf32 = True
        diffusers_quantization_config = BitsAndBytesConfig(
            load_in_4bit=quantization == "4bit",
            load_in_8bit=quantization == "8bit",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        transformers_quantization_config = TransformersBitsAndBytesConfig(
            load_in_4bit=quantization == "4bit",
            load_in_8bit=quantization == "8bit",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        transformer = FluxTransformer2DModel.from_pretrained(
            str(model_dir),
            subfolder="transformer",
            torch_dtype=torch_dtype,
            quantization_config=diffusers_quantization_config,
            device_map={"": requested_device},
            local_files_only=True,
        )
        text_encoder_2 = T5EncoderModel.from_pretrained(
            str(model_dir),
            subfolder="text_encoder_2",
            torch_dtype=torch_dtype,
            quantization_config=transformers_quantization_config,
            device_map={"": requested_device},
            local_files_only=True,
        )
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    pipe_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "local_files_only": True,
    }
    if is_quantized:
        pipe_kwargs["transformer"] = transformer
        pipe_kwargs["text_encoder_2"] = text_encoder_2
    try:
        pipe = FluxFillPipeline.from_pretrained(
            str(model_dir),
            low_cpu_mem_usage=True,
            **pipe_kwargs,
        )
    except TypeError:
        pipe = FluxFillPipeline.from_pretrained(str(model_dir), **pipe_kwargs)
    if str(requested_device).startswith("cuda"):
        if hasattr(pipe, "enable_sequential_cpu_offload") and quantization == "none":
            pipe.enable_sequential_cpu_offload()
        if hasattr(pipe, "vae"):
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        if hasattr(pipe, "enable_model_cpu_offload"):
            try:
                pipe.enable_model_cpu_offload()
            except TypeError:
                # Backward compatibility for older diffusers signatures.
                pipe.enable_model_cpu_offload(requested_device)
            except Exception:
                move_unquantized_pipeline_parts_to_device(pipe, requested_device)
        elif quantization == "none":
            pipe = pipe.to(requested_device)
        else:
            move_unquantized_pipeline_parts_to_device(pipe, requested_device)
        generator_device = requested_device
    else:
        pipe = pipe.to("cpu")
        generator_device = "cpu"
    return pipe, torch, generator_device


def _is_cuda_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message and "cuda" in message


def _is_quantized_offload_bug(exc: BaseException) -> bool:
    message = str(exc).lower()
    return ("params4bit" in message and "_is_hf_initialized" in message) or "accelerate" in message


def _clear_cuda_memory(torch: Any | None = None) -> None:
    import gc

    gc.collect()
    if torch is None:
        try:
            import torch as _torch
        except Exception:
            return
        torch = _torch
    if not hasattr(torch, "cuda") or not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "ipc_collect"):
        torch.cuda.ipc_collect()


def _unload_flux_pipeline(pipe, torch: Any | None) -> None:
    if pipe is None:
        return
    try:
        if hasattr(pipe, "to"):
            pipe.to("cpu")
    except Exception:
        pass
    for component_name in ("transformer", "text_encoder", "text_encoder_2", "vae", "scheduler"):
        try:
            component = getattr(pipe, component_name)
        except Exception:
            continue
        if component is None:
            continue
        try:
            if hasattr(component, "to"):
                component.to("cpu")
        except Exception:
            pass
    try:
        if hasattr(pipe, "remove_controlnet") and hasattr(pipe, "controlnet"):
            pipe.remove_controlnet()
    except Exception:
        pass
    try:
        del pipe
    except Exception:
        pass


def move_unquantized_pipeline_parts_to_device(pipe, device: str) -> None:
    for name in ("text_encoder", "vae"):
        component = getattr(pipe, name, None)
        if component is not None and hasattr(component, "to"):
            component.to(device)


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
        image=image,
        mask_image=mask,
        height=canvas_size,
        width=canvas_size,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        generator=generator,
        max_sequence_length=512,
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
        "model": "flux-fill",
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
