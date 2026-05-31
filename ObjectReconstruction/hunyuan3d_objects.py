from __future__ import annotations

import json
import gc
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, MutableMapping

from PIL import Image

from ObjectReconstruction.triposr_objects import (
    clean_reconstruction_outputs,
    prepare_reconstruction_input,
    reconstruction_artifacts_dir,
    relative_to_object_dir,
    select_source_image,
)

DEFAULT_TEXTURE_PROMPT = "high quality object only, no floor, no ground plane, no base slab, no platform"


def run_hunyuan3d_object_reconstruction(
    objects_dir: str | Path,
    *,
    model: str = "tencent/Hunyuan3D-2.1",
    device: str | None = "auto",
    source: str = "completed",
    max_objects: int = 0,
    with_texture: bool = False,
    texture_resolution: int = 512,
    texture_views: int = 6,
    texture_use_remesh: bool = True,
    texture_prompt: str | None = None,
    texture_reference_mode: str = "original",
    texture_matte_backend: str = "auto",
    texture_matte_model_dir: str | Path = "Models/Segmentation/BRIA/RMBG-2.0",
    completed_mask_backend: str = "auto",
    completed_mask_sam3_repo_dir: str | Path | None = None,
    completed_mask_sam3_model_dir: str | Path | None = None,
    completed_mask_prompt: str | None = None,
    completed_mask_score_threshold: float = 0.25,
) -> dict[str, Any]:
    validate_hunyuan_texture_options(texture_resolution, texture_views)
    root = Path(objects_dir)
    if not root.is_dir():
        return write_manifest(root, [], "missing_objects_dir", model=model, device=device, source=source)

    object_dirs = [path for path in sorted(root.iterdir()) if path.is_dir()]
    selected_dirs = object_dirs if max_objects <= 0 else object_dirs[:max_objects]
    if not selected_dirs:
        return write_manifest(root, [], "no_objects", model=model, device=device, source=source)
    clean_reconstruction_outputs(root, selected_dirs, backend="hunyuan3d")

    torch = import_torch()
    resolved_device = resolve_device(torch=torch, device=device)
    completed_mask_segmenter = build_completed_mask_segmenter(
        backend=completed_mask_backend,
        sam3_repo_dir=completed_mask_sam3_repo_dir,
        sam3_model_dir=completed_mask_sam3_model_dir,
        device=resolved_device,
        score_threshold=completed_mask_score_threshold,
    )
    prepare_completed_masks(
        selected_dirs,
        source=source,
        completed_mask_backend=completed_mask_backend,
        completed_mask_segmenter=completed_mask_segmenter,
        completed_mask_prompt=completed_mask_prompt,
    )
    del completed_mask_segmenter
    release_torch_memory(torch, resolved_device)

    pipeline = load_pipeline(model=model, device=resolved_device)
    records: list[dict[str, Any]] = []
    print(f"Running Hunyuan3D object reconstruction for {len(selected_dirs)} of {len(object_dirs)} objects.", flush=True)
    for index, object_dir in enumerate(selected_dirs, start=1):
        print(f"Hunyuan3D reconstruction {index}/{len(selected_dirs)}: {object_dir.name}", flush=True)
        records.append(
            reconstruct_object_dir(
                object_dir,
                pipeline=pipeline,
                source=source,
                completed_mask_backend=completed_mask_backend,
                completed_mask_segmenter=None,
                completed_mask_prompt=completed_mask_prompt,
                order_index=index,
            )
        )
    del pipeline
    release_torch_memory(torch, resolved_device)

    if with_texture:
        texture_records(
            root,
            records,
            device=resolved_device,
            resolution=texture_resolution,
            views=texture_views,
            use_remesh=texture_use_remesh,
            prompt=texture_prompt,
            reference_mode=texture_reference_mode,
            matte_backend=texture_matte_backend,
            matte_model_dir=texture_matte_model_dir,
        )
    return write_manifest(
        root,
        records,
        "complete",
        model=model,
        device=str(resolved_device),
        source=source,
        with_texture=with_texture,
        texture_resolution=texture_resolution,
        texture_views=texture_views,
        texture_use_remesh=texture_use_remesh,
        texture_prompt=texture_prompt,
        texture_reference_mode=texture_reference_mode,
        texture_matte_backend=texture_matte_backend,
        texture_matte_model_dir=str(texture_matte_model_dir),
        completed_mask_backend=completed_mask_backend,
    )


def import_torch():
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("Hunyuan3D reconstruction requires torch.") from exc
    return torch


def resolve_device(*, torch, device: str | None) -> str:
    resolved_device = device if device not in (None, "auto") else ("cuda:0" if torch.cuda.is_available() else "cpu")
    if str(resolved_device).isdigit():
        resolved_device = f"cuda:{resolved_device}"
    if str(resolved_device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Hunyuan3D was requested on CUDA, but torch.cuda.is_available() is false.")
    return str(resolved_device)


def load_pipeline(*, model: str, device: str | None):
    add_local_hunyuan3d_paths()
    apply_hunyuan3d_hf_cache_env(os.environ)
    os.environ.setdefault("HY3DGEN_MODELS", str(Path("Models/Mesh/Hunyuan3D/hf").resolve()))
    try:
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    except Exception as exc:
        raise RuntimeError(
            "Hunyuan3D reconstruction requires the Hunyuan3D-2.1 repo at Models/Mesh/Hunyuan3D/repo. "
            "Expected import: from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline."
        ) from exc

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model, device=device)
    if hasattr(pipeline, "to"):
        pipeline.to(device)
    return pipeline


def add_local_hunyuan3d_paths() -> None:
    root = Path("Models/Mesh/Hunyuan3D/repo").resolve()
    for path in (root, root / "hy3dshape", root / "hy3dpaint"):
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def build_completed_mask_segmenter(
    *,
    backend: str,
    sam3_repo_dir: str | Path | None,
    sam3_model_dir: str | Path | None,
    device: str,
    score_threshold: float,
):
    if backend not in {"auto", "sam3"}:
        return None
    repo_dir = Path(sam3_repo_dir) if sam3_repo_dir is not None else Path("Models/OpenVocabulary/SAM3/repo")
    model_dir = Path(sam3_model_dir) if sam3_model_dir is not None else Path("Models/OpenVocabulary/SAM3/hf")
    if backend == "auto" and (not repo_dir.is_dir() or not model_dir.is_dir()):
        return None

    from Segmentation.sam3_segmenter import Sam3Segmenter

    return Sam3Segmenter(
        repo_dir=repo_dir,
        model_dir=model_dir,
        text_prompt="foreground object .",
        score_threshold=score_threshold,
        device=device,
    )


def prepare_completed_masks(
    object_dirs: list[Path],
    *,
    source: str,
    completed_mask_backend: str,
    completed_mask_segmenter,
    completed_mask_prompt: str | None,
) -> None:
    if completed_mask_backend == "original-alpha":
        return
    for object_dir in object_dirs:
        source_path, source_kind = select_source_image(object_dir, source)
        if source_path is None or source_kind != "completed":
            continue
        object_prompt = completed_mask_prompt or completed_mask_prompt_for_object(object_dir)
        try:
            prepare_reconstruction_input(
                source_path,
                object_dir,
                completed_mask_backend=completed_mask_backend,
                completed_mask_segmenter=completed_mask_segmenter,
                completed_mask_prompt=object_prompt,
            )
        except Exception:
            if completed_mask_backend != "auto":
                raise
            prepare_reconstruction_input(
                source_path,
                object_dir,
                completed_mask_backend="foreground",
                completed_mask_segmenter=None,
                completed_mask_prompt=object_prompt,
            )


def reconstruct_object_dir(
    object_dir: Path,
    *,
    pipeline,
    source: str,
    completed_mask_backend: str,
    completed_mask_segmenter,
    completed_mask_prompt: str | None,
    order_index: int,
) -> dict[str, Any]:
    source_path, source_kind = select_source_image(object_dir, source)
    if source_path is None:
        return {
            "object_dir": str(object_dir),
            "status": "skipped",
            "reason": "missing_source_image",
            "order_index": order_index,
        }

    artifacts_dir = reconstruction_artifacts_dir(object_dir)
    input_path = artifacts_dir / "hunyuan3d_input.png"
    mask_path = artifacts_dir / "hunyuan3d_mask.png"
    obj_path = object_dir / "hunyuan3d_mesh.obj"
    glb_path = object_dir / "hunyuan3d_mesh.glb"
    object_prompt = completed_mask_prompt or completed_mask_prompt_for_object(object_dir)
    prepared = prepare_reconstruction_input(
        source_path,
        object_dir,
        completed_mask_backend=completed_mask_backend,
        completed_mask_segmenter=completed_mask_segmenter,
        completed_mask_prompt=object_prompt,
    )
    prepared_image = prepared.image
    prepared_mask = prepared.mask
    prepared_image.save(input_path)
    prepared_mask.save(mask_path)

    try:
        image = apply_mask_to_input(prepared_image, prepared_mask)
        result = pipeline(image=image)
        mesh = result[0] if isinstance(result, (list, tuple)) else result
        export_mesh(mesh, obj_path)
        export_mesh(mesh, glb_path)
        status = "ok"
        reason = None
    except Exception as exc:
        status = "failed"
        reason = str(exc)

    record = {
        "object_dir": str(object_dir),
        "status": status,
        "reason": reason,
        "source": source_kind,
        "source_image": source_path.name,
        "mask_source": prepared.mask_source,
        "completed_mask": prepared.completed_mask_path,
        "completed_mask_prompt": object_prompt,
        "hunyuan3d_input": relative_to_object_dir(object_dir, input_path),
        "hunyuan3d_mask": relative_to_object_dir(object_dir, mask_path),
        "mesh": obj_path.name if status == "ok" else None,
        "glb": glb_path.name if status == "ok" else None,
        "order_index": order_index,
    }
    write_object_metadata(object_dir, record)
    return record


def completed_mask_prompt_for_object(object_dir: Path) -> str:
    metadata_path = object_dir / "metadata.json"
    label = "foreground object"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            label = str(metadata.get("detector_label") or label)
        except Exception:
            pass
    return f"{label} . foreground object . object only . no floor . no ground plane . no base slab ."


def texture_records(
    root: Path,
    records: list[dict[str, Any]],
    *,
    device: str,
    resolution: int,
    views: int,
    use_remesh: bool,
    prompt: str | None,
    reference_mode: str,
    matte_backend: str,
    matte_model_dir: str | Path,
) -> None:
    ok_records = [record for record in records if record.get("status") == "ok"]
    if not ok_records:
        return
    if not device.startswith("cuda"):
        raise RuntimeError("Hunyuan3D paint is only enabled for CUDA in SceneForge right now.")
    print(f"Running Hunyuan3D paint for {len(ok_records)} reconstructed objects.", flush=True)
    for index, record in enumerate(ok_records, start=1):
        object_dir = root / Path(record["object_dir"]).name
        print(f"Hunyuan3D paint {index}/{len(ok_records)}: {object_dir.name}", flush=True)
        texture_object_dir_in_fresh_process(
            object_dir,
            record,
            device=device,
            resolution=resolution,
            views=views,
            use_remesh=use_remesh,
            prompt=prompt,
            reference_mode=reference_mode,
            matte_backend=matte_backend,
            matte_model_dir=matte_model_dir,
        )


def texture_object_dir_in_fresh_process(
    object_dir: Path,
    record: dict[str, Any],
    *,
    device: str,
    resolution: int,
    views: int,
    use_remesh: bool,
    prompt: str | None,
    reference_mode: str,
    matte_backend: str,
    matte_model_dir: str | Path,
) -> None:
    command = [
        sys.executable,
        "-m",
        "ObjectReconstruction.hunyuan3d_objects",
        "--texture-one",
        str(object_dir),
        "--device",
        device,
        "--resolution",
        str(resolution),
        "--views",
        str(views),
        "--prompt",
        prompt or DEFAULT_TEXTURE_PROMPT,
        "--reference-mode",
        reference_mode,
        "--matte-backend",
        matte_backend,
        "--matte-model-dir",
        str(matte_model_dir),
    ]
    command.append("--use-remesh" if use_remesh else "--no-remesh")
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    apply_hunyuan3d_hf_cache_env(env)
    result = subprocess.run(command, cwd=Path.cwd(), env=env, check=False)
    metadata_path = object_dir / "hunyuan3d_metadata.json"
    if metadata_path.is_file():
        try:
            updated_record = json.loads(metadata_path.read_text(encoding="utf-8"))
            record.update(updated_record)
            return
        except Exception:
            pass
    if result.returncode != 0:
        record.update(
            {
                "texture_status": "failed",
                "texture_reason": f"hunyuan3d_paint_process_failed_exit_{result.returncode}",
                "textured_mesh": None,
                "textured_glb": None,
            }
        )
        write_object_metadata(object_dir, record)


def load_paint_pipeline(*, device: str, resolution: int, views: int, prompt: str | None = None):
    validate_hunyuan_texture_options(resolution, views)
    add_local_hunyuan3d_paths()
    paint_root = Path("Models/Mesh/Hunyuan3D/repo/hy3dpaint").resolve()
    realesrgan_path = paint_root / "ckpt" / "RealESRGAN_x4plus.pth"
    if not realesrgan_path.is_file():
        raise RuntimeError(
            "Hunyuan3D paint needs RealESRGAN_x4plus.pth at "
            "Models/Mesh/Hunyuan3D/repo/hy3dpaint/ckpt/RealESRGAN_x4plus.pth."
        )
    apply_hunyuan3d_hf_cache_env(os.environ)
    try:
        from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline
    except Exception as exc:
        raise RuntimeError(
            "Hunyuan3D paint import failed. It needs the hy3dpaint dependencies and compiled rasterizer/renderer extensions."
        ) from exc

    install_torchvision_functional_tensor_shim()
    config = Hunyuan3DPaintConfig(max_num_view=views, resolution=resolution)
    config.device = device
    config.image_caption = prompt or DEFAULT_TEXTURE_PROMPT
    config.realesrgan_ckpt_path = str(realesrgan_path)
    config.multiview_cfg_path = str(paint_root / "cfgs" / "hunyuan-paint-pbr.yaml")
    config.custom_pipeline = str(paint_root / "hunyuanpaintpbr")
    return Hunyuan3DPaintPipeline(config)


def validate_hunyuan_texture_options(resolution: int, views: int) -> None:
    if resolution not in {512, 768}:
        raise ValueError("Hunyuan3D paint texture resolution must be 512 or 768.")
    if not 6 <= views <= 12:
        raise ValueError("Hunyuan3D paint texture views must be between 6 and 12.")


def apply_hunyuan3d_hf_cache_env(env: MutableMapping[str, str]) -> None:
    root = Path("Models/Mesh/Hunyuan3D").resolve()
    hub_cache = root / "hf-cache"
    env["HF_HUB_CACHE"] = str(hub_cache)
    env["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    env["HF_MODULES_CACHE"] = str(root / "diffusers-modules")
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"


def install_torchvision_functional_tensor_shim() -> None:
    module_name = "torchvision.transforms.functional_tensor"
    if module_name in sys.modules:
        return
    try:
        from torchvision.transforms import functional as functional
    except Exception:
        return
    import types

    shim = types.ModuleType(module_name)
    shim.rgb_to_grayscale = functional.rgb_to_grayscale
    sys.modules[module_name] = shim


def texture_object_dir(
    object_dir: Path,
    record: dict[str, Any],
    *,
    paint_pipeline,
    use_remesh: bool,
    prompt: str | None = None,
    reference_mode: str = "original",
    matte_backend: str = "auto",
    matte_model_dir: str | Path = "Models/Segmentation/BRIA/RMBG-2.0",
) -> None:
    mesh_path = object_dir / "hunyuan3d_mesh.glb"
    if not mesh_path.is_file():
        mesh_path = object_dir / "hunyuan3d_mesh.obj"
    source_obj_path = object_dir / "hunyuan3d_mesh.obj"
    artifacts_dir = reconstruction_artifacts_dir(object_dir)
    texture_artifacts_dir = object_dir / "artifacts" / "textures"
    texture_artifacts_dir.mkdir(parents=True, exist_ok=True)
    input_path = object_artifact_path(object_dir, "hunyuan3d_input.png")
    mask_path = object_artifact_path(object_dir, "hunyuan3d_mask.png")
    paint_input_path = artifacts_dir / "hunyuan3d_paint_input.png"
    output_obj = texture_artifacts_dir / "hunyuan3d_textured.obj"
    output_glb = object_dir / "hunyuan3d_textured.glb"
    try:
        if not use_remesh:
            source_face_count = count_obj_faces(source_obj_path)
            if source_face_count is not None and source_face_count > 120_000:
                raise RuntimeError(
                    "Skipping no-remesh Hunyuan3D paint because the source mesh has "
                    f"{source_face_count} faces. Use remesh, or decimate below 120000 faces before no-remesh texture paint."
                )
        if reference_mode == "original":
            paint_reference_path = input_path
            paint_matte_source = "original_input"
        elif reference_mode == "masked-crop":
            paint_reference_path, paint_matte_source = prepare_paint_reference_image(
                input_path,
                mask_path,
                paint_input_path,
                matte_backend=matte_backend,
                matte_model_dir=matte_model_dir,
                device=str(getattr(paint_pipeline.config, "device", "cpu")),
            )
        else:
            raise ValueError(f"Unsupported texture reference mode: {reference_mode}")
        paint_pipeline(
            mesh_path=str(mesh_path),
            image_path=str(paint_reference_path),
            output_mesh_path=str(output_obj),
            use_remesh=use_remesh,
            save_glb=True,
        )
        generated_glb = output_obj.with_suffix(".glb")
        if generated_glb.is_file():
            if output_glb.exists():
                output_glb.unlink()
            shutil.move(str(generated_glb), output_glb)
        elif output_obj.is_file():
            export_textured_obj_to_glb(output_obj, output_glb)
        remesh_path = object_dir / "white_mesh_remesh.obj"
        if remesh_path.is_file():
            remesh_target = artifacts_dir / remesh_path.name
            if remesh_target.exists():
                remesh_target.unlink()
            shutil.move(str(remesh_path), remesh_target)
        support_cleanup = postprocess_hunyuan_textured_glb(object_dir, record, output_obj, output_glb)
        record.update(
            {
                "texture_status": "ok",
                "texture_reason": None,
                "paint_input": relative_to_object_dir(object_dir, paint_input_path),
                "paint_matte_source": paint_matte_source,
                "texture_reference_mode": reference_mode,
                "texture_prompt": prompt or getattr(paint_pipeline.config, "image_caption", None),
                "texture_use_remesh": use_remesh,
                "textured_mesh": relative_to_object_dir(object_dir, output_obj),
                "textured_glb": output_glb.name if output_glb.is_file() else None,
                "support_sheet_cleanup": support_cleanup,
            }
        )
    except Exception as exc:
        record.update(
            {
                "texture_status": "failed",
                "texture_reason": str(exc),
                "textured_mesh": None,
                "textured_glb": None,
            }
        )
    write_object_metadata(object_dir, record)


def export_textured_obj_to_glb(output_obj: Path, output_glb: Path) -> None:
    try:
        import trimesh
    except Exception as exc:
        raise RuntimeError("Converting Hunyuan3D textured OBJ to GLB requires trimesh.") from exc
    loaded = trimesh.load(output_obj, force="scene", process=False)
    output_glb.parent.mkdir(parents=True, exist_ok=True)
    loaded.export(output_glb)


def postprocess_hunyuan_textured_glb(
    object_dir: Path,
    record: dict[str, Any],
    output_obj: Path,
    output_glb: Path,
) -> dict[str, Any]:
    if not should_remove_hunyuan_support_sheet(object_dir, record):
        return {"status": "skipped", "reason": "not_table_like"}
    if not output_obj.is_file() and not output_glb.is_file():
        return {"status": "skipped", "reason": "missing_textured_mesh"}
    source_path = output_glb if output_glb.is_file() else output_obj
    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False, dir=str(object_dir)) as handle:
        temp_output = Path(handle.name)
    temp_output.unlink(missing_ok=True)
    command = [
        "blender",
        "--background",
        "--python",
        str(Path("Tools/Scripts/remove_hunyuan_support_sheets.py")),
        "--",
        str(source_path),
        str(temp_output),
        "--sheet-axis",
        "z",
        "--sheet-side",
        "min",
        "--sheet-band-ratio",
        "0.12",
        "--keep-radius-ratio",
        "0.01",
    ]
    result = subprocess.run(command, cwd=Path.cwd(), text=True, capture_output=True, check=False)
    if result.returncode != 0 or not temp_output.is_file():
        if temp_output.exists():
            temp_output.unlink()
        return {
            "status": "failed",
            "reason": f"support_sheet_cleanup_exit_{result.returncode}",
            "stderr_tail": result.stderr[-800:],
        }
    if output_glb.exists():
        output_glb.unlink()
    shutil.move(str(temp_output), output_glb)
    return {
        "status": "applied",
        "method": "remove_hunyuan_support_sheets",
        "sheet_axis": "z",
        "sheet_side": "min",
        "sheet_band_ratio": 0.12,
        "keep_radius_ratio": 0.01,
        "removed_faces": parse_removed_faces(result.stdout),
    }


def should_remove_hunyuan_support_sheet(object_dir: Path, record: dict[str, Any]) -> bool:
    mesh_quality = record.get("mesh_quality")
    if isinstance(mesh_quality, dict) and mesh_quality.get("has_large_support_sheet") is True:
        return True
    label_parts = [object_dir.name]
    metadata_path = object_dir / "metadata.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            label_parts.extend(str(metadata.get(key) or "") for key in ("detector_label", "primitive_label"))
        except Exception:
            pass
    label_parts.extend(str(record.get(key) or "") for key in ("detector_label", "primitive_label", "completed_mask_prompt"))
    label = " ".join(label_parts).lower()
    base_prone_labels = (
        "table",
        "chair",
        "stool",
        "bench",
        "sofa",
        "couch",
        "cabinet",
        "shelf",
        "desk",
        "dresser",
        "nightstand",
    )
    return any(part in label for part in base_prone_labels)


def parse_removed_faces(output: str) -> int | None:
    marker = '"removed_faces":'
    if marker not in output:
        return None
    tail = output.rsplit(marker, 1)[1].lstrip()
    digits = []
    for character in tail:
        if character.isdigit():
            digits.append(character)
        elif digits:
            break
    return int("".join(digits)) if digits else None


def object_artifact_path(object_dir: Path, name: str) -> Path:
    root_path = object_dir / name
    if root_path.exists():
        return root_path
    artifact_path = object_dir / "artifacts" / "reconstruction" / name
    return artifact_path if artifact_path.exists() else root_path


def count_obj_faces(obj_path: Path) -> int | None:
    if not obj_path.is_file():
        return None
    count = 0
    with obj_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("f "):
                count += 1
    return count


def prepare_paint_reference_image(
    input_path: Path,
    mask_path: Path,
    output_path: Path,
    *,
    size: int = 1024,
    matte_backend: str = "auto",
    matte_model_dir: str | Path = "Models/Segmentation/BRIA/RMBG-2.0",
    device: str = "cpu",
) -> tuple[Path, str]:
    if not input_path.is_file() or not mask_path.is_file():
        return input_path, "input"
    image = Image.open(input_path).convert("RGBA")
    mask, matte_source = paint_matte_for_image(
        image.convert("RGB"),
        mask_path,
        backend=matte_backend,
        model_dir=matte_model_dir,
        device=device,
    )
    bbox = padded_bbox(mask.getbbox(), image.size, padding_ratio=0.18)
    if bbox is None:
        return input_path, "input"

    cropped_image = image.crop(bbox)
    cropped_mask = mask.crop(bbox)
    cropped_image.putalpha(cropped_mask)
    cropped_image.thumbnail((int(size * 0.9), int(size * 0.9)), Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    x = (size - cropped_image.width) // 2
    y = (size - cropped_image.height) // 2
    canvas.paste(cropped_image, (x, y), cropped_image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output_path)
    return output_path, matte_source


def paint_matte_for_image(
    image: Image.Image,
    mask_path: Path,
    *,
    backend: str,
    model_dir: str | Path,
    device: str,
) -> tuple[Image.Image, str]:
    if backend in {"auto", "bria-rmbg"}:
        model_path = Path(model_dir)
        if model_path.is_dir():
            try:
                matte = bria_rmbg_mask(image, model_path, device=device)
                return mask_to_binary(matte.resize(image.size, Image.Resampling.LANCZOS)), "bria-rmbg"
            except Exception:
                if backend == "bria-rmbg":
                    raise
        elif backend == "bria-rmbg":
            raise RuntimeError(f"BRIA RMBG model directory does not exist: {model_path}")
    return mask_to_binary(Image.open(mask_path).convert("L").resize(image.size, Image.Resampling.LANCZOS)), "hunyuan3d_mask"


def bria_rmbg_mask(image: Image.Image, model_dir: Path, *, device: str) -> Image.Image:
    try:
        from transformers import pipeline
    except Exception as exc:
        raise RuntimeError("BRIA RMBG matte generation requires transformers.") from exc
    device_arg = 0 if str(device).startswith("cuda") else -1
    segmenter = pipeline(
        "image-segmentation",
        model=str(model_dir),
        trust_remote_code=True,
        device=device_arg,
    )
    result = segmenter(image.convert("RGB"))
    if isinstance(result, dict):
        result = [result]
    if not result:
        raise RuntimeError("BRIA RMBG returned no segmentation result.")
    mask = result[0].get("mask") if isinstance(result[0], dict) else None
    if mask is None:
        raise RuntimeError("BRIA RMBG result did not include a mask.")
    return mask.convert("L")


def mask_to_binary(mask: Image.Image) -> Image.Image:
    return mask.point(lambda value: 255 if value > 24 else 0)


def padded_bbox(bbox: tuple[int, int, int, int] | None, size: tuple[int, int], *, padding_ratio: float) -> tuple[int, int, int, int] | None:
    if bbox is None:
        return None
    left, top, right, bottom = bbox
    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        return None
    padding = int(max(width, height) * padding_ratio)
    return (
        max(0, left - padding),
        max(0, top - padding),
        min(size[0], right + padding),
        min(size[1], bottom + padding),
    )


def write_object_metadata(object_dir: Path, record: dict[str, Any]) -> None:
    (object_dir / "hunyuan3d_metadata.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def release_torch_memory(torch, device: str) -> None:
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def apply_mask_to_input(image: Image.Image, mask: Image.Image) -> Image.Image:
    rgb = image.convert("RGB")
    alpha = mask.convert("L").resize(rgb.size)
    rgba = rgb.convert("RGBA")
    rgba.putalpha(alpha)
    return rgba


def export_mesh(mesh, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(mesh, "export"):
        mesh.export(output_path)
        return
    raise RuntimeError(f"Hunyuan3D returned an unsupported mesh object: {type(mesh)!r}")


def write_manifest(
    objects_dir: Path,
    records: list[dict[str, Any]],
    status: str,
    *,
    model: str,
    device: str | None,
    source: str,
    with_texture: bool = False,
    texture_resolution: int | None = None,
    texture_views: int | None = None,
    texture_use_remesh: bool = True,
    texture_prompt: str | None = None,
    texture_reference_mode: str = "original",
    texture_matte_backend: str = "auto",
    texture_matte_model_dir: str = "Models/Segmentation/BRIA/RMBG-2.0",
    completed_mask_backend: str = "auto",
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "status": status,
        "backend": "hunyuan3d",
        "model": model,
        "device": device,
        "source": source,
        "completed_mask_backend": completed_mask_backend,
        "with_texture": with_texture,
        "texture_resolution": texture_resolution,
        "texture_views": texture_views,
        "texture_use_remesh": texture_use_remesh,
        "texture_prompt": texture_prompt,
        "texture_reference_mode": texture_reference_mode,
        "texture_matte_backend": texture_matte_backend,
        "texture_matte_model_dir": texture_matte_model_dir,
        "object_count": len(records),
        "objects": records,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    objects_dir.mkdir(parents=True, exist_ok=True)
    (objects_dir / "hunyuan3d_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Internal Hunyuan3D object helpers.")
    parser.add_argument("--texture-one", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--views", type=int, default=6)
    parser.add_argument("--use-remesh", action="store_true")
    parser.add_argument("--no-remesh", action="store_true")
    parser.add_argument("--prompt", default=DEFAULT_TEXTURE_PROMPT)
    parser.add_argument("--reference-mode", choices=("original", "masked-crop"), default="original")
    parser.add_argument("--matte-backend", choices=("auto", "bria-rmbg", "mask"), default="auto")
    parser.add_argument("--matte-model-dir", default="Models/Segmentation/BRIA/RMBG-2.0")
    args = parser.parse_args()

    if args.texture_one is None:
        parser.error("--texture-one is required for direct module execution")
    metadata_path = args.texture_one / "hunyuan3d_metadata.json"
    if metadata_path.is_file():
        record = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        record = {"object_dir": str(args.texture_one), "status": "ok"}
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    paint_pipeline = load_paint_pipeline(device=args.device, resolution=args.resolution, views=args.views, prompt=args.prompt)
    texture_object_dir(
        args.texture_one,
        record,
        paint_pipeline=paint_pipeline,
        use_remesh=not args.no_remesh,
        prompt=args.prompt,
        reference_mode=args.reference_mode,
        matte_backend=args.matte_backend,
        matte_model_dir=args.matte_model_dir,
    )
    torch = import_torch()
    del paint_pipeline
    release_torch_memory(torch, args.device)
    if record.get("texture_status") == "ok":
        textured_mesh = record.get("textured_mesh")
        if textured_mesh:
            print(f"Wrote {args.texture_one / textured_mesh}")
        return 0
    reason = record.get("texture_reason") or "unknown texture failure"
    print(f"Hunyuan3D paint failed: {reason}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
