from __future__ import annotations

import json
import gc
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from ObjectReconstruction.triposr_objects import (
    prepare_triposr_input,
    select_source_image,
)


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
) -> dict[str, Any]:
    root = Path(objects_dir)
    if not root.is_dir():
        return write_manifest(root, [], "missing_objects_dir", model=model, device=device, source=source)

    object_dirs = [path for path in sorted(root.iterdir()) if path.is_dir()]
    selected_dirs = object_dirs if max_objects <= 0 else object_dirs[:max_objects]
    if not selected_dirs:
        return write_manifest(root, [], "no_objects", model=model, device=device, source=source)

    torch = import_torch()
    resolved_device = resolve_device(torch=torch, device=device)
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
                order_index=index,
            )
        )
    del pipeline
    release_torch_memory(torch, resolved_device)

    if with_texture:
        texture_records(root, records, device=resolved_device, resolution=texture_resolution, views=texture_views, use_remesh=texture_use_remesh)
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


def reconstruct_object_dir(
    object_dir: Path,
    *,
    pipeline,
    source: str,
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

    input_path = object_dir / "hunyuan3d_input.png"
    mask_path = object_dir / "hunyuan3d_mask.png"
    obj_path = object_dir / "hunyuan3d_mesh.obj"
    glb_path = object_dir / "hunyuan3d_mesh.glb"
    prepared_image, prepared_mask = prepare_triposr_input(source_path, object_dir)
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
        "hunyuan3d_input": input_path.name,
        "hunyuan3d_mask": mask_path.name,
        "mesh": obj_path.name if status == "ok" else None,
        "glb": glb_path.name if status == "ok" else None,
        "order_index": order_index,
    }
    write_object_metadata(object_dir, record)
    return record


def texture_records(
    root: Path,
    records: list[dict[str, Any]],
    *,
    device: str,
    resolution: int,
    views: int,
    use_remesh: bool,
) -> None:
    ok_records = [record for record in records if record.get("status") == "ok"]
    if not ok_records:
        return
    if not device.startswith("cuda"):
        raise RuntimeError("Hunyuan3D paint is only enabled for CUDA in SceneForge right now.")
    print(f"Running Hunyuan3D paint for {len(ok_records)} reconstructed objects.", flush=True)
    paint_pipeline = load_paint_pipeline(device=device, resolution=resolution, views=views)
    for index, record in enumerate(ok_records, start=1):
        object_dir = root / Path(record["object_dir"]).name
        print(f"Hunyuan3D paint {index}/{len(ok_records)}: {object_dir.name}", flush=True)
        texture_object_dir(object_dir, record, paint_pipeline=paint_pipeline, use_remesh=use_remesh)


def load_paint_pipeline(*, device: str, resolution: int, views: int):
    add_local_hunyuan3d_paths()
    paint_root = Path("Models/Mesh/Hunyuan3D/repo/hy3dpaint").resolve()
    realesrgan_path = paint_root / "ckpt" / "RealESRGAN_x4plus.pth"
    if not realesrgan_path.is_file():
        raise RuntimeError(
            "Hunyuan3D paint needs RealESRGAN_x4plus.pth at "
            "Models/Mesh/Hunyuan3D/repo/hy3dpaint/ckpt/RealESRGAN_x4plus.pth."
        )
    os.environ.setdefault("HF_HUB_CACHE", str(Path("Models/Mesh/Hunyuan3D/hf-cache").resolve()))
    try:
        from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline
    except Exception as exc:
        raise RuntimeError(
            "Hunyuan3D paint import failed. It needs the hy3dpaint dependencies and compiled rasterizer/renderer extensions."
        ) from exc

    config = Hunyuan3DPaintConfig(max_num_view=views, resolution=resolution)
    config.device = device
    config.realesrgan_ckpt_path = str(realesrgan_path)
    config.multiview_cfg_path = str(paint_root / "cfgs" / "hunyuan-paint-pbr.yaml")
    config.custom_pipeline = str(paint_root / "hunyuanpaintpbr")
    return Hunyuan3DPaintPipeline(config)


def texture_object_dir(
    object_dir: Path,
    record: dict[str, Any],
    *,
    paint_pipeline,
    use_remesh: bool,
) -> None:
    mesh_path = object_dir / "hunyuan3d_mesh.glb"
    if not mesh_path.is_file():
        mesh_path = object_dir / "hunyuan3d_mesh.obj"
    input_path = object_dir / "hunyuan3d_input.png"
    output_obj = object_dir / "hunyuan3d_textured.obj"
    output_glb = object_dir / "hunyuan3d_textured.glb"
    try:
        paint_pipeline(
            mesh_path=str(mesh_path),
            image_path=str(input_path),
            output_mesh_path=str(output_obj),
            use_remesh=use_remesh,
            save_glb=True,
        )
        record.update(
            {
                "texture_status": "ok",
                "texture_reason": None,
                "textured_mesh": output_obj.name,
                "textured_glb": output_glb.name if output_glb.is_file() else None,
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
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "status": status,
        "backend": "hunyuan3d",
        "model": model,
        "device": device,
        "source": source,
        "with_texture": with_texture,
        "texture_resolution": texture_resolution,
        "texture_views": texture_views,
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
