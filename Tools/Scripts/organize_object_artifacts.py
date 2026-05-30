from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

try:
    import bpy
except Exception:
    bpy = None


SEGMENTATION_FILES = (
    "full_mask.png",
    "rgb_crop.png",
    "mask.png",
    "context_crop.png",
    "context_mask.png",
    "context_masked_crop.png",
    "context_focus_crop.png",
)
COMPLETION_FILES = (
    "completion_openai_input.png",
    "completion_openai_reference.png",
    "completion_input.png",
    "completion_mask.png",
    "completion_expected_alpha.png",
    "completion_input_crop.png",
)
RECONSTRUCTION_FILES = (
    "completed_mask.png",
    "completed_mask_metadata.json",
    "hunyuan3d_input.png",
    "hunyuan3d_mask.png",
    "hunyuan3d_paint_input.png",
    "white_mesh_remesh.obj",
    "triposr_input.png",
    "triposr_mask.png",
)
TEXTURE_BUNDLE_FILES = (
    "hunyuan3d_textured.obj",
    "hunyuan3d_textured.mtl",
    "hunyuan3d_textured.jpg",
    "hunyuan3d_textured.png",
    "hunyuan3d_textured_metallic.jpg",
    "hunyuan3d_textured_metallic.png",
    "hunyuan3d_textured_roughness.jpg",
    "hunyuan3d_textured_roughness.png",
)
DIAGNOSTIC_FILES = (
    "mesh_inspection.json",
    "texture_inspection.json",
    "texture_inspection_latest.json",
    "uv_diagnostics.json",
    "uv_occupancy.png",
    "uv_wireframe.png",
    "uv_overlap_heat.png",
    "preview_obj.png",
    "preview_glb.png",
    "hunyuan3d_textured_clean_import.blend",
)


def move_file(object_dir: Path, name: str, target_subdir: str) -> str | None:
    source = object_dir / name
    if not source.is_file():
        return None
    target_dir = object_dir / "artifacts" / target_subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / name
    if target.exists():
        target.unlink()
    shutil.move(str(source), str(target))
    return target.relative_to(object_dir).as_posix()


def convert_textured_obj_to_glb(object_dir: Path) -> Path | None:
    glb_path = object_dir / "hunyuan3d_textured.glb"
    if glb_path.is_file():
        return glb_path
    legacy_glb = object_dir / "hunyuan3d_textured_from_obj.glb"
    if legacy_glb.is_file():
        legacy_glb.replace(glb_path)
        return glb_path
    obj_path = object_dir / "hunyuan3d_textured.obj"
    if not obj_path.is_file():
        obj_path = object_dir / "artifacts" / "textures" / "hunyuan3d_textured.obj"
    if not obj_path.is_file() or bpy is None:
        return None
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    bpy.ops.wm.obj_import(filepath=str(obj_path))
    bpy.ops.export_scene.gltf(filepath=str(glb_path), use_active_scene=True)
    return glb_path if glb_path.is_file() else None


def update_json_paths(path: Path, replacements: dict[str, str]) -> None:
    if not path.is_file() or not replacements:
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    updated = replace_json_values(payload, replacements)
    path.write_text(json.dumps(updated, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def update_json_fields(path: Path, fields: dict[str, Any]) -> None:
    if not path.is_file() or not fields:
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(payload, dict):
        return
    payload.update(fields)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def replace_json_values(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        return replacements.get(value, value)
    if isinstance(value, list):
        return [replace_json_values(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: replace_json_values(item, replacements) for key, item in value.items()}
    return value


def repair_hunyuan_manifest(objects_dir: Path) -> None:
    manifest_path = objects_dir / "hunyuan3d_manifest.json"
    if not manifest_path.is_file():
        return
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return
    records = payload.get("objects") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        return
    for record in records:
        if not isinstance(record, dict):
            continue
        object_name = Path(str(record.get("object_dir", ""))).name
        if object_name and (objects_dir / object_name / "hunyuan3d_textured.glb").is_file():
            record["textured_glb"] = "hunyuan3d_textured.glb"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def organize_object_dir(object_dir: Path, *, convert_glb: bool) -> dict[str, Any]:
    replacements: dict[str, str] = {}
    if convert_glb:
        convert_textured_obj_to_glb(object_dir)

    for name in SEGMENTATION_FILES:
        moved = move_file(object_dir, name, "segmentation")
        if moved:
            replacements[name] = moved
    for name in COMPLETION_FILES:
        moved = move_file(object_dir, name, "completion")
        if moved:
            replacements[name] = moved
    for name in RECONSTRUCTION_FILES:
        moved = move_file(object_dir, name, "reconstruction")
        if moved:
            replacements[name] = moved

    textured_glb_exists = (object_dir / "hunyuan3d_textured.glb").is_file()
    if textured_glb_exists:
        for name in ("hunyuan3d_mesh.obj", "hunyuan3d_mesh.glb"):
            moved = move_file(object_dir, name, "reconstruction")
            if moved:
                replacements[name] = moved
        for name in TEXTURE_BUNDLE_FILES:
            moved = move_file(object_dir, name, "textures")
            if moved:
                replacements[name] = moved

    for name in DIAGNOSTIC_FILES:
        moved = move_file(object_dir, name, "diagnostics")
        if moved:
            replacements[name] = moved

    for json_path in (
        object_dir / "completion_metadata.json",
        object_dir / "hunyuan3d_metadata.json",
        object_dir / "triposr_metadata.json",
        object_dir / "artifacts" / "reconstruction" / "completed_mask_metadata.json",
    ):
        update_json_paths(json_path, replacements)
    if textured_glb_exists:
        update_json_fields(object_dir / "hunyuan3d_metadata.json", {"textured_glb": "hunyuan3d_textured.glb"})

    return {
        "object_dir": str(object_dir),
        "moved": len(replacements),
        "replacements": replacements,
        "textured_glb": "hunyuan3d_textured.glb" if textured_glb_exists else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Move nonessential object outputs into per-object artifacts folders.")
    parser.add_argument("objects_dir", type=Path)
    parser.add_argument("--convert-glb", action="store_true", help="Convert root textured OBJ bundles to root GLB before moving the OBJ bundle.")
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(argv)

    object_dirs = [path for path in sorted(args.objects_dir.iterdir()) if path.is_dir()]
    records = [organize_object_dir(object_dir, convert_glb=args.convert_glb) for object_dir in object_dirs]
    manifest_replacements: dict[str, str] = {}
    for record in records:
        manifest_replacements.update(record["replacements"])
    for manifest_path in args.objects_dir.glob("*_manifest.json"):
        update_json_paths(manifest_path, manifest_replacements)
    repair_hunyuan_manifest(args.objects_dir)
    print(json.dumps({"objects_dir": str(args.objects_dir), "objects": records}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
