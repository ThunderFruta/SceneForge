from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def import_mesh(path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    elif suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
    else:
        raise ValueError(f"Unsupported mesh format: {path.suffix}")


def export_mesh(path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix in {".glb", ".gltf"}:
        bpy.ops.export_scene.gltf(filepath=str(path), use_active_scene=True)
    elif suffix == ".obj":
        bpy.ops.wm.obj_export(filepath=str(path))
    else:
        raise ValueError(f"Unsupported mesh format: {path.suffix}")


def clean_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def object_bounds(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    vertices = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
    min_v = Vector((min(v.x for v in vertices), min(v.y for v in vertices), min(v.z for v in vertices)))
    max_v = Vector((max(v.x for v in vertices), max(v.y for v in vertices), max(v.z for v in vertices)))
    return min_v, max_v


def remove_support_sheets(
    obj: bpy.types.Object,
    *,
    sheet_axis: str = "z",
    sheet_side: str = "max",
    sheet_band_ratio: float = 0.085,
    keep_radius_ratio: float = 0.31,
) -> int:
    axes = {"x": 0, "y": 1, "z": 2}
    axis = axes[sheet_axis]
    other_axes = [index for index in range(3) if index != axis]
    min_v, max_v = object_bounds(obj)
    min_values = [min_v.x, min_v.y, min_v.z]
    max_values = [max_v.x, max_v.y, max_v.z]
    extents = [max_values[i] - min_values[i] for i in range(3)]
    if min(extents) <= 0:
        return 0

    center_other = [(min_values[i] + max_values[i]) * 0.5 for i in other_axes]
    radius_other = [max(extents[i] * keep_radius_ratio, 1e-6) for i in other_axes]
    band = extents[axis] * sheet_band_ratio
    if sheet_side == "max":
        in_band = lambda value: value >= max_values[axis] - band
    elif sheet_side == "min":
        in_band = lambda value: value <= min_values[axis] + band
    else:
        raise ValueError("--sheet-side must be min or max")

    mesh = obj.data
    remove_indices: list[int] = []
    for poly in mesh.polygons:
        coords = [obj.matrix_world @ mesh.vertices[index].co for index in poly.vertices]
        center = sum(coords, Vector()) / len(coords)
        values = [center.x, center.y, center.z]
        if not in_band(values[axis]):
            continue
        normalized = [
            abs(values[other_axis] - center_other[i]) / radius_other[i]
            for i, other_axis in enumerate(other_axes)
        ]
        if max(normalized) > 1.0:
            remove_indices.append(poly.index)

    if not remove_indices:
        return 0
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_mode(type="FACE")
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.object.mode_set(mode="OBJECT")
    for vertex in mesh.vertices:
        vertex.select = False
    for edge in mesh.edges:
        edge.select = False
    for poly in mesh.polygons:
        poly.select = poly.index in remove_indices
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.delete(type="FACE")
    bpy.ops.object.mode_set(mode="OBJECT")
    return len(remove_indices)


def run(input_path: Path, output_path: Path, *, sheet_axis: str, sheet_side: str, sheet_band_ratio: float, keep_radius_ratio: float) -> dict[str, object]:
    clean_scene()
    import_mesh(input_path)
    removed = 0
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            removed += remove_support_sheets(
                obj,
                sheet_axis=sheet_axis,
                sheet_side=sheet_side,
                sheet_band_ratio=sheet_band_ratio,
                keep_radius_ratio=keep_radius_ratio,
            )
    export_mesh(output_path)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "removed_faces": removed,
        "sheet_axis": sheet_axis,
        "sheet_side": sheet_side,
        "sheet_band_ratio": sheet_band_ratio,
        "keep_radius_ratio": keep_radius_ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove large thin Hunyuan support sheets from object GLB/OBJ outputs.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sheet-axis", choices=("x", "y", "z"), default="z")
    parser.add_argument("--sheet-side", choices=("min", "max"), default="max")
    parser.add_argument("--sheet-band-ratio", type=float, default=0.085)
    parser.add_argument("--keep-radius-ratio", type=float, default=0.31)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(argv)
    report = run(
        args.input,
        args.output,
        sheet_axis=args.sheet_axis,
        sheet_side=args.sheet_side,
        sheet_band_ratio=args.sheet_band_ratio,
        keep_radius_ratio=args.keep_radius_ratio,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
