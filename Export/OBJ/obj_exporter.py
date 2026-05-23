from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from Core.Types.mesh_data import MeshData
from Core.Types.scene_data import StructuredSceneData


@dataclass(frozen=True)
class ObjExportResult:
    obj_path: Path
    mtl_path: Path | None
    texture_path: Path | None


def export_obj(
    mesh: MeshData,
    output_path: str | Path,
    *,
    texture_image: Image.Image | None = None,
    material_name: str = "sceneforge_material",
) -> ObjExportResult:
    obj_path = Path(output_path)
    obj_path.parent.mkdir(parents=True, exist_ok=True)

    mtl_path = None
    texture_path = None
    if texture_image is not None:
        texture_path = obj_path.with_name(f"{obj_path.stem}_texture.png")
        mtl_path = obj_path.with_suffix(".mtl")
        texture_image.save(texture_path)
        _write_mtl(mtl_path, texture_path.name, material_name)

    lines = []
    if mtl_path is not None:
        lines.append(f"mtllib {mtl_path.name}\n")
        lines.append(f"usemtl {material_name}\n")

    for x, y, z in mesh.vertices:
        lines.append(f"v {x:.6f} {y:.6f} {z:.6f}\n")
    for u, v in mesh.uvs:
        lines.append(f"vt {u:.6f} {v:.6f}\n")
    normals = mesh.normals if mesh.normals is not None else []
    for x, y, z in normals:
        lines.append(f"vn {x:.6f} {y:.6f} {z:.6f}\n")
    for a, b, c in mesh.faces:
        if normals:
            lines.append(
                f"f {a + 1}/{a + 1}/{a + 1} "
                f"{b + 1}/{b + 1}/{b + 1} "
                f"{c + 1}/{c + 1}/{c + 1}\n"
            )
        else:
            lines.append(
                f"f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}\n"
            )

    obj_path.write_text("".join(lines), encoding="utf-8")
    return ObjExportResult(obj_path=obj_path, mtl_path=mtl_path, texture_path=texture_path)


def export_scene_obj(
    scene: StructuredSceneData,
    output_path: str | Path,
    *,
    texture_image: Image.Image | None = None,
    material_name: str = "sceneforge_material",
) -> ObjExportResult:
    obj_path = Path(output_path)
    obj_path.parent.mkdir(parents=True, exist_ok=True)

    mtl_path = None
    texture_path = None
    if texture_image is not None:
        texture_path = obj_path.with_name(f"{obj_path.stem}_texture.png")
        mtl_path = obj_path.with_suffix(".mtl")
        texture_image.save(texture_path)
        _write_mtl(mtl_path, texture_path.name, material_name)

    lines = []
    if mtl_path is not None:
        lines.append(f"mtllib {mtl_path.name}\n")

    vertex_offset = 0
    uv_offset = 0
    normal_offset = 0
    for part in scene.all_parts:
        lines.append(f"o {part.name}\n")
        if mtl_path is not None:
            lines.append(f"usemtl {material_name}\n")
        for x, y, z in part.vertices:
            lines.append(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for u, v in part.uvs:
            lines.append(f"vt {u:.6f} {v:.6f}\n")
        normals = part.normals if part.normals is not None else []
        for x, y, z in normals:
            lines.append(f"vn {x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in part.faces:
            if normals:
                lines.append(
                    "f "
                    f"{a + 1 + vertex_offset}/{a + 1 + uv_offset}/{a + 1 + normal_offset} "
                    f"{b + 1 + vertex_offset}/{b + 1 + uv_offset}/{b + 1 + normal_offset} "
                    f"{c + 1 + vertex_offset}/{c + 1 + uv_offset}/{c + 1 + normal_offset}\n"
                )
            else:
                lines.append(
                    "f "
                    f"{a + 1 + vertex_offset}/{a + 1 + uv_offset} "
                    f"{b + 1 + vertex_offset}/{b + 1 + uv_offset} "
                    f"{c + 1 + vertex_offset}/{c + 1 + uv_offset}\n"
                )
        vertex_offset += len(part.vertices)
        uv_offset += len(part.uvs)
        normal_offset += len(normals)

    obj_path.write_text("".join(lines), encoding="utf-8")
    return ObjExportResult(obj_path=obj_path, mtl_path=mtl_path, texture_path=texture_path)


def _write_mtl(mtl_path: Path, texture_name: str, material_name: str) -> None:
    mtl_path.write_text(
        "\n".join(
            [
                f"newmtl {material_name}",
                "Ka 1.000000 1.000000 1.000000",
                "Kd 1.000000 1.000000 1.000000",
                "Ks 0.000000 0.000000 0.000000",
                "d 1.0",
                "illum 2",
                f"map_Kd {texture_name}",
                "",
            ]
        ),
        encoding="utf-8",
    )
