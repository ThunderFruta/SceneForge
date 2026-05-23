from __future__ import annotations

from PIL import Image

from Core.Types.mesh_data import MeshData
from Core.Types.scene_data import SceneMeshPart, StructuredSceneData
from Export.OBJ.obj_exporter import export_obj, export_scene_obj


def test_export_obj_writes_geometry_material_and_texture(tmp_path) -> None:
    mesh = MeshData(
        vertices=[(-0.5, 0.5, 0.0), (0.5, 0.5, 0.5), (-0.5, -0.5, 1.0)],
        faces=[(0, 1, 2)],
        uvs=[(0.0, 1.0), (1.0, 1.0), (0.0, 0.0)],
        normals=[(0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0)],
        columns=2,
        rows=2,
    )
    image = Image.new("RGB", (1, 1), (255, 0, 0))

    result = export_obj(mesh, tmp_path / "mesh.obj", texture_image=image)

    obj_text = result.obj_path.read_text(encoding="utf-8")
    assert "mtllib mesh.mtl" in obj_text
    assert "usemtl sceneforge_material" in obj_text
    assert "v -0.500000 0.500000 0.000000" in obj_text
    assert "vt 1.000000 1.000000" in obj_text
    assert "vn 0.000000 0.000000 1.000000" in obj_text
    assert "f 1/1/1 2/2/2 3/3/3" in obj_text
    assert result.mtl_path is not None
    assert "map_Kd mesh_texture.png" in result.mtl_path.read_text(encoding="utf-8")
    assert result.texture_path is not None
    assert result.texture_path.exists()


def test_export_scene_obj_writes_named_parts(tmp_path) -> None:
    scene = StructuredSceneData(
        plane_parts=[
            SceneMeshPart(
                name="plane_000",
                kind="plane",
                vertices=[(-0.5, 0.5, 0.1), (0.5, 0.5, 0.1), (-0.5, -0.5, 0.1)],
                faces=[(0, 1, 2)],
                uvs=[(0.0, 1.0), (1.0, 1.0), (0.0, 0.0)],
                normals=[(0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0)],
            )
        ],
        detail_parts=[
            SceneMeshPart(
                name="detail_000",
                kind="detail",
                vertices=[(0.0, 0.0, 0.2), (0.25, 0.0, 0.2), (0.0, -0.25, 0.2)],
                faces=[(0, 1, 2)],
                uvs=[(0.5, 0.5), (0.75, 0.5), (0.5, 0.25)],
                normals=[(1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
            )
        ],
    )

    result = export_scene_obj(scene, tmp_path / "scene.obj")
    obj_text = result.obj_path.read_text(encoding="utf-8")

    assert "o plane_000" in obj_text
    assert "o detail_000" in obj_text
    assert "vn 0.000000 1.000000 0.000000" in obj_text
    assert "vn 1.000000 0.000000 0.000000" in obj_text
    assert "f 1/1/1 2/2/2 3/3/3" in obj_text
    assert "f 4/4/4 5/5/5 6/6/6" in obj_text
