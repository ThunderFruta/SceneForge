from __future__ import annotations

from pathlib import Path

from Export.Blend.blend_exporter import BlendExportResult
from Pipeline.ImageToMesh import image_to_mesh_pipeline


FIXTURES = Path("Assets/Fixtures")


def test_pipeline_exports_blend_without_persistent_obj_by_default(
    tmp_path,
    monkeypatch,
) -> None:
    def fake_export_blend_from_obj(*, obj_path, blend_path, blender_executable):
        Path(blend_path).write_text(f"fake blend from {Path(obj_path).name}", encoding="utf-8")
        preview_path = Path(blend_path).with_name("preview.png")
        preview_path.write_text("fake preview", encoding="utf-8")
        return BlendExportResult(
            blend_path=Path(blend_path),
            preview_path=preview_path,
        )

    monkeypatch.setattr(
        image_to_mesh_pipeline,
        "export_blend_from_obj",
        fake_export_blend_from_obj,
    )

    result = image_to_mesh_pipeline.run_image_to_mesh_pipeline(
        image_path=FIXTURES / "tiny_rgb.ppm",
        depth_path=FIXTURES / "tiny_depth.pgm",
        output_path=tmp_path / "scene.blend",
        resolution=2,
        depth_strength=1.0,
        write_texture=True,
    )

    assert result.blend_path == tmp_path / "scene.blend"
    assert result.preview_path == tmp_path / "preview.png"
    assert result.blend_path.exists()
    assert result.preview_path.exists()
    assert result.obj_result is None
    assert not (tmp_path / "scene.obj").exists()
    assert not (tmp_path / "scene.mtl").exists()
    assert not (tmp_path / "scene_texture.png").exists()


def test_pipeline_keeps_sidecar_obj_when_requested(tmp_path, monkeypatch) -> None:
    def fake_export_blend_from_obj(*, obj_path, blend_path, blender_executable):
        Path(blend_path).write_text(f"fake blend from {Path(obj_path).name}", encoding="utf-8")
        preview_path = Path(blend_path).with_name("preview.png")
        preview_path.write_text("fake preview", encoding="utf-8")
        return BlendExportResult(
            blend_path=Path(blend_path),
            preview_path=preview_path,
        )

    monkeypatch.setattr(
        image_to_mesh_pipeline,
        "export_blend_from_obj",
        fake_export_blend_from_obj,
    )

    result = image_to_mesh_pipeline.run_image_to_mesh_pipeline(
        image_path=FIXTURES / "tiny_rgb.ppm",
        depth_path=FIXTURES / "tiny_depth.pgm",
        output_path=tmp_path / "scene.blend",
        resolution=2,
        depth_strength=1.0,
        write_texture=True,
        keep_obj=True,
    )

    assert result.blend_path.exists()
    assert result.preview_path == tmp_path / "preview.png"
    assert result.preview_path.exists()
    assert result.obj_result is not None
    assert result.obj_result.obj_path == tmp_path / "scene.obj"

    obj_text = result.obj_result.obj_path.read_text(encoding="utf-8")
    assert obj_text.count("\nv ") == 4
    assert obj_text.count("\nvn ") == 4
    assert obj_text.startswith("mtllib scene.mtl\nusemtl sceneforge_material\n")
    assert "v -0.500000 0.500000 0.000000" in obj_text
    assert "f 1/1/1 3/3/3 2/2/2" in obj_text
    assert result.obj_result.mtl_path is not None
    assert result.obj_result.texture_path is not None
