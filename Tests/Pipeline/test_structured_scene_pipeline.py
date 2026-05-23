from __future__ import annotations

from pathlib import Path

from PIL import Image

from Export.Blend.blend_exporter import BlendExportResult
from Pipeline.StructuredScene import structured_scene_pipeline


def _write_flat_fixture_pair(tmp_path: Path) -> tuple[Path, Path]:
    image_path = tmp_path / "flat_rgb.png"
    depth_path = tmp_path / "flat_depth.png"
    Image.new("RGB", (8, 8), (80, 140, 200)).save(image_path)
    Image.new("L", (8, 8), 128).save(depth_path)
    return image_path, depth_path


def test_build_structured_scene_data_omits_details_by_default() -> None:
    depth = [[0.4 for _column in range(5)] for _row in range(5)]
    depth[2][2] = 0.9

    scene = structured_scene_pipeline.build_structured_scene_data(
        depth,
        resolution=5,
        depth_strength=1.0,
    )

    assert scene.plane_parts
    assert not scene.detail_parts
    assert scene.plane_parts[0].name.startswith("plane_")


def test_build_structured_scene_data_can_include_detail_patches() -> None:
    depth = [[0.4 for _column in range(5)] for _row in range(5)]
    depth[2][2] = 0.9

    scene = structured_scene_pipeline.build_structured_scene_data(
        depth,
        resolution=5,
        depth_strength=1.0,
        include_details=True,
    )

    assert scene.plane_parts
    assert scene.detail_parts
    assert scene.detail_parts[0].name.startswith("detail_")
    assert scene.detail_parts[-1].name == "coverage_000"


def test_structured_scene_data_solidifies_by_default() -> None:
    depth = [[0.4 for _column in range(5)] for _row in range(5)]

    solidified = structured_scene_pipeline.build_structured_scene_data(
        depth,
        resolution=5,
        depth_strength=1.0,
    )
    front_only = structured_scene_pipeline.build_structured_scene_data(
        depth,
        resolution=5,
        depth_strength=1.0,
        solidify=False,
    )

    assert len(solidified.plane_parts[0].vertices) > len(front_only.plane_parts[0].vertices)
    assert len(solidified.plane_parts[0].faces) > len(front_only.plane_parts[0].faces)
    assert solidified.plane_parts[0].normals is not None
    assert len(solidified.plane_parts[0].normals) == len(solidified.plane_parts[0].vertices)


def test_structured_scene_data_can_disable_solidification() -> None:
    depth = [[0.4 for _column in range(5)] for _row in range(5)]

    scene = structured_scene_pipeline.build_structured_scene_data(
        depth,
        resolution=5,
        depth_strength=1.0,
        solidify=False,
    )

    part = scene.plane_parts[0]
    assert part.normals is not None
    assert len(part.vertices) == 36


def test_structured_pipeline_exports_blend_without_persistent_obj_by_default(
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
        structured_scene_pipeline,
        "export_blend_from_obj",
        fake_export_blend_from_obj,
    )
    image_path, depth_path = _write_flat_fixture_pair(tmp_path)

    result = structured_scene_pipeline.run_structured_scene_pipeline(
        image_path=image_path,
        depth_path=depth_path,
        output_path=tmp_path / "scene.blend",
        resolution=8,
        depth_strength=1.0,
        write_texture=True,
    )

    assert result.blend_path == tmp_path / "scene.blend"
    assert result.preview_path == tmp_path / "preview.png"
    assert result.blend_path.exists()
    assert result.preview_path.exists()
    assert result.obj_result is None
    assert result.scene_data.all_parts
    assert not (tmp_path / "scene.obj").exists()


def test_structured_pipeline_keeps_sidecar_obj_when_requested(
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
        structured_scene_pipeline,
        "export_blend_from_obj",
        fake_export_blend_from_obj,
    )
    image_path, depth_path = _write_flat_fixture_pair(tmp_path)

    result = structured_scene_pipeline.run_structured_scene_pipeline(
        image_path=image_path,
        depth_path=depth_path,
        output_path=tmp_path / "scene.blend",
        resolution=8,
        depth_strength=1.0,
        write_texture=True,
        keep_obj=True,
    )

    assert result.obj_result is not None
    assert result.preview_path == tmp_path / "preview.png"
    assert result.preview_path.exists()
    assert result.obj_result.obj_path == tmp_path / "scene.obj"
    obj_text = result.obj_result.obj_path.read_text(encoding="utf-8")
    assert "o " in obj_text
    assert "mtllib scene.mtl" in obj_text
