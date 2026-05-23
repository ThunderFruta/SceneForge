from __future__ import annotations

import pytest

from Export.Blend.blend_exporter import _build_blender_import_script, export_blend_from_obj


def test_export_blend_from_obj_requires_blender(tmp_path) -> None:
    obj_path = tmp_path / "mesh.obj"
    obj_path.write_text("o empty\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Blender executable was not found"):
        export_blend_from_obj(
            obj_path=obj_path,
            blend_path=tmp_path / "mesh.blend",
            blender_executable="definitely-not-blender",
        )


def test_blender_import_script_uses_source_facing_preview_camera(tmp_path) -> None:
    script = _build_blender_import_script(
        tmp_path / "mesh.obj",
        tmp_path / "mesh.blend",
        tmp_path / "preview.png",
    )

    assert 'camera.name = "sceneforge_source_preview_camera"' in script
    assert "camera_location = Vector((0.0, 0.0, 0.0))" in script
    assert 'forward_axis="Y"' in script
    assert 'up_axis="Z"' in script
    assert "global_scale=scene_scale" in script
    assert "scene_scale = 4.0" in script
    assert "vertex.co.x *= -1.0" not in script
    assert "obj.data.flip_normals()" not in script
    assert 'nodes.new(type="ShaderNodeEmission")' in script
    assert "direction = Vector((0.0, 1.0, 0.0))" in script
    assert 'direction.to_track_quat("-Z", "Y")' in script
    assert "texture_aspect = image.size[0] / image.size[1]" in script
