from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest

from Export.Blend.blend_exporter import BlendExportResult
from Pipeline.StructuredScene import structured_scene_pipeline
from Segmentation.Core.segmentation_labels import SegmentationLabel
from Segmentation.Core.segmentation_mask import SegmentationMask


def _write_flat_fixture_pair(tmp_path: Path) -> tuple[Path, Path]:
    image_path = tmp_path / "flat_rgb.png"
    depth_path = tmp_path / "flat_depth.png"
    Image.new("RGB", (8, 8), (80, 140, 200)).save(image_path)
    Image.new("L", (8, 8), 128).save(depth_path)
    return image_path, depth_path


def _fake_export_blend_from_obj(*, obj_path, blend_path, blender_executable):
    Path(blend_path).write_text(f"fake blend from {Path(obj_path).name}", encoding="utf-8")
    preview_path = Path(blend_path).with_name("preview.png")
    preview_path.write_text("fake preview", encoding="utf-8")
    return BlendExportResult(
        blend_path=Path(blend_path),
        preview_path=preview_path,
    )


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


def test_build_structured_scene_data_uses_segmentation_mask() -> None:
    depth = [[0.4, 0.8], [0.2, 0.9]]
    mask = SegmentationMask.from_labels(
        [
            [SegmentationLabel.WALL, SegmentationLabel.WALL],
            [SegmentationLabel.OBJECT, SegmentationLabel.OBJECT],
        ]
    )

    scene = structured_scene_pipeline.build_structured_scene_data(
        depth,
        resolution=2,
        depth_strength=1.0,
        include_details=True,
        solidify=False,
        segmentation_mask=mask,
    )

    assert [part.name for part in scene.plane_parts] == ["plane_000"]
    assert [part.name for part in scene.detail_parts] == ["detail_000", "coverage_000"]
    assert scene.plane_parts[0].normals is not None


def test_build_structured_scene_data_keeps_segmentation_detail_without_details() -> None:
    depth = [[0.4, 0.8], [0.2, 0.9]]
    mask = SegmentationMask.from_labels(
        [
            [SegmentationLabel.OBJECT, SegmentationLabel.OBJECT],
            [SegmentationLabel.OBJECT, SegmentationLabel.OBJECT],
        ]
    )

    scene = structured_scene_pipeline.build_structured_scene_data(
        depth,
        resolution=2,
        depth_strength=1.0,
        solidify=False,
        segmentation_mask=mask,
    )

    assert not scene.plane_parts
    assert [part.name for part in scene.detail_parts] == ["detail_000"]


def test_structured_pipeline_rejects_mask_without_mask_mode(tmp_path) -> None:
    image_path, depth_path = _write_flat_fixture_pair(tmp_path)
    mask_path = tmp_path / "mask.png"
    Image.new("RGB", (8, 8), (255, 0, 0)).save(mask_path)

    with pytest.raises(ValueError, match="requires `--segmentation mask`"):
        structured_scene_pipeline.run_structured_scene_pipeline(
            image_path=image_path,
            depth_path=depth_path,
            output_path=tmp_path / "scene.blend",
            segmentation="none",
            mask_path=mask_path,
        )


def test_structured_pipeline_auto_segmentation_runs_without_extra_dependencies(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        structured_scene_pipeline,
        "export_blend_from_obj",
        _fake_export_blend_from_obj,
    )
    image_path, depth_path = _write_flat_fixture_pair(tmp_path)

    result = structured_scene_pipeline.run_structured_scene_pipeline(
        image_path=image_path,
        depth_path=depth_path,
        output_path=tmp_path / "scene.blend",
        segmentation="auto",
    )

    assert result.scene_data.all_parts
    assert (tmp_path / "metrics.json").exists()


def test_structured_pipeline_records_mask_cleanup_counts(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        structured_scene_pipeline,
        "export_blend_from_obj",
        _fake_export_blend_from_obj,
    )
    image_path = tmp_path / "room_rgb.png"
    depth_path = tmp_path / "room_depth.png"
    mask_path = tmp_path / "room_mask.png"
    Image.new("RGB", (5, 5), (80, 140, 200)).save(image_path)
    Image.new("L", (5, 5), 128).save(depth_path)
    mask = Image.new("RGB", (5, 5), (255, 0, 0))
    mask.putpixel((2, 2), (0, 0, 0))
    mask.save(mask_path)

    result = structured_scene_pipeline.run_structured_scene_pipeline(
        image_path=image_path,
        depth_path=depth_path,
        output_path=tmp_path / "scene.blend",
        resolution=5,
        segmentation="mask",
        mask_path=mask_path,
        solidify=False,
    )

    metrics = json.loads((result.blend_path.parent / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["cleanup_counts"]["filled_mask_holes"] == 1


def test_structured_pipeline_exports_blend_without_persistent_obj_by_default(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        structured_scene_pipeline,
        "export_blend_from_obj",
        _fake_export_blend_from_obj,
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
    monkeypatch.setattr(
        structured_scene_pipeline,
        "export_blend_from_obj",
        _fake_export_blend_from_obj,
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


def test_structured_pipeline_writes_metrics_json_with_expected_fields(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        structured_scene_pipeline,
        "export_blend_from_obj",
        _fake_export_blend_from_obj,
    )
    image_path, depth_path = _write_flat_fixture_pair(tmp_path)

    result = structured_scene_pipeline.run_structured_scene_pipeline(
        image_path=image_path,
        depth_path=depth_path,
        output_path=tmp_path / "scene.blend",
        resolution=8,
        depth_strength=1.0,
        write_texture=False,
    )

    metrics_path = result.blend_path.parent / "metrics.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert set(metrics) >= {
        "runtime_breakdown",
        "peak_memory_bytes",
        "region_confidence_inputs",
        "fallback_counts",
        "cleanup_counts",
        "mesh_validity",
        "seam_diagnostics",
    }
    runtime = metrics["runtime_breakdown"]
    assert set(runtime) >= {"segmentation", "region_build", "mesh_build", "export"}
    for value in runtime.values():
        assert isinstance(value, (int, float))
        assert value >= 0.0

    assert isinstance(metrics["peak_memory_bytes"], int) or metrics["peak_memory_bytes"] is None
    assert isinstance(metrics["region_confidence_inputs"], list)
    assert all(
        {
            "region_name",
            "kind",
            "silhouette_error_proxy",
            "depth_error_proxy",
            "curvature_spike_proxy",
        } <= set(region)
        for region in metrics["region_confidence_inputs"]
    )
    assert set(metrics["fallback_counts"]) >= {"primitive", "base"}
    for count in metrics["fallback_counts"].values():
        assert isinstance(count, int)
        assert count >= 0

    assert set(metrics["cleanup_counts"]) >= {
        "filled_mask_holes",
        "removed_mask_islands",
        "patched_mesh_holes",
        "rejected_spikes",
    }
    for count in metrics["cleanup_counts"].values():
        assert isinstance(count, int)
        assert count >= 0

    validity = metrics["mesh_validity"]
    assert set(validity) >= {
        "non_manifold_edge_count",
        "degenerate_face_count",
        "disconnected_component_count",
    }
    assert all(
        isinstance(validity[field], int) and validity[field] >= 0
        for field in validity
    )

    seams = metrics["seam_diagnostics"]
    assert set(seams) >= {
        "boundary_edge_count_before_cleanup",
        "boundary_edge_count_after_cleanup",
        "occlusion_gap_count",
        "occlusion_gap_area_proxy",
    }
    assert seams["boundary_edge_count_before_cleanup"] >= 0
    assert seams["boundary_edge_count_after_cleanup"] >= 0
    assert seams["occlusion_gap_count"] >= 0
    assert seams["occlusion_gap_area_proxy"] >= 0.0
