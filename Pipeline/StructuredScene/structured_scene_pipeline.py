from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import tempfile
import warnings

from Core.Types.scene_data import StructuredSceneData
from Export.Blend.blend_exporter import export_blend_from_obj
from Export.OBJ.obj_exporter import ObjExportResult, export_scene_obj
from Geometry.Cleanup.mask_cleanup import cleanup_segmentation_mask
from Geometry.Cleanup.scene_cleanup import cleanup_structured_scene
from Geometry.Mesh.coverage_relief_builder import build_coverage_relief_part
from Geometry.Mesh.region_relief_builder import build_region_relief_part
from Geometry.Normals.normal_builder import with_scene_normals
from Geometry.Planes.masked_plane_mesh_builder import build_masked_plane_part_with_fallback
from Geometry.Regions.region_analyzer import DepthRegion, analyze_depth_regions
from Geometry.Solidify.scan_solidifier import solidify_scene
from Input.Depth.depth_loader import derive_depth_from_luminance, load_depth_map
from Input.Image.image_loader import load_rgb_image
from Pipeline.ImageToMesh.image_to_mesh_pipeline import _as_blend_path
from Segmentation.Core.segmentation_mask import SegmentationMask
from Segmentation.Integration.mask_to_regions import segmentation_mask_to_regions
from Segmentation.Providers.Heuristic.heuristic_segmenter import build_heuristic_segmentation
from Segmentation.Providers.Manual.mask_loader import load_segmentation_mask
from Pipeline.StructuredScene.structured_scene_metrics import (
    MemoryTracker,
    build_structured_scene_metrics_payload,
    write_structured_scene_metrics,
)


@dataclass(frozen=True)
class StructuredSceneResult:
    blend_path: Path
    preview_path: Path
    obj_result: ObjExportResult | None
    scene_data: StructuredSceneData


@dataclass(frozen=True)
class _StructuredSceneBuildArtifacts:
    region_analysis: list[DepthRegion]
    region_build_seconds: float
    mesh_build_seconds: float
    analysis_columns: int
    analysis_rows: int
    fallback_counts: dict[str, int]
    cleanup_counts: dict[str, int]
    occlusion_gap_count: int
    occlusion_gap_area_proxy: float
    scene_before_cleanup: StructuredSceneData
    scene_after_cleanup: StructuredSceneData


def run_structured_scene_pipeline(
    *,
    image_path: str | Path,
    output_path: str | Path,
    depth_path: str | Path | None = None,
    resolution: int = 64,
    depth_strength: float = 1.0,
    write_texture: bool = True,
    keep_obj: bool = False,
    blender_executable: str = "blender",
    include_details: bool = False,
    solidify: bool = True,
    solidify_thickness: float = 0.04,
    depth_edge_threshold: float = 0.12,
    segmentation: str = "none",
    mask_path: str | Path | None = None,
    cleanup: bool = True,
    hole_fill_size: int = 12,
    spike_threshold: str = "balanced",
    collect_metrics: bool = True,
) -> StructuredSceneResult:
    memory_tracker = MemoryTracker().start() if collect_metrics else MemoryTracker()

    image = load_rgb_image(image_path)
    if depth_path:
        depth_map = load_depth_map(depth_path)
    else:
        warnings.warn(
            "Structured mode is using luminance fallback depth; plane detection may be weak.",
            stacklevel=2,
        )
        depth_map = derive_depth_from_luminance(image)

    if image.size != (len(depth_map[0]), len(depth_map)):
        raise ValueError(
            "Image and depth map dimensions must match. "
            f"Image is {image.size}, depth is {(len(depth_map[0]), len(depth_map))}."
        )

    segmentation_start = perf_counter()
    segmentation_mask = _load_segmentation_mask(
        segmentation,
        mask_path,
        depth_map=depth_map,
        expected_size=image.size,
    )
    segmentation_seconds = perf_counter() - segmentation_start

    scene_data, metrics_artifacts = _build_structured_scene_data_internal(
        depth_map,
        segmentation_mask=segmentation_mask,
        resolution=resolution,
        depth_strength=depth_strength,
        include_details=include_details,
        solidify=solidify,
        solidify_thickness=solidify_thickness,
        depth_edge_threshold=depth_edge_threshold,
        cleanup=cleanup,
        hole_fill_size=hole_fill_size,
        spike_threshold=spike_threshold,
    )

    runtime_breakdown = {
        "segmentation": segmentation_seconds,
        "region_build": metrics_artifacts.region_build_seconds,
        "mesh_build": metrics_artifacts.mesh_build_seconds,
        "export": 0.0,
    }

    blend_path = _as_blend_path(output_path)
    export_start = perf_counter()
    if keep_obj:
        obj_result = export_scene_obj(
            scene_data,
            blend_path.with_suffix(".obj"),
            texture_image=image if write_texture else None,
        )
        blend_result = export_blend_from_obj(
            obj_path=obj_result.obj_path,
            blend_path=blend_path,
            blender_executable=blender_executable,
        )
    else:
        with tempfile.TemporaryDirectory(prefix="sceneforge_structured_obj_") as temp_dir:
            obj_result = export_scene_obj(
                scene_data,
                Path(temp_dir) / f"{blend_path.stem}.obj",
                texture_image=image if write_texture else None,
            )
            blend_result = export_blend_from_obj(
                obj_path=obj_result.obj_path,
                blend_path=blend_path,
                blender_executable=blender_executable,
            )
    runtime_breakdown["export"] = perf_counter() - export_start

    if collect_metrics:
        peak_memory_bytes = memory_tracker.stop()
        metrics_payload = build_structured_scene_metrics_payload(
            runtime_seconds=runtime_breakdown,
            peak_memory_bytes=peak_memory_bytes,
            depth_map=depth_map,
            regions=metrics_artifacts.region_analysis,
            analysis_columns=metrics_artifacts.analysis_columns,
            analysis_rows=metrics_artifacts.analysis_rows,
            fallback_counts=metrics_artifacts.fallback_counts,
            cleanup_counts=metrics_artifacts.cleanup_counts,
            occlusion_gap_count=metrics_artifacts.occlusion_gap_count,
            occlusion_gap_area_proxy=metrics_artifacts.occlusion_gap_area_proxy,
            scene_before_cleanup=metrics_artifacts.scene_before_cleanup,
            scene_after_cleanup=metrics_artifacts.scene_after_cleanup,
        )
        write_structured_scene_metrics(
            output_path=blend_result.blend_path.parent / "metrics.json",
            payload=metrics_payload,
        )

    return StructuredSceneResult(
        blend_path=blend_result.blend_path,
        preview_path=blend_result.preview_path,
        obj_result=obj_result if keep_obj else None,
        scene_data=scene_data,
    )


def build_structured_scene_data(
    depth_map: list[list[float]],
    *,
    resolution: int = 64,
    depth_strength: float = 1.0,
    include_details: bool = False,
    solidify: bool = True,
    solidify_thickness: float = 0.04,
    depth_edge_threshold: float = 0.12,
    segmentation_mask: SegmentationMask | None = None,
    cleanup: bool = True,
    hole_fill_size: int = 12,
    spike_threshold: str = "balanced",
) -> StructuredSceneData:
    scene, _ = _build_structured_scene_data_internal(
        depth_map,
        resolution=resolution,
        depth_strength=depth_strength,
        include_details=include_details,
        solidify=solidify,
        solidify_thickness=solidify_thickness,
        depth_edge_threshold=depth_edge_threshold,
        segmentation_mask=segmentation_mask,
        cleanup=cleanup,
        hole_fill_size=hole_fill_size,
        spike_threshold=spike_threshold,
    )
    return scene


def _build_structured_scene_data_internal(
    depth_map: list[list[float]],
    *,
    resolution: int = 64,
    depth_strength: float = 1.0,
    include_details: bool = False,
    solidify: bool = True,
    solidify_thickness: float = 0.04,
    depth_edge_threshold: float = 0.12,
    segmentation_mask: SegmentationMask | None = None,
    cleanup: bool = True,
    hole_fill_size: int = 12,
    spike_threshold: str = "balanced",
) -> tuple[StructuredSceneData, _StructuredSceneBuildArtifacts]:
    source_rows = len(depth_map)
    source_columns = len(depth_map[0])
    aspect_ratio = source_columns / source_rows
    analysis_columns = max(2, min(resolution, source_columns))
    analysis_rows = max(2, min(round(analysis_columns * source_rows / source_columns), source_rows))

    region_start = perf_counter()
    if segmentation_mask is None:
        regions = analyze_depth_regions(
            depth_map,
            analysis_columns=analysis_columns,
            analysis_rows=analysis_rows,
            depth_bucket_size=0.04,
            min_plane_cells=min(
                max(12, analysis_columns // 2),
                max(4, analysis_columns * analysis_rows // 4),
            ),
        )
    else:
        segmentation_mask.validate_size(width=source_columns, height=source_rows)
        if cleanup:
            mask_result = cleanup_segmentation_mask(
                segmentation_mask,
                max_hole_cells=hole_fill_size,
            )
            segmentation_mask = mask_result.mask
            mask_cleanup_counts = {
                "filled_mask_holes": mask_result.filled_mask_holes,
                "removed_mask_islands": mask_result.removed_mask_islands,
            }
        else:
            mask_cleanup_counts = {
                "filled_mask_holes": 0,
                "removed_mask_islands": 0,
            }
        regions = segmentation_mask_to_regions(
            segmentation_mask,
            depth_map,
            analysis_columns=analysis_columns,
            analysis_rows=analysis_rows,
        )
    if segmentation_mask is None:
        mask_cleanup_counts = {
            "filled_mask_holes": 0,
            "removed_mask_islands": 0,
        }
    region_seconds = perf_counter() - region_start

    mesh_start = perf_counter()
    plane_parts = []
    detail_parts = []
    fallback_counts = {"primitive": 0, "base": 0}

    for region in regions:
        if region.kind == "plane":
            part_result = build_masked_plane_part_with_fallback(
                region,
                depth_map,
                analysis_columns=analysis_columns,
                analysis_rows=analysis_rows,
                depth_strength=depth_strength,
                aspect_ratio=aspect_ratio,
                depth_edge_threshold=depth_edge_threshold,
            )
            plane_parts.append(part_result.part)
            if part_result.used_plane_fallback:
                fallback_counts["primitive"] += 1
            if not part_result.part.faces:
                fallback_counts["base"] += 1
        elif include_details or segmentation_mask is not None:
            part = build_region_relief_part(
                region,
                depth_map,
                analysis_columns=analysis_columns,
                analysis_rows=analysis_rows,
                depth_strength=depth_strength,
                aspect_ratio=aspect_ratio,
                depth_edge_threshold=depth_edge_threshold,
            )
            detail_parts.append(part)
            if not part.faces:
                fallback_counts["base"] += 1

    if include_details:
        coverage_part = build_coverage_relief_part(
            depth_map,
            analysis_columns=analysis_columns,
            analysis_rows=analysis_rows,
            depth_strength=depth_strength,
            aspect_ratio=aspect_ratio,
            depth_edge_threshold=depth_edge_threshold * 1.5,
            depth_offset=solidify_thickness * 0.5 if solidify else 0.02,
        )
        detail_parts.append(coverage_part)
        if not coverage_part.faces:
            fallback_counts["base"] += 1

    scene_before_cleanup = StructuredSceneData(plane_parts=plane_parts, detail_parts=detail_parts)
    scene = scene_before_cleanup
    cleanup_counts = {
        "filled_mask_holes": mask_cleanup_counts["filled_mask_holes"],
        "removed_mask_islands": mask_cleanup_counts["removed_mask_islands"],
        "patched_mesh_holes": 0,
        "rejected_spikes": 0,
    }
    occlusion_gap_count = 0
    occlusion_gap_area_proxy = 0.0

    if cleanup:
        cleanup_result = cleanup_structured_scene(
            scene,
            hole_fill_size=hole_fill_size,
            spike_threshold=spike_threshold,
        )
        scene = cleanup_result.scene
        for key, count in cleanup_result.cleanup_counts.items():
            cleanup_counts[key] = cleanup_counts.get(key, 0) + count
        occlusion_gap_count = cleanup_result.occlusion_gap_count
        occlusion_gap_area_proxy = cleanup_result.occlusion_gap_area_proxy

    scene_after_cleanup = scene

    if solidify:
        scene = solidify_scene(
            scene,
            plane_thickness=solidify_thickness,
            detail_thickness=solidify_thickness * 0.5,
        )

    scene = with_scene_normals(scene)
    mesh_seconds = perf_counter() - mesh_start

    return scene, _StructuredSceneBuildArtifacts(
        region_analysis=regions,
        region_build_seconds=region_seconds,
        mesh_build_seconds=mesh_seconds,
        analysis_columns=analysis_columns,
        analysis_rows=analysis_rows,
        fallback_counts=fallback_counts,
        cleanup_counts=cleanup_counts,
        occlusion_gap_count=occlusion_gap_count,
        occlusion_gap_area_proxy=occlusion_gap_area_proxy,
        scene_before_cleanup=scene_before_cleanup,
        scene_after_cleanup=scene_after_cleanup,
    )


def _load_segmentation_mask(
    segmentation: str,
    mask_path: str | Path | None,
    *,
    depth_map: list[list[float]],
    expected_size: tuple[int, int],
) -> SegmentationMask | None:
    if segmentation == "none":
        if mask_path is not None:
            raise ValueError("`--mask` requires `--segmentation mask`.")
        return None
    if segmentation == "mask":
        if mask_path is None:
            raise ValueError("`--segmentation mask` requires `--mask PATH`.")
        mask = load_segmentation_mask(mask_path)
    elif segmentation == "auto":
        if mask_path is not None:
            raise ValueError("`--mask` cannot be used with `--segmentation auto`.")
        mask = build_heuristic_segmentation(depth_map)
    else:
        raise ValueError(f"Unsupported segmentation mode: {segmentation}")

    mask.validate_size(width=expected_size[0], height=expected_size[1])
    return mask
