from __future__ import annotations

from dataclasses import dataclass

from Core.Types.scene_data import SceneMeshPart, StructuredSceneData
from Geometry.Cleanup.mesh_hole_patcher import patch_small_mesh_holes
from Geometry.Cleanup.spike_filter import filter_spike_faces


@dataclass(frozen=True)
class CleanupResult:
    scene: StructuredSceneData
    cleanup_counts: dict[str, int]
    occlusion_gap_count: int
    occlusion_gap_area_proxy: float


def cleanup_structured_scene(
    scene: StructuredSceneData,
    *,
    hole_fill_size: int = 12,
    spike_threshold: str = "balanced",
) -> CleanupResult:
    cleanup_counts = {
        "filled_mask_holes": 0,
        "removed_mask_islands": 0,
        "patched_mesh_holes": 0,
        "rejected_spikes": 0,
    }
    occlusion_gap_count = 0
    occlusion_gap_area_proxy = 0.0

    plane_parts = []
    for part in scene.plane_parts:
        cleaned, counts, gaps, gap_area = _cleanup_part(
            part,
            hole_fill_size=hole_fill_size,
            spike_threshold=spike_threshold,
        )
        plane_parts.append(cleaned)
        cleanup_counts["patched_mesh_holes"] += counts["patched_mesh_holes"]
        cleanup_counts["rejected_spikes"] += counts["rejected_spikes"]
        occlusion_gap_count += gaps
        occlusion_gap_area_proxy += gap_area

    detail_parts = []
    for part in scene.detail_parts:
        cleaned, counts, gaps, gap_area = _cleanup_part(
            part,
            hole_fill_size=hole_fill_size,
            spike_threshold=spike_threshold,
        )
        detail_parts.append(cleaned)
        cleanup_counts["patched_mesh_holes"] += counts["patched_mesh_holes"]
        cleanup_counts["rejected_spikes"] += counts["rejected_spikes"]
        occlusion_gap_count += gaps
        occlusion_gap_area_proxy += gap_area

    return CleanupResult(
        scene=StructuredSceneData(plane_parts=plane_parts, detail_parts=detail_parts),
        cleanup_counts=cleanup_counts,
        occlusion_gap_count=occlusion_gap_count,
        occlusion_gap_area_proxy=occlusion_gap_area_proxy,
    )


def _cleanup_part(
    part: SceneMeshPart,
    *,
    hole_fill_size: int,
    spike_threshold: str,
) -> tuple[SceneMeshPart, dict[str, int], int, float]:
    filtered, rejected = filter_spike_faces(part, threshold=spike_threshold)
    patched, patched_holes, large_gaps = patch_small_mesh_holes(
        filtered,
        max_boundary_edges=hole_fill_size,
    )
    return (
        patched,
        {
            "patched_mesh_holes": patched_holes,
            "rejected_spikes": rejected,
        },
        large_gaps,
        float(large_gaps),
    )
