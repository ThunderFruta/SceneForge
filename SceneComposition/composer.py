from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from SceneGeometry.VGGT.pipeline import scene_point_to_gltf_vertex
from SceneGeometry.coordinate_contract import DEFAULT_FOV_DEGREES


SCHEMA_VERSION = 1
TABLE_SUPPORT_LABELS = ("table", "desk", "counter")
TABLETOP_OBJECT_LABELS = ("vase", "flower", "plant", "lamp", "book", "bowl", "cup", "glass", "plate", "pot")
PROJECTION_VERTICAL_EDGE_REJECT_RATIO = 0.35
PROJECTION_OCCLUDED_BOTTOM_EDGE_REVIEW_RATIO = 0.65
PROJECTION_OCCLUDED_BOTTOM_TOP_EDGE_REVIEW_RATIO = 0.40
PROJECTION_HORIZONTAL_EDGE_REJECT_RATIO = 0.55
PROJECTION_CENTER_REJECT_RATIO = 0.35
PROJECTION_CENTER_Y_REJECT_RATIO = 0.20
PROJECTION_OCCLUDED_BOTTOM_AREA_REJECT_RATIO = 1.60
PROJECTION_AREA_ERROR_REJECT_RATIO = 0.35
VGGT_CANDIDATE_POINT_SAMPLE_COUNT = 512
VGGT_POINT_MATCH_MESH_SAMPLE_COUNT = 384
VGGT_POINT_MATCH_TARGET_SAMPLE_COUNT = 384
PROJECTED_VERTEX_SAMPLE_COUNT = 2048
VGGT_CANDIDATE_LOSS_WEIGHT = 1.15
VGGT_POINT_MATCH_LOSS_WEIGHT = 0.42
VGGT_YAW_PRIOR_WEIGHT = 0.16
MESH_FACING_PRIOR_WEIGHT = 0.95
REPEATED_INSTANCE_SIZE_PRIOR_WEIGHT = 4.0
PHYSICAL_SIZE_PRIOR_CANDIDATE_REJECT_LOSS = 0.35
OBJECT_AVOIDANCE_PRIOR_WEIGHT = 4.0
MASK_CANDIDATE_LOSS_WEIGHT = 0.45
MASK_CANDIDATE_POOL_SIZE = 48
MASK_CANDIDATE_RENDER_MAX_SIZE = 320
MASK_CANDIDATE_FACE_SAMPLE_COUNT = 12000
EVIDENCE_SCALE_NEIGHBORS = (0.92, 1.0, 1.08)
MIN_EVIDENCE_SCALE_CANDIDATE = 0.25
MAX_EVIDENCE_SCALE_CANDIDATE = 3.0
DEFAULT_UNIFORM_SCALE_CANDIDATES = (0.50, 0.60, 0.70, 0.82, 0.92, 1.0, 1.08, 1.25, 1.40)
DEFAULT_TRANSLATION_X_CANDIDATES = (-0.16, -0.08, -0.04, 0.0, 0.04, 0.08, 0.16)
DEFAULT_TRANSLATION_Z_CANDIDATES = (-0.56, -0.44, -0.32, -0.24, -0.16, -0.08, -0.04, 0.0, 0.04, 0.08, 0.16)
PROJECTION_TRANSLATION_STEP_RATIO = 0.16
PROJECTION_TRANSLATION_MIN_STEP = 0.04
PROJECTION_TRANSLATION_MAX_STEP = 0.12
PROJECTION_TRANSLATION_NEIGHBORS = (1.0,)
DEFAULT_YAW_CANDIDATES = (
    -float(np.pi),
    -float(np.pi) * 0.75,
    -float(np.pi) / 2.0,
    -float(np.pi) / 4.0,
    -0.50,
    -0.25,
    0.0,
    0.25,
    0.50,
    float(np.pi) / 4.0,
    float(np.pi) / 2.0,
    float(np.pi) * 0.75,
    float(np.pi),
)
LARGE_TARGET_HEIGHT_RATIO = 0.45
LARGE_TARGET_MIN_UNIFORM_SCALE = 0.70
SUPPORT_CONTACT_FLOOR_QUANTILES = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 35.0)
SUPPORT_CONTACT_TABLETOP_QUANTILES = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 35.0, 50.0)
SUPPORT_CONTACT_FALLBACK_QUANTILE = 20.0
SUPPORT_CONTACT_EXCELLENT_AREA_RATIO = 0.65
SUPPORT_CONTACT_MIN_AREA_RATIO = 0.10
SUPPORT_CONTACT_MIN_SPAN_RATIO = 0.12
SUPPORT_CONTACT_MIN_VERTEX_RATIO = 0.002
TABLETOP_CONTACT_MIN_VERTEX_RATIO = 0.15


def compose_scene(
    *,
    background_path: str | Path,
    objects_dir: str | Path,
    object_geometry_path: str | Path,
    placements_path: str | Path | None = None,
    output_dir: str | Path,
    output_name: str = "scene.glb",
    object_mesh_name: str = "hunyuan3d_textured.glb",
    include_review: bool = False,
    scale_mode: str = "fit-box",
    placement_orientation: str = "upright",
    object_scale_factor: float = 0.85,
    background_fit: str = "camera-clipped",
    background_margin: float = 1.0,
    background_depth_offset: float = 0.12,
    background_vggt_dir: str | Path | None = None,
    background_stride: int = 16,
    clip_background_masks: bool = False,
    background_clip_dilation_px: int = 8,
    snap_objects_to_floor: bool = True,
    optimize_placements: bool = True,
    source_image_path: str | Path | None = None,
) -> dict[str, Any]:
    if scale_mode != "fit-box":
        raise ValueError(f"Unsupported scale mode: {scale_mode}")
    if placement_orientation not in {"upright", "obb"}:
        raise ValueError(f"Unsupported placement orientation: {placement_orientation}")
    if object_scale_factor <= 0:
        raise ValueError("object_scale_factor must be positive")
    if background_fit not in {"room-corner", "camera-clipped", "placement-bounds", "raw"}:
        raise ValueError(f"Unsupported background fit mode: {background_fit}")
    if background_margin <= 0:
        raise ValueError("background_margin must be positive")
    if background_depth_offset < 0:
        raise ValueError("background_depth_offset must be non-negative")

    background_path = Path(background_path)
    objects_dir = Path(objects_dir)
    object_geometry_path = Path(object_geometry_path)
    placements_path = Path(placements_path) if placements_path is not None else None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    background_vggt_dir = Path(background_vggt_dir) if background_vggt_dir is not None else infer_background_vggt_dir(background_path)

    geometry = load_json(object_geometry_path)
    explicit_placements = load_json(placements_path) if placements_path is not None else None
    placement_source = "object_placements_json" if explicit_placements is not None else "object_geometry_json"
    coordinate_contract = (explicit_placements or {}).get("coordinate_contract") or geometry.get("coordinate_contract")
    source_report = explicit_placements if explicit_placements is not None else geometry
    source_image_path = Path(source_image_path) if source_image_path is not None else infer_source_image_path(source_report)
    object_dirs = index_object_dirs(objects_dir)
    placements = (explicit_placements or geometry).get("objects", [])
    scene = new_scene()
    if explicit_placements is not None:
        placement_bounds = explicit_placement_bounds_gltf(placements)
    else:
        placement_bounds = placement_bounds_gltf(
            placements,
            placement_orientation=placement_orientation,
            object_scale_factor=object_scale_factor,
        )
    if background_fit == "room-corner" and placement_bounds is not None:
        plane_texture_path = room_corner_plane_texture_path(background_vggt_dir)
        background_stats = add_room_corner_background(
            scene,
            placement_bounds=placement_bounds,
            margin=background_margin,
            depth_offset=background_depth_offset,
            texture_image_path=plane_texture_path,
            coordinate_contract=coordinate_contract,
        )
    elif background_fit == "camera-clipped" and background_vggt_dir is not None:
        background_stats = add_vggt_background_mesh(
            scene,
            vggt_dir=background_vggt_dir,
            objects_dir=objects_dir,
            object_dirs=object_dirs,
            stride=background_stride,
            clip_masks=clip_background_masks,
            clip_dilation_px=background_clip_dilation_px,
            placement_bounds=placement_bounds,
            margin=background_margin,
            depth_offset=background_depth_offset,
            coordinate_contract=coordinate_contract,
        )
    else:
        background_transform = None
        if background_fit == "placement-bounds" and placement_bounds is not None:
            background_transform = background_fit_transform(
                source_bounds=combined_bounds(load_meshes(background_path)),
                placement_bounds=placement_bounds,
                margin=background_margin,
                depth_offset=background_depth_offset,
            )
        background_stats = add_scene_asset(
            scene,
            background_path,
            name_prefix="background",
            transform=background_transform,
            normalize=False,
        )

    floor_y = background_floor_y(background_stats) if snap_objects_to_floor else None
    if explicit_placements is not None:
        support_targets = explicit_support_targets(
            placements,
            floor_y=floor_y,
            object_dirs=object_dirs,
            object_mesh_name=object_mesh_name,
            include_review=include_review,
        )
    else:
        support_targets = object_support_targets(
            placements,
            object_dirs=object_dirs,
            object_mesh_name=object_mesh_name,
            include_review=include_review,
            placement_orientation=placement_orientation,
            object_scale_factor=object_scale_factor,
            floor_y=floor_y,
        )
    records: list[dict[str, Any]] = []
    for placement in placements:
        detection_id = int(placement.get("detection_id", 0))
        if explicit_placements is not None:
            record = compose_explicit_placement_record(
                scene=scene,
                placement=placement,
                object_dirs=object_dirs,
                object_mesh_name=object_mesh_name,
                include_review=include_review,
                support_target=support_targets.get(detection_id),
                room_bounds=np.asarray(background_stats.get("transformed_bounds"), dtype=np.float64),
            )
        else:
            record = compose_object_record(
                scene=scene,
                placement=placement,
                object_dirs=object_dirs,
                object_mesh_name=object_mesh_name,
                include_review=include_review,
                placement_orientation=placement_orientation,
                object_scale_factor=object_scale_factor,
                support_target=support_targets.get(detection_id),
                coordinate_contract=coordinate_contract,
                optimize_placements=optimize_placements,
            )
        records.append(record)
    suppressed_objects = suppressed_objects_report(records)
    object_overlap_warnings = object_overlap_warnings_report(records)

    scene_path = output_dir / safe_output_name(output_name)
    scene.export(scene_path)
    overlay_path = output_dir / "input_vs_projection_overlay.png"
    report = {
        "schema_version": SCHEMA_VERSION,
        "background_path": str(background_path),
        "objects_dir": str(objects_dir),
        "object_geometry_path": str(object_geometry_path),
        "placements_path": str(placements_path) if placements_path is not None else None,
        "placement_source": placement_source,
        "artifacts": {
            "scene_glb": str(scene_path),
            "scene_alignment": str(output_dir / "scene_alignment.json"),
            "input_vs_projection_overlay": str(overlay_path) if source_image_path is not None else None,
        },
        "coordinate_contract": coordinate_contract,
        "scale_mode": scale_mode,
        "placement_orientation": placement_orientation,
        "object_scale_factor": float(object_scale_factor),
        "background_fit": background_fit,
        "background_margin": float(background_margin),
        "background_depth_offset": float(background_depth_offset),
        "background_vggt_dir": str(background_vggt_dir) if background_vggt_dir is not None else None,
        "background_stride": int(background_stride),
        "clip_background_masks": bool(clip_background_masks),
        "background_clip_dilation_px": int(background_clip_dilation_px),
        "snap_objects_to_floor": bool(snap_objects_to_floor),
        "optimize_placements": bool(optimize_placements),
        "source_image_path": str(source_image_path) if source_image_path is not None else None,
        "floor_y": float(floor_y) if floor_y is not None else None,
        "support_targets": support_targets,
        "object_mesh_name": object_mesh_name,
        "include_review": bool(include_review),
        "background": background_stats,
        "objects": records,
        "suppressed_objects": suppressed_objects,
        "object_overlap_warnings": object_overlap_warnings,
        "projection_quality": projection_quality_summary(records),
        "summary": {
            "placement_count": len(records),
            "composed_count": sum(1 for item in records if item["status"] == "composed"),
            "skipped_count": sum(1 for item in records if item["status"] == "skipped"),
            "failed_count": sum(1 for item in records if item["status"] == "failed"),
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if source_image_path is not None:
        write_projection_overlay(source_image_path, records, overlay_path)
    (output_dir / "scene_alignment.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def explicit_placement_bounds_gltf(placements: list[dict[str, Any]]) -> np.ndarray | None:
    bounds: list[np.ndarray] = []
    for placement in placements:
        if placement.get("status") != "accepted":
            continue
        transformed = bounds_array(placement.get("transformed_bounds"))
        if transformed is not None:
            bounds.append(transformed)
    if not bounds:
        return None
    return merge_bounds(bounds)


def explicit_support_targets(
    placements: list[dict[str, Any]],
    *,
    floor_y: float | None = None,
    object_dirs: dict[int, Path] | None = None,
    object_mesh_name: str = "hunyuan3d_textured.glb",
    include_review: bool = False,
) -> dict[int, dict[str, Any]]:
    targets: dict[int, dict[str, Any]] = {}
    for placement in placements:
        detection_id = int(placement.get("detection_id", 0))
        support = placement.get("support") or {}
        support_kind = normalized_support_kind(support.get("support_kind")) or support_kind_from_mode(support.get("mode"))
        support_y = support.get("support_y_gltf")
        if support_kind == "floor" and floor_y is not None:
            support_y = floor_y
        targets[detection_id] = {
            "support_kind": support_kind,
            "support_detection_id": support.get("support_detection_id"),
            "support_y": float(support_y) if support_y is not None else None,
            "support_plane_id": support.get("support_plane_id"),
            "support_label": support.get("support_label"),
            "support_confidence": support.get("support_confidence"),
        }
    support_tops = explicit_support_object_tops(
        placements,
        targets=targets,
        object_dirs=object_dirs or {},
        object_mesh_name=object_mesh_name,
        include_review=include_review,
    )
    for target in targets.values():
        if target.get("support_kind") != "tabletop":
            continue
        support_detection_id = target.get("support_detection_id")
        if support_detection_id in support_tops:
            target["support_y"] = support_tops[support_detection_id]
    return targets


def explicit_support_object_tops(
    placements: list[dict[str, Any]],
    *,
    targets: dict[int, dict[str, Any]],
    object_dirs: dict[int, Path],
    object_mesh_name: str,
    include_review: bool,
) -> dict[int, float]:
    support_tops: dict[int, float] = {}
    for placement in placements:
        if placement.get("status") != "accepted":
            continue
        if placement.get("suppressed_by_composite"):
            continue
        if bool(placement.get("needs_review", False)) and not include_review:
            continue
        detection_id = int(placement.get("detection_id", 0))
        label = str(placement.get("detector_label") or "")
        if not is_table_support_label(label):
            continue
        target = targets.get(detection_id)
        if not target or target.get("support_y") is None:
            continue
        mesh_path = explicit_placement_mesh_path(placement, object_dirs, object_mesh_name)
        if mesh_path is None:
            continue
        try:
            transform = np.asarray(placement.get("transform_gltf"), dtype=np.float64)
            if transform.shape != (4, 4) or not np.isfinite(transform).all():
                continue
            snapped, _delta = snap_transform_to_support(
                mesh_path,
                transform,
                float(target["support_y"]),
                support_kind=target.get("support_kind"),
            )
            support_tops[detection_id] = float(transformed_mesh_bounds(mesh_path, snapped)[1, 1])
        except Exception:
            continue
    return support_tops


def explicit_placement_mesh_path(
    placement: dict[str, Any],
    object_dirs: dict[int, Path],
    object_mesh_name: str,
) -> Path | None:
    mesh_path = Path(str(placement.get("mesh_path"))) if placement.get("mesh_path") else None
    if mesh_path is not None and mesh_path.is_file():
        return mesh_path
    detection_id = int(placement.get("detection_id", 0))
    object_dir = object_dirs.get(int(placement.get("source_object_dir_id") or detection_id))
    return resolve_object_mesh_path(object_dir, object_mesh_name) if object_dir else None


def compose_explicit_placement_record(
    *,
    scene: Any,
    placement: dict[str, Any],
    object_dirs: dict[int, Path],
    object_mesh_name: str,
    include_review: bool,
    support_target: dict[str, Any] | None = None,
    room_bounds: np.ndarray | None = None,
) -> dict[str, Any]:
    detection_id = int(placement.get("detection_id", 0))
    label = str(placement.get("detector_label") or "object")
    support = placement.get("support") or {}
    support_record = support_target or {
        "support_kind": normalized_support_kind(support.get("support_kind")) or support_kind_from_mode(support.get("mode")),
        "support_detection_id": support.get("support_detection_id"),
        "support_y": support.get("support_y_gltf"),
        "support_plane_id": support.get("support_plane_id"),
        "support_label": support.get("support_label"),
        "support_confidence": support.get("support_confidence"),
        "mode": support.get("mode"),
    }
    support_kind = support_record.get("support_kind") or support_kind_from_mode(support.get("mode"))
    base = {
        "detection_id": detection_id,
        "detector_label": placement.get("detector_label"),
        "box_type": None,
        "needs_review": bool(placement.get("needs_review", False)),
        "relation_role": placement.get("relation_role") or "primary",
        "composite_id": placement.get("composite_id"),
        "suppressed_by_composite": placement.get("suppressed_by_composite"),
        "source_detection_ids": placement.get("source_detection_ids"),
        "source_object_dir_id": placement.get("source_object_dir_id"),
        "object_dir": str(object_dirs[detection_id]) if detection_id in object_dirs else None,
        "object_mesh": placement.get("mesh_path"),
        "status": "skipped",
        "reason": None,
        "placement_source": "object_placements_json",
        "placement_status": placement.get("status"),
        "transform_gltf": placement.get("transform_gltf"),
        "support_kind": support_kind,
        "support_detection_id": support_record.get("support_detection_id"),
        "support_y": support_record.get("support_y"),
        "support_plane_id": support_record.get("support_plane_id"),
        "support_label": support_record.get("support_label"),
        "support_confidence": support_record.get("support_confidence"),
        "support_degrees_of_freedom": support_degrees_of_freedom(support_record),
        "render_to_input_optimization": placement.get("render_to_input_optimization"),
        "projection_quality": (placement.get("render_to_input_optimization") or {}).get("projection_quality"),
        "losses": placement.get("losses"),
        "placement_quality": placement.get("quality"),
    }
    if placement.get("suppressed_by_composite"):
        base.update(reason="suppressed_by_composite")
        return base
    if placement.get("status") != "accepted":
        base.update(reason=placement.get("reason") or "placement_not_accepted")
        return base
    if base["needs_review"] and not include_review:
        base.update(reason="needs_review")
        return base

    mesh_path = Path(str(placement.get("mesh_path"))) if placement.get("mesh_path") else None
    if mesh_path is None or not mesh_path.is_file():
        object_dir = object_dirs.get(int(placement.get("source_object_dir_id") or detection_id))
        mesh_path = resolve_object_mesh_path(object_dir, object_mesh_name) if object_dir else None
    if mesh_path is None or not mesh_path.is_file():
        base.update(status="failed", reason="missing_object_mesh", object_mesh=None)
        return base

    try:
        transform = np.asarray(placement.get("transform_gltf"), dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("transform_gltf must be a finite 4x4 matrix")
        support_snap_delta = 0.0
        if support_record.get("support_y") is not None:
            support_y = float(support_record["support_y"])
            transform, support_snap_delta = snap_transform_to_support(
                mesh_path,
                transform,
                support_y,
                support_kind=support_kind,
            )
        room_boundary_adjustment = room_boundary_adjustment_report(mesh_path, transform, room_bounds)
        transform = np.asarray(room_boundary_adjustment["transform_gltf"], dtype=np.float64)
        object_stats = add_scene_asset(
            scene,
            mesh_path,
            name_prefix=f"object_{detection_id:02d}_{slugify(label)}",
            transform=transform,
        )
        support_contact = mesh_support_contact_report(mesh_path, transform, support_kind)
    except Exception as exc:
        base.update(status="failed", reason=f"composition_failed: {exc}")
        return base

    base.update(
        object_mesh=str(mesh_path),
        status="composed",
        reason=None,
        transform_gltf=transform.tolist(),
        floor_snap_delta=float(support_snap_delta if support_kind == "floor" else 0.0),
        support_snap_delta=float(support_snap_delta),
        room_boundary_adjustment=room_boundary_adjustment,
        support_contact=support_contact,
        source_bounds=object_stats["source_bounds"],
        transformed_bounds=object_stats["transformed_bounds"],
        mesh_cleanup=object_stats.get("mesh_cleanup"),
    )
    return base


def support_kind_from_mode(mode: Any) -> str | None:
    return normalized_support_kind(mode)


def normalized_support_kind(mode: Any) -> str | None:
    mode_text = str(mode or "").lower()
    if mode_text.startswith("floor"):
        return "floor"
    if mode_text.startswith("tabletop"):
        return "tabletop"
    if mode_text.startswith("wall"):
        return "wall"
    if mode_text.startswith("ceiling"):
        return "ceiling"
    if mode_text.startswith("unknown"):
        return "unknown"
    return None


def compose_object_record(
    *,
    scene: Any,
    placement: dict[str, Any],
    object_dirs: dict[int, Path],
    object_mesh_name: str,
    include_review: bool,
    placement_orientation: str,
    object_scale_factor: float,
    support_target: dict[str, Any] | None,
    coordinate_contract: dict[str, Any] | None = None,
    optimize_placements: bool = True,
) -> dict[str, Any]:
    detection_id = int(placement.get("detection_id", 0))
    object_dir_id = int(placement.get("source_object_dir_id") or detection_id)
    label = str(placement.get("detector_label") or "object")
    base = {
        "detection_id": detection_id,
        "detector_label": placement.get("detector_label"),
        "box_type": placement.get("box_type"),
        "needs_review": bool(placement.get("needs_review", False)),
        "relation_role": placement.get("relation_role") or "primary",
        "composite_id": placement.get("composite_id"),
        "suppressed_by_composite": placement.get("suppressed_by_composite"),
        "source_detection_ids": placement.get("source_detection_ids"),
        "source_object_dir_id": object_dir_id if object_dir_id != detection_id else None,
        "object_dir": str(object_dirs[object_dir_id]) if object_dir_id in object_dirs else None,
        "object_mesh": None,
        "status": "skipped",
        "reason": None,
        "transform_gltf": None,
        "support_kind": support_target.get("support_kind") if support_target else None,
        "support_detection_id": support_target.get("support_detection_id") if support_target else None,
        "support_y": support_target.get("support_y") if support_target else None,
        "support_degrees_of_freedom": support_degrees_of_freedom(support_target),
        "render_to_input_optimization": None,
        "projection_quality": None,
    }
    if placement.get("suppressed_by_composite"):
        base.update(reason="suppressed_by_composite")
        return base
    if placement.get("box_type") == "failed":
        base.update(reason=placement.get("failure_reason") or "placement_failed")
        return base
    if base["needs_review"] and not include_review:
        base.update(reason="needs_review")
        return base
    object_dir = object_dirs.get(object_dir_id)
    if object_dir is None:
        base.update(status="failed", reason="missing_object_dir")
        return base
    mesh_path = resolve_object_mesh_path(object_dir, object_mesh_name)
    if mesh_path is None:
        base.update(status="failed", reason="missing_object_mesh")
        return base

    try:
        transform = placement_transform_to_gltf(
            placement,
            placement_orientation=placement_orientation,
            object_scale_factor=object_scale_factor,
        )
        support_snap_delta = 0.0
        snapped_support_y = None
        if support_target is not None and support_target.get("support_y") is not None:
            snapped_support_y = float(support_target["support_y"])
            transform, support_snap_delta = snap_transform_to_support(
                mesh_path,
                transform,
                snapped_support_y,
                support_kind=base["support_kind"],
            )
        optimization = optimize_transform_to_input(
            mesh_path=mesh_path,
            placement=placement,
            transform=transform,
            support_target=support_target,
            coordinate_contract=coordinate_contract,
            enabled=optimize_placements,
        )
        transform = optimization["transform"]
        if snapped_support_y is not None:
            transform, final_snap_delta = snap_transform_to_support(
                mesh_path,
                transform,
                snapped_support_y,
                support_kind=base["support_kind"],
            )
            if support_snap_delta == 0.0:
                support_snap_delta = final_snap_delta
        projection_quality = optimization["report"].get("projection_quality")
        if projection_quality and projection_quality.get("status") == "rejected":
            base["needs_review"] = True
        object_stats = add_scene_asset(
            scene,
            mesh_path,
            name_prefix=f"object_{detection_id:02d}_{slugify(label)}",
            transform=transform,
        )
        support_contact = mesh_support_contact_report(mesh_path, transform, base["support_kind"])
    except Exception as exc:
        base.update(status="failed", reason=f"composition_failed: {exc}")
        return base

    base.update(
        object_mesh=str(mesh_path),
        status="composed",
        reason=None,
        transform_gltf=np.asarray(transform, dtype=float).tolist(),
        floor_snap_delta=float(support_snap_delta if base["support_kind"] == "floor" else 0.0),
        support_snap_delta=float(support_snap_delta),
        support_contact=support_contact,
        render_to_input_optimization=optimization["report"],
        projection_quality=optimization["report"].get("projection_quality"),
        source_bounds=object_stats["source_bounds"],
        transformed_bounds=object_stats["transformed_bounds"],
        mesh_cleanup=object_stats.get("mesh_cleanup"),
    )
    return base


def support_degrees_of_freedom(support_target: dict[str, Any] | None) -> dict[str, Any] | None:
    if support_target is None or support_target.get("support_y") is None:
        return None
    support_kind = str(support_target.get("support_kind") or "support")
    return {
        "model": "support_plane_4dof",
        "support_kind": support_kind,
        "locked_vertical_axis": "gltf_y",
        "locked_support_y": float(support_target["support_y"]),
        "free_parameters": ["plane_x", "plane_z", "yaw_y", "uniform_scale"],
    }


def optimize_transform_to_input(
    *,
    mesh_path: Path,
    placement: dict[str, Any],
    transform: np.ndarray,
    support_target: dict[str, Any] | None,
    coordinate_contract: dict[str, Any] | None,
    enabled: bool,
    facing_target_gltf: Any = None,
    physical_size_target_extent_gltf: Any = None,
    avoid_bounds_gltf: Any = None,
) -> dict[str, Any]:
    target_bbox = bbox_array(placement.get("bbox_xyxy"))
    meshes = load_meshes(mesh_path)
    source_bounds = combined_bounds(meshes)
    projection_vertices = sample_mesh_vertices_for_projection(meshes, PROJECTED_VERTEX_SAMPLE_COUNT)
    initial_bbox = projected_mesh_sample_bbox(projection_vertices, source_bounds, transform, coordinate_contract)
    if initial_bbox is None:
        initial_bbox = projected_transform_bbox(source_bounds, transform, coordinate_contract)
    vggt_fit = load_vggt_candidate_fit(placement)
    initial_vggt = vggt_candidate_transform_loss(vggt_fit, source_bounds, transform)
    vggt_loss_weight = VGGT_CANDIDATE_LOSS_WEIGHT if vggt_fit.get("available") else 0.0
    yaw_prior = vggt_yaw_prior_from_placement(placement)
    initial_yaw = transform_yaw_gltf(transform)
    initial_yaw_prior_loss = yaw_prior_loss(yaw_prior, initial_yaw)
    facing_prior = mesh_facing_prior_from_target(meshes=meshes, transform=transform, facing_target_gltf=facing_target_gltf)
    initial_facing_prior_loss = facing_prior_loss(facing_prior, transform)
    size_prior = physical_size_prior_from_target(
        source_bounds=source_bounds,
        transform=transform,
        target_extent_gltf=physical_size_target_extent_gltf,
    )
    initial_size_prior_loss = physical_size_prior_loss(size_prior, source_bounds, transform)
    avoidance = object_avoidance_prior_from_bounds(avoid_bounds_gltf)
    initial_avoidance_loss = object_avoidance_prior_loss(avoidance, source_bounds, transform)
    mask_fit = load_mask_candidate_fit(placement, coordinate_contract)
    mask_loss_weight = MASK_CANDIDATE_LOSS_WEIGHT if mask_fit.get("available") else 0.0
    initial_mask = mask_candidate_transform_loss(
        mask_fit,
        meshes=meshes,
        source_bounds=source_bounds,
        transform=transform,
    )
    base_report = {
        "enabled": bool(enabled),
        "method": "support_plane_discrete_render_proxy_v1",
        "target_bbox_xyxy": target_bbox.tolist() if target_bbox is not None else None,
        "initial_projected_bbox_xyxy": initial_bbox.tolist() if initial_bbox is not None else None,
        "optimized_projected_bbox_xyxy": initial_bbox.tolist() if initial_bbox is not None else None,
        "candidate_projected_bbox_xyxy": None,
        "initial_loss": None,
        "optimized_loss": None,
        "candidate_loss": None,
        "initial_bbox_loss": None,
        "optimized_bbox_loss": None,
        "candidate_bbox_loss": None,
        "vggt_candidate_fit": vggt_candidate_fit_report(
            vggt_fit,
            loss_weight=vggt_loss_weight,
            initial=initial_vggt,
            optimized=initial_vggt,
            candidate=initial_vggt,
        ),
        "vggt_yaw_prior": vggt_yaw_prior_report(
            yaw_prior,
            loss_weight=VGGT_YAW_PRIOR_WEIGHT,
            initial_yaw=initial_yaw,
            optimized_yaw=initial_yaw,
            candidate_yaw=initial_yaw,
            initial_loss=initial_yaw_prior_loss,
            optimized_loss=initial_yaw_prior_loss,
            candidate_loss=initial_yaw_prior_loss,
        ),
        "mesh_facing_prior": mesh_facing_prior_report(
            facing_prior,
            loss_weight=MESH_FACING_PRIOR_WEIGHT,
            initial_transform=transform,
            optimized_transform=transform,
            candidate_transform_value=transform,
            initial_loss=initial_facing_prior_loss,
            optimized_loss=initial_facing_prior_loss,
            candidate_loss=initial_facing_prior_loss,
        ),
        "physical_size_prior": physical_size_prior_report(
            size_prior,
            initial=initial_size_prior_loss,
            optimized=initial_size_prior_loss,
            candidate=initial_size_prior_loss,
        ),
        "object_avoidance_prior": object_avoidance_prior_report(
            avoidance,
            initial=initial_avoidance_loss,
            optimized=initial_avoidance_loss,
            candidate=initial_avoidance_loss,
        ),
        "mask_candidate_fit": mask_candidate_fit_report(
            mask_fit,
            loss_weight=mask_loss_weight,
            initial=initial_mask,
            optimized=initial_mask,
            candidate=initial_mask,
            candidate_count=0,
            selected_yaw=0.0,
            selected_scale=1.0,
            fallback_reason="search_not_run",
        ),
        "delta_gltf": [0.0, 0.0, 0.0],
        "yaw_delta_radians": 0.0,
        "uniform_scale_delta": 1.0,
        "scale_candidates": list(DEFAULT_UNIFORM_SCALE_CANDIDATES),
        "minimum_scale_delta": min(DEFAULT_UNIFORM_SCALE_CANDIDATES),
        "minimum_scale_reason": "default",
        "candidate_count": 0,
        "orientation_search": empty_orientation_search_report(),
        "projection_quality": projection_quality_report(initial_bbox, target_bbox, accepted=True),
    }
    if not enabled or target_bbox is None or initial_bbox is None or support_target is None or support_target.get("support_y") is None:
        return {"transform": transform, "report": base_report}

    support_y = float(support_target["support_y"])
    allow_occluded_bottom = bool((support_target or {}).get("support_kind") == "floor")
    initial_bbox_loss = bbox_projection_loss(initial_bbox, target_bbox)
    initial_support_loss = support_penalty(source_bounds, transform, support_y)
    initial_loss = objective_transform_loss(
        bbox_loss=initial_bbox_loss,
        support_loss=initial_support_loss,
        scale_loss=0.0,
        vggt_loss=initial_vggt.get("loss"),
        vggt_loss_weight=vggt_loss_weight,
        yaw_prior_loss=initial_yaw_prior_loss,
        facing_prior_loss=initial_facing_prior_loss,
        physical_size_prior_loss=initial_size_prior_loss,
        object_avoidance_prior_loss=initial_avoidance_loss,
    )
    initial_quality = projection_quality_report(initial_bbox, target_bbox, allow_occluded_bottom=allow_occluded_bottom)
    scale_candidates, scale_floor_report = scale_candidates_for_target(
        target_bbox,
        coordinate_contract,
        initial_bbox=initial_bbox,
        vggt_fit=vggt_fit,
        source_bounds=source_bounds,
        transform=transform,
        physical_size_prior=size_prior,
    )
    yaw_candidates = yaw_candidates_for_placement(placement, yaw_prior, facing_prior)
    best_candidate_transform = np.asarray(transform, dtype=np.float64)
    best_candidate_bbox = initial_bbox
    best_candidate_loss = initial_loss
    best_candidate_bbox_loss = initial_bbox_loss
    best_candidate_vggt = initial_vggt
    best_candidate_yaw_prior_loss = initial_yaw_prior_loss
    best_candidate_facing_prior_loss = initial_facing_prior_loss
    best_candidate_size_prior_loss = initial_size_prior_loss
    best_candidate_avoidance_loss = initial_avoidance_loss
    best_candidate_delta = np.zeros(3, dtype=np.float64)
    best_candidate_yaw = 0.0
    best_candidate_scale = 1.0
    best_candidate_quality = initial_quality
    best_accepted_transform = np.asarray(transform, dtype=np.float64)
    best_accepted_bbox = initial_bbox
    best_accepted_loss = initial_loss if initial_quality.get("status") != "rejected" else float("inf")
    best_accepted_bbox_loss = initial_bbox_loss
    best_accepted_vggt = initial_vggt
    best_accepted_yaw_prior_loss = initial_yaw_prior_loss
    best_accepted_facing_prior_loss = initial_facing_prior_loss
    best_accepted_size_prior_loss = initial_size_prior_loss
    best_accepted_avoidance_loss = initial_avoidance_loss
    best_accepted_delta = np.zeros(3, dtype=np.float64)
    best_accepted_yaw = 0.0
    best_accepted_scale = 1.0
    best_accepted_quality = initial_quality
    candidate_pool: list[dict[str, Any]] = []
    support_pivot = support_plane_pivot_local(source_bounds, support_target)
    if initial_quality.get("status") != "rejected":
        keep_mask_candidate(
            candidate_pool,
            transform=np.asarray(transform, dtype=np.float64),
            bbox=initial_bbox,
            loss=initial_loss,
            bbox_loss=initial_bbox_loss,
            vggt=initial_vggt,
            yaw_prior_loss=initial_yaw_prior_loss,
            facing_prior_loss=initial_facing_prior_loss,
            physical_size_prior_loss=initial_size_prior_loss,
            object_avoidance_prior_loss=initial_avoidance_loss,
            delta=np.zeros(3, dtype=np.float64),
            yaw=0.0,
            scale=1.0,
            quality=initial_quality,
        )
    candidate_count = 0
    accepted_candidate_count = 0
    dx_candidates, dz_candidates, translation_prior_report = translation_candidates_for_avoidance(
        source_bounds,
        transform,
        avoidance,
    )
    projection_translation_report = unavailable_projection_translation_report("refinement_not_run")
    for dx in dx_candidates:
        for dz in dz_candidates:
            for yaw in yaw_candidates:
                for scale in scale_candidates:
                    candidate_count += 1
                    candidate = candidate_transform(
                        best_transform=transform,
                        delta=np.array([dx, 0.0, dz]),
                        yaw=yaw,
                        scale=scale,
                        pivot_local=support_pivot.get("pivot_local"),
                    )
                    candidate, _delta = snap_transform_to_support_bounds(source_bounds, candidate, support_y)
                    projected = projected_mesh_sample_bbox(projection_vertices, source_bounds, candidate, coordinate_contract)
                    if projected is None:
                        projected = projected_transform_bbox(source_bounds, candidate, coordinate_contract)
                    if projected is None:
                        continue
                    bbox_loss = bbox_projection_loss(projected, target_bbox)
                    support_loss = support_penalty(source_bounds, candidate, support_y)
                    scale_loss = abs(float(np.log(scale))) * 0.08
                    vggt_loss = vggt_candidate_transform_loss(vggt_fit, source_bounds, candidate)
                    yaw_loss = yaw_prior_loss(yaw_prior, transform_yaw_gltf(candidate))
                    facing_loss = facing_prior_loss(facing_prior, candidate)
                    size_loss = physical_size_prior_loss(size_prior, source_bounds, candidate)
                    if size_loss is not None and float(size_loss) > PHYSICAL_SIZE_PRIOR_CANDIDATE_REJECT_LOSS:
                        continue
                    avoidance_loss = object_avoidance_prior_loss(avoidance, source_bounds, candidate)
                    loss = objective_transform_loss(
                        bbox_loss=bbox_loss,
                        support_loss=support_loss,
                        scale_loss=scale_loss,
                        vggt_loss=vggt_loss.get("loss"),
                        vggt_loss_weight=vggt_loss_weight,
                        yaw_prior_loss=yaw_loss,
                        facing_prior_loss=facing_loss,
                        physical_size_prior_loss=size_loss,
                        object_avoidance_prior_loss=avoidance_loss,
                    )
                    quality = projection_quality_report(projected, target_bbox, allow_occluded_bottom=allow_occluded_bottom)
                    if loss < best_candidate_loss:
                        best_candidate_loss = loss
                        best_candidate_bbox_loss = bbox_loss
                        best_candidate_vggt = vggt_loss
                        best_candidate_yaw_prior_loss = yaw_loss
                        best_candidate_facing_prior_loss = facing_loss
                        best_candidate_size_prior_loss = size_loss
                        best_candidate_avoidance_loss = avoidance_loss
                        best_candidate_transform = candidate
                        best_candidate_bbox = projected
                        best_candidate_delta = np.array([dx, 0.0, dz], dtype=np.float64)
                        best_candidate_yaw = yaw
                        best_candidate_scale = scale
                        best_candidate_quality = quality
                    candidate_accepted = quality.get("status") != "rejected"
                    if candidate_accepted:
                        accepted_candidate_count += 1
                        keep_mask_candidate(
                            candidate_pool,
                            transform=candidate,
                            bbox=projected,
                            loss=loss,
                            bbox_loss=bbox_loss,
                            vggt=vggt_loss,
                            yaw_prior_loss=yaw_loss,
                            facing_prior_loss=facing_loss,
                            physical_size_prior_loss=size_loss,
                            object_avoidance_prior_loss=avoidance_loss,
                            delta=np.array([dx, 0.0, dz], dtype=np.float64),
                            yaw=yaw,
                            scale=scale,
                            quality=quality,
                        )
                        if loss < best_accepted_loss:
                            best_accepted_loss = loss
                            best_accepted_bbox_loss = bbox_loss
                            best_accepted_vggt = vggt_loss
                            best_accepted_yaw_prior_loss = yaw_loss
                            best_accepted_facing_prior_loss = facing_loss
                            best_accepted_size_prior_loss = size_loss
                            best_accepted_avoidance_loss = avoidance_loss
                            best_accepted_transform = candidate
                            best_accepted_bbox = projected
                            best_accepted_delta = np.array([dx, 0.0, dz], dtype=np.float64)
                            best_accepted_yaw = yaw
                            best_accepted_scale = scale
                            best_accepted_quality = quality
    if np.isfinite(best_accepted_loss):
        refinement_dx, refinement_dz, projection_translation_report = translation_candidates_for_projection_residual(
            source_bounds=source_bounds,
            projection_vertices=projection_vertices,
            transform=best_accepted_transform,
            target_bbox=target_bbox,
            coordinate_contract=coordinate_contract,
            support_y=support_y,
            pivot_local=support_pivot.get("pivot_local"),
        )
        if refinement_dx and refinement_dz:
            refinement_delta = np.array([float(refinement_dx[0]), 0.0, float(refinement_dz[0])], dtype=np.float64)
            refined = candidate_transform(
                best_transform=best_accepted_transform,
                delta=refinement_delta,
                yaw=0.0,
                scale=1.0,
                pivot_local=support_pivot.get("pivot_local"),
            )
            refined, _delta = snap_transform_to_support_bounds(source_bounds, refined, support_y)
            projected = projected_mesh_sample_bbox(projection_vertices, source_bounds, refined, coordinate_contract)
            if projected is None:
                projected = projected_transform_bbox(source_bounds, refined, coordinate_contract)
            if projected is not None:
                bbox_loss = bbox_projection_loss(projected, target_bbox)
                support_loss = support_penalty(source_bounds, refined, support_y)
                scale_loss = abs(float(np.log(best_accepted_scale))) * 0.08
                vggt_loss = vggt_candidate_transform_loss(vggt_fit, source_bounds, refined)
                yaw_loss = yaw_prior_loss(yaw_prior, transform_yaw_gltf(refined))
                facing_loss = facing_prior_loss(facing_prior, refined)
                size_loss = physical_size_prior_loss(size_prior, source_bounds, refined)
                avoidance_loss = object_avoidance_prior_loss(avoidance, source_bounds, refined)
                quality = projection_quality_report(projected, target_bbox, allow_occluded_bottom=allow_occluded_bottom)
                if size_loss is None or float(size_loss) <= PHYSICAL_SIZE_PRIOR_CANDIDATE_REJECT_LOSS:
                    loss = objective_transform_loss(
                        bbox_loss=bbox_loss,
                        support_loss=support_loss,
                        scale_loss=scale_loss,
                        vggt_loss=vggt_loss.get("loss"),
                        vggt_loss_weight=vggt_loss_weight,
                        yaw_prior_loss=yaw_loss,
                        facing_prior_loss=facing_loss,
                        physical_size_prior_loss=size_loss,
                        object_avoidance_prior_loss=avoidance_loss,
                    )
                    if quality.get("status") != "rejected":
                        total_delta = best_accepted_delta + refinement_delta
                        keep_mask_candidate(
                            candidate_pool,
                            transform=refined,
                            bbox=projected,
                            loss=loss,
                            bbox_loss=bbox_loss,
                            vggt=vggt_loss,
                            yaw_prior_loss=yaw_loss,
                            facing_prior_loss=facing_loss,
                            physical_size_prior_loss=size_loss,
                            object_avoidance_prior_loss=avoidance_loss,
                            delta=total_delta,
                            yaw=best_accepted_yaw,
                            scale=best_accepted_scale,
                            quality=quality,
                        )
                        if loss < best_accepted_loss:
                            accepted_candidate_count += 1
                            best_accepted_loss = loss
                            best_accepted_bbox_loss = bbox_loss
                            best_accepted_vggt = vggt_loss
                            best_accepted_yaw_prior_loss = yaw_loss
                            best_accepted_facing_prior_loss = facing_loss
                            best_accepted_size_prior_loss = size_loss
                            best_accepted_avoidance_loss = avoidance_loss
                            best_accepted_transform = refined
                            best_accepted_bbox = projected
                            best_accepted_delta = total_delta
                            best_accepted_quality = quality
    accepted = np.isfinite(best_accepted_loss)
    final_transform = best_accepted_transform if accepted else np.asarray(transform, dtype=np.float64)
    final_bbox = best_accepted_bbox if accepted else initial_bbox
    final_loss = best_accepted_loss if accepted else initial_loss
    final_bbox_loss = best_accepted_bbox_loss if accepted else initial_bbox_loss
    final_vggt = best_accepted_vggt if accepted else initial_vggt
    final_yaw_prior_loss = best_accepted_yaw_prior_loss if accepted else initial_yaw_prior_loss
    final_facing_prior_loss = best_accepted_facing_prior_loss if accepted else initial_facing_prior_loss
    final_size_prior_loss = best_accepted_size_prior_loss if accepted else initial_size_prior_loss
    final_avoidance_loss = best_accepted_avoidance_loss if accepted else initial_avoidance_loss
    final_quality = best_accepted_quality if accepted else initial_quality
    final_mask = initial_mask
    mask_fallback_reason = None
    if accepted:
        selected_mask_candidate, mask_report = select_mask_candidate(
            mask_fit,
            candidates=candidate_pool,
            meshes=meshes,
            source_bounds=source_bounds,
            vggt_fit=vggt_fit,
        )
        mask_fallback_reason = mask_report.get("fallback_reason")
        if selected_mask_candidate is not None:
            final_transform = np.asarray(selected_mask_candidate["transform"], dtype=np.float64)
            final_bbox = np.asarray(selected_mask_candidate["bbox"], dtype=np.float64)
            final_loss = float(selected_mask_candidate["combined_loss"])
            final_bbox_loss = float(selected_mask_candidate["bbox_loss"])
            final_vggt = selected_mask_candidate["vggt"]
            final_yaw_prior_loss = selected_mask_candidate["yaw_prior_loss"]
            final_facing_prior_loss = selected_mask_candidate["facing_prior_loss"]
            final_size_prior_loss = selected_mask_candidate["physical_size_prior_loss"]
            final_avoidance_loss = selected_mask_candidate["object_avoidance_prior_loss"]
            final_quality = selected_mask_candidate["quality"]
            best_accepted_delta = np.asarray(selected_mask_candidate["delta"], dtype=np.float64)
            best_accepted_yaw = float(selected_mask_candidate["yaw"])
            best_accepted_scale = float(selected_mask_candidate["scale"])
            final_mask = selected_mask_candidate["mask"]
        elif mask_report.get("status") == "unavailable":
            final_mask = initial_mask
        post_dx, post_dz, post_projection_translation_report = translation_candidates_for_projection_residual(
            source_bounds=source_bounds,
            projection_vertices=projection_vertices,
            transform=final_transform,
            target_bbox=target_bbox,
            coordinate_contract=coordinate_contract,
            support_y=support_y,
            pivot_local=support_pivot.get("pivot_local"),
        )
        if post_dx and post_dz:
            refinement_delta = np.array([float(post_dx[0]), 0.0, float(post_dz[0])], dtype=np.float64)
            refined = candidate_transform(
                best_transform=final_transform,
                delta=refinement_delta,
                yaw=0.0,
                scale=1.0,
                pivot_local=support_pivot.get("pivot_local"),
            )
            refined, _delta = snap_transform_to_support_bounds(source_bounds, refined, support_y)
            projected = projected_mesh_sample_bbox(projection_vertices, source_bounds, refined, coordinate_contract)
            if projected is None:
                projected = projected_transform_bbox(source_bounds, refined, coordinate_contract)
            if projected is not None:
                bbox_loss = bbox_projection_loss(projected, target_bbox)
                support_loss = support_penalty(source_bounds, refined, support_y)
                scale_loss = abs(float(np.log(best_accepted_scale))) * 0.08
                vggt_loss = vggt_candidate_transform_loss(vggt_fit, source_bounds, refined)
                yaw_loss = yaw_prior_loss(yaw_prior, transform_yaw_gltf(refined))
                facing_loss = facing_prior_loss(facing_prior, refined)
                size_loss = physical_size_prior_loss(size_prior, source_bounds, refined)
                avoidance_loss = object_avoidance_prior_loss(avoidance, source_bounds, refined)
                quality = projection_quality_report(projected, target_bbox, allow_occluded_bottom=allow_occluded_bottom)
                if quality.get("status") != "rejected" and (size_loss is None or float(size_loss) <= PHYSICAL_SIZE_PRIOR_CANDIDATE_REJECT_LOSS):
                    objective_loss = objective_transform_loss(
                        bbox_loss=bbox_loss,
                        support_loss=support_loss,
                        scale_loss=scale_loss,
                        vggt_loss=vggt_loss.get("loss"),
                        vggt_loss_weight=vggt_loss_weight,
                        yaw_prior_loss=yaw_loss,
                        facing_prior_loss=facing_loss,
                        physical_size_prior_loss=size_loss,
                        object_avoidance_prior_loss=avoidance_loss,
                    )
                    refined_mask = mask_candidate_transform_loss(
                        mask_fit,
                        meshes=meshes,
                        source_bounds=source_bounds,
                        transform=refined,
                    )
                    point_match = vggt_point_match_transform_loss(
                        vggt_fit,
                        meshes=meshes,
                        source_bounds=source_bounds,
                        transform=refined,
                    )
                    if refined_mask.get("status") == "accepted" and refined_mask.get("loss") is not None:
                        combined_loss = float(objective_loss) + float(refined_mask["loss"]) * MASK_CANDIDATE_LOSS_WEIGHT
                        if point_match.get("loss") is not None:
                            combined_loss += float(point_match["loss"]) * VGGT_POINT_MATCH_LOSS_WEIGHT
                        if combined_loss < final_loss:
                            final_transform = refined
                            final_bbox = projected
                            final_loss = combined_loss
                            final_bbox_loss = bbox_loss
                            final_vggt = vggt_loss
                            final_yaw_prior_loss = yaw_loss
                            final_facing_prior_loss = facing_loss
                            final_size_prior_loss = size_loss
                            final_avoidance_loss = avoidance_loss
                            final_quality = quality
                            best_accepted_delta = best_accepted_delta + refinement_delta
                            report = dict(refined_mask)
                            report.update(
                                status="accepted",
                                candidate_count=len(candidate_pool),
                                evaluated_count=None,
                                selected_yaw=float(best_accepted_yaw),
                                selected_scale=float(best_accepted_scale),
                                base_loss=float(objective_loss),
                                combined_loss=float(combined_loss),
                                loss_weight=float(MASK_CANDIDATE_LOSS_WEIGHT),
                                vggt_point_match=point_match,
                                vggt_point_match_loss_weight=float(VGGT_POINT_MATCH_LOSS_WEIGHT if point_match.get("loss") is not None else 0.0),
                                selected_projected_bbox_xyxy=projected.tolist(),
                                fallback_reason=None,
                            )
                            final_mask = report
                            projection_translation_report = post_projection_translation_report
    else:
        mask_report = unavailable_mask_candidate_report(
            mask_fit,
            fallback_reason="no_projection_accepted_candidate",
            candidate_count=len(candidate_pool),
        )
    base_report.update(
        initial_loss=float(initial_loss),
        optimized_loss=float(final_loss),
        candidate_loss=float(best_candidate_loss),
        initial_bbox_loss=float(initial_bbox_loss),
        optimized_bbox_loss=float(final_bbox_loss),
        candidate_bbox_loss=float(best_candidate_bbox_loss),
        optimized_projected_bbox_xyxy=final_bbox.tolist(),
        candidate_projected_bbox_xyxy=best_candidate_bbox.tolist(),
        delta_gltf=[float(value) for value in best_accepted_delta] if accepted else [0.0, 0.0, 0.0],
        yaw_delta_radians=float(best_accepted_yaw) if accepted else 0.0,
        uniform_scale_delta=float(best_accepted_scale) if accepted else 1.0,
        scale_candidates=[float(value) for value in scale_candidates],
        yaw_candidates=[float(value) for value in yaw_candidates],
        minimum_scale_delta=float(scale_floor_report["minimum_scale_delta"]),
        minimum_scale_reason=scale_floor_report["reason"],
        target_height_ratio=scale_floor_report["target_height_ratio"],
        candidate_count=int(candidate_count),
        accepted_candidate_count=int(accepted_candidate_count),
        candidate_projection_quality=best_candidate_quality,
        candidate_uniform_scale_delta=float(best_candidate_scale),
        candidate_delta_gltf=[float(value) for value in best_candidate_delta],
        candidate_yaw_delta_radians=float(best_candidate_yaw),
        orientation_search=orientation_search_report(
            yaw_candidates=yaw_candidates,
            dx_candidates=dx_candidates,
            dz_candidates=dz_candidates,
            selected_yaw=best_accepted_yaw if accepted else 0.0,
            candidate_count=candidate_count,
            accepted_candidate_count=accepted_candidate_count,
            initial_loss=initial_loss,
            optimized_loss=final_loss,
            bbox_loss=final_bbox_loss,
            support_loss=support_penalty(source_bounds, final_transform, support_y),
            scale_delta=best_accepted_scale if accepted else 1.0,
            vggt_loss=final_vggt.get("loss"),
            yaw_prior_loss=final_yaw_prior_loss,
            facing_prior_loss=final_facing_prior_loss,
            physical_size_prior_loss=final_size_prior_loss,
            object_avoidance_prior_loss=final_avoidance_loss,
            mask_loss=final_mask.get("loss"),
            support_pivot=support_pivot,
            translation_prior=translation_prior_report,
            projection_translation_prior=projection_translation_report,
            fallback_reason=None if accepted else "no_projection_accepted_candidate",
        ),
        vggt_candidate_fit=vggt_candidate_fit_report(
            vggt_fit,
            loss_weight=vggt_loss_weight,
            initial=initial_vggt,
            optimized=final_vggt,
            candidate=best_candidate_vggt,
        ),
        vggt_yaw_prior=vggt_yaw_prior_report(
            yaw_prior,
            loss_weight=VGGT_YAW_PRIOR_WEIGHT,
            initial_yaw=initial_yaw,
            optimized_yaw=best_accepted_yaw if accepted else initial_yaw,
            candidate_yaw=best_candidate_yaw,
            initial_loss=initial_yaw_prior_loss,
            optimized_loss=final_yaw_prior_loss,
            candidate_loss=best_candidate_yaw_prior_loss,
        ),
        mesh_facing_prior=mesh_facing_prior_report(
            facing_prior,
            loss_weight=MESH_FACING_PRIOR_WEIGHT,
            initial_transform=transform,
            optimized_transform=final_transform,
            candidate_transform_value=best_candidate_transform,
            initial_loss=initial_facing_prior_loss,
            optimized_loss=final_facing_prior_loss,
            candidate_loss=best_candidate_facing_prior_loss,
        ),
        physical_size_prior=physical_size_prior_report(
            size_prior,
            initial=initial_size_prior_loss,
            optimized=final_size_prior_loss,
            candidate=best_candidate_size_prior_loss,
        ),
        object_avoidance_prior=object_avoidance_prior_report(
            avoidance,
            initial=initial_avoidance_loss,
            optimized=final_avoidance_loss,
            candidate=best_candidate_avoidance_loss,
        ),
        mask_candidate_fit=mask_candidate_fit_report(
            mask_fit,
            loss_weight=mask_loss_weight,
            initial=initial_mask,
            optimized=final_mask,
            candidate=mask_report.get("candidate") or final_mask,
            candidate_count=len(candidate_pool),
            selected_yaw=best_accepted_yaw if accepted else 0.0,
            selected_scale=best_accepted_scale if accepted else 1.0,
            fallback_reason=mask_fallback_reason,
        ),
        projection_quality=final_quality,
    )
    return {"transform": final_transform, "report": base_report}


def yaw_candidates_for_placement(placement: dict[str, Any], yaw_prior: dict[str, Any], facing_prior: dict[str, Any]) -> tuple[float, ...]:
    _ = placement
    candidates = {round(float(value), 6) for value in DEFAULT_YAW_CANDIDATES}
    prior_yaws = list(yaw_prior.get("candidate_yaws") or []) + list(facing_prior.get("candidate_yaws") or [])
    for value in prior_yaws:
        yaw = normalize_angle(float(value))
        for delta in (0.0, -0.25, 0.25):
            candidates.add(round(normalize_angle(yaw + delta), 6))
    return tuple(sorted(candidates))


def vggt_yaw_prior_from_placement(placement: dict[str, Any]) -> dict[str, Any]:
    rotation = np.asarray(placement.get("rotation_matrix"), dtype=np.float64)
    extent = np.asarray(placement.get("extent_xyz"), dtype=np.float64)
    if rotation.shape != (3, 3) or extent.shape != (3,) or not np.isfinite(rotation).all() or not np.isfinite(extent).all():
        return unavailable_vggt_yaw_prior("missing_or_invalid_obb")
    if np.any(extent <= 1e-8):
        return unavailable_vggt_yaw_prior("degenerate_extent")
    candidates: list[dict[str, Any]] = []
    for axis_index in range(3):
        scene_axis = rotation[:, axis_index]
        gltf_axis = np.asarray([scene_axis[0], scene_axis[2], -scene_axis[1]], dtype=np.float64)
        horizontal = np.asarray([gltf_axis[0], gltf_axis[2]], dtype=np.float64)
        horizontal_norm = float(np.linalg.norm(horizontal))
        if horizontal_norm <= 1e-8:
            continue
        horizontal_ratio = horizontal_norm / max(float(np.linalg.norm(gltf_axis)), 1e-8)
        if horizontal_ratio < 0.35:
            continue
        yaw = float(np.arctan2(horizontal[0], horizontal[1]))
        axis_weight = float(extent[axis_index]) * horizontal_ratio
        for offset in (0.0, float(np.pi)):
            candidates.append(
                {
                    "yaw": normalize_angle(yaw + offset),
                    "axis_index": axis_index,
                    "axis_extent": float(extent[axis_index]),
                    "horizontal_ratio": horizontal_ratio,
                    "weight": axis_weight,
                }
            )
    if not candidates:
        return unavailable_vggt_yaw_prior("no_horizontal_obb_axis")
    candidates.sort(key=lambda item: float(item["weight"]), reverse=True)
    return {
        "available": True,
        "method": "vggt_obb_horizontal_axis_yaw_prior_v1",
        "reason": None,
        "candidate_yaws": [float(item["yaw"]) for item in candidates],
        "candidates": candidates,
    }


def unavailable_vggt_yaw_prior(reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "method": "vggt_obb_horizontal_axis_yaw_prior_v1",
        "reason": reason,
        "candidate_yaws": [],
        "candidates": [],
    }


def yaw_prior_loss(yaw_prior: dict[str, Any], yaw: float) -> float | None:
    yaws = yaw_prior.get("candidate_yaws") or []
    if not yaws:
        return None
    distance = min(abs(angle_difference(float(yaw), float(candidate))) for candidate in yaws)
    return float(distance / np.pi)


def vggt_yaw_prior_report(
    yaw_prior: dict[str, Any],
    *,
    loss_weight: float,
    initial_yaw: float,
    optimized_yaw: float,
    candidate_yaw: float,
    initial_loss: float | None,
    optimized_loss: float | None,
    candidate_loss: float | None,
) -> dict[str, Any]:
    return {
        "method": yaw_prior.get("method"),
        "status": "accepted" if yaw_prior.get("available") else "unavailable",
        "reason": yaw_prior.get("reason"),
        "loss_weight": float(loss_weight if yaw_prior.get("available") else 0.0),
        "candidate_yaws": [float(value) for value in yaw_prior.get("candidate_yaws") or []],
        "candidates": yaw_prior.get("candidates") or [],
        "initial": {"yaw": float(initial_yaw), "loss": float(initial_loss) if initial_loss is not None else None},
        "optimized": {"yaw": float(optimized_yaw), "loss": float(optimized_loss) if optimized_loss is not None else None},
        "candidate": {"yaw": float(candidate_yaw), "loss": float(candidate_loss) if candidate_loss is not None else None},
    }


def mesh_facing_prior_from_target(
    *,
    meshes: list[Any],
    transform: np.ndarray,
    facing_target_gltf: Any,
) -> dict[str, Any]:
    target = vector_or_none(facing_target_gltf)
    if target is None:
        return unavailable_mesh_facing_prior("missing_facing_target")
    asymmetry = mesh_vertical_asymmetry_direction(meshes)
    if asymmetry is None:
        return unavailable_mesh_facing_prior("mesh_has_no_stable_vertical_asymmetry")
    center = np.asarray(transform, dtype=np.float64)[:3, 3]
    desired = np.asarray([target[0] - center[0], target[2] - center[2]], dtype=np.float64)
    desired_norm = float(np.linalg.norm(desired))
    if desired_norm <= 1e-8:
        return unavailable_mesh_facing_prior("degenerate_facing_target")
    desired /= desired_norm
    local_front = -np.asarray(asymmetry["high_side_local_xz"], dtype=np.float64)
    current_front = transform_local_xz_direction(transform, local_front)
    if current_front is None:
        return unavailable_mesh_facing_prior("degenerate_front_direction")
    delta = signed_angle_2d(current_front, desired)
    current_yaw = transform_yaw_gltf(transform)
    candidate_yaw = normalize_angle(current_yaw + delta)
    return {
        "available": True,
        "method": "mesh_vertical_asymmetry_faces_nearby_anchor_v1",
        "reason": None,
        "target_gltf": [float(value) for value in target],
        "desired_direction_xz": [float(value) for value in desired],
        "local_front_xz": [float(value) for value in local_front],
        "candidate_yaws": [candidate_yaw, normalize_angle(candidate_yaw + np.pi)],
        "asymmetry": asymmetry,
    }


def physical_size_prior_from_target(
    *,
    source_bounds: np.ndarray,
    transform: np.ndarray,
    target_extent_gltf: Any,
) -> dict[str, Any]:
    target, target_volume = physical_size_target_extent_and_volume(target_extent_gltf)
    if target is None:
        return unavailable_physical_size_prior("missing_physical_size_target")
    if np.any(target <= 1e-8):
        return unavailable_physical_size_prior("degenerate_physical_size_target")
    current_bounds = transformed_bounds_from_source_bounds(source_bounds, transform)
    current_extent = current_bounds[1] - current_bounds[0]
    current_volume = physical_extent_volume(current_extent)
    if target_volume is None:
        target_volume = physical_extent_volume(target)
    active = target > max(float(np.linalg.norm(target)) * 0.025, 1e-5)
    if not bool(np.any(active)):
        return unavailable_physical_size_prior("no_active_target_axes")
    ratios = target[active] / np.maximum(current_extent[active], 1e-8)
    if len(ratios) == 0 or not np.isfinite(ratios).all():
        return unavailable_physical_size_prior("invalid_scale_ratio")
    axis_scale = float(np.median(ratios))
    volume_scale = None
    if current_volume is not None and target_volume is not None and current_volume > 1e-12 and target_volume > 1e-12:
        volume_scale = float((target_volume / current_volume) ** (1.0 / 3.0))
    scale_candidate = float(volume_scale if volume_scale is not None else axis_scale)
    if not np.isfinite(scale_candidate) or scale_candidate <= 1e-8:
        return unavailable_physical_size_prior("invalid_scale_candidate")
    return {
        "available": True,
        "method": "repeated_instance_physical_extent_prior_v1",
        "reason": None,
        "target_extent_gltf": [float(value) for value in target],
        "target_volume_gltf": float(target_volume) if target_volume is not None else None,
        "initial_extent_gltf": [float(value) for value in current_extent],
        "initial_volume_gltf": float(current_volume) if current_volume is not None else None,
        "axis_scale_candidate": axis_scale,
        "volume_scale_candidate": volume_scale,
        "scale_candidate": scale_candidate,
    }


def physical_size_target_extent_and_volume(value: Any) -> tuple[np.ndarray | None, float | None]:
    if isinstance(value, dict):
        extent = vector_or_none(value.get("target_extent_gltf") or value.get("extent_gltf"))
        volume_value = value.get("target_volume_gltf") if "target_volume_gltf" in value else value.get("volume_gltf")
        try:
            volume = float(volume_value) if volume_value is not None else None
        except (TypeError, ValueError):
            volume = None
        if volume is not None and (not np.isfinite(volume) or volume <= 0.0):
            volume = None
        return extent, volume
    return vector_or_none(value), None


def physical_extent_volume(extent: np.ndarray) -> float | None:
    values = np.asarray(extent, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all() or np.any(values <= 1e-8):
        return None
    return float(np.prod(values))


def unavailable_physical_size_prior(reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "method": "repeated_instance_physical_extent_prior_v1",
        "reason": reason,
        "target_extent_gltf": None,
        "target_volume_gltf": None,
        "initial_extent_gltf": None,
        "initial_volume_gltf": None,
        "axis_scale_candidate": None,
        "volume_scale_candidate": None,
        "scale_candidate": None,
    }


def physical_size_prior_loss(prior: dict[str, Any], source_bounds: np.ndarray, transform: np.ndarray) -> float | None:
    if not prior.get("available"):
        return None
    target = np.asarray(prior.get("target_extent_gltf"), dtype=np.float64)
    if target.shape != (3,) or not np.isfinite(target).all() or np.any(target <= 1e-8):
        return None
    current_bounds = transformed_bounds_from_source_bounds(source_bounds, transform)
    extent = np.maximum(current_bounds[1] - current_bounds[0], 1e-8)
    active = target > max(float(np.linalg.norm(target)) * 0.025, 1e-5)
    if not bool(np.any(active)):
        return None
    extent_loss = float(np.mean(np.abs(np.log(extent[active] / target[active]))))
    current_volume = physical_extent_volume(extent)
    target_volume = prior.get("target_volume_gltf")
    volume_loss = None
    if current_volume is not None and target_volume is not None:
        volume_loss = float(abs(np.log(current_volume / max(float(target_volume), 1e-12))))
    if volume_loss is None:
        return extent_loss
    return float(0.75 * volume_loss + 0.25 * extent_loss)


def physical_size_prior_report(
    prior: dict[str, Any],
    *,
    initial: float | None,
    optimized: float | None,
    candidate: float | None,
) -> dict[str, Any]:
    return {
        "method": prior.get("method"),
        "status": "accepted" if prior.get("available") else "unavailable",
        "reason": prior.get("reason"),
        "loss_weight": float(REPEATED_INSTANCE_SIZE_PRIOR_WEIGHT if prior.get("available") else 0.0),
        "target_extent_gltf": prior.get("target_extent_gltf"),
        "target_volume_gltf": prior.get("target_volume_gltf"),
        "initial_extent_gltf": prior.get("initial_extent_gltf"),
        "initial_volume_gltf": prior.get("initial_volume_gltf"),
        "axis_scale_candidate": prior.get("axis_scale_candidate"),
        "volume_scale_candidate": prior.get("volume_scale_candidate"),
        "scale_candidate": prior.get("scale_candidate"),
        "initial": {"loss": float(initial) if initial is not None else None},
        "optimized": {"loss": float(optimized) if optimized is not None else None},
        "candidate": {"loss": float(candidate) if candidate is not None else None},
    }


def unavailable_mesh_facing_prior(reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "method": "mesh_vertical_asymmetry_faces_nearby_anchor_v1",
        "reason": reason,
        "target_gltf": None,
        "desired_direction_xz": None,
        "local_front_xz": None,
        "candidate_yaws": [],
        "asymmetry": None,
    }


def mesh_vertical_asymmetry_direction(meshes: list[Any]) -> dict[str, Any] | None:
    vertices: list[np.ndarray] = []
    for mesh in meshes:
        values = np.asarray(mesh.vertices, dtype=np.float64)
        if values.ndim == 2 and values.shape[1] == 3 and len(values) > 0:
            vertices.append(values[np.isfinite(values).all(axis=1)])
    vertices = [item for item in vertices if len(item) > 0]
    if not vertices:
        return None
    points = np.concatenate(vertices, axis=0)
    bounds = np.stack([points.min(axis=0), points.max(axis=0)], axis=0)
    extent = bounds[1] - bounds[0]
    if np.any(extent <= 1e-8):
        return None
    y_threshold = float(np.percentile(points[:, 1], 75.0))
    high = points[points[:, 1] >= y_threshold]
    if len(high) < 4:
        return None
    offset = high[:, [0, 2]].mean(axis=0) - ((bounds[0, [0, 2]] + bounds[1, [0, 2]]) / 2.0)
    horizontal_extent = max(float(extent[0]), float(extent[2]), 1e-8)
    ratio = float(np.linalg.norm(offset) / horizontal_extent)
    if ratio < 0.12:
        return None
    direction = offset / max(float(np.linalg.norm(offset)), 1e-8)
    return {
        "method": "top_quartile_horizontal_offset",
        "high_side_local_xz": [float(value) for value in direction],
        "offset_local_xz": [float(value) for value in offset],
        "offset_ratio": ratio,
        "top_quantile": 75.0,
        "sample_count": int(len(points)),
        "high_sample_count": int(len(high)),
    }


def facing_prior_loss(facing_prior: dict[str, Any], transform: np.ndarray) -> float | None:
    if not facing_prior.get("available"):
        return None
    desired = np.asarray(facing_prior.get("desired_direction_xz"), dtype=np.float64)
    local_front = np.asarray(facing_prior.get("local_front_xz"), dtype=np.float64)
    current = transform_local_xz_direction(transform, local_front)
    if current is None or desired.shape != (2,) or not np.isfinite(desired).all():
        return None
    return float(abs(signed_angle_2d(current, desired)) / np.pi)


def mesh_facing_prior_report(
    facing_prior: dict[str, Any],
    *,
    loss_weight: float,
    initial_transform: np.ndarray,
    optimized_transform: np.ndarray,
    candidate_transform_value: np.ndarray,
    initial_loss: float | None,
    optimized_loss: float | None,
    candidate_loss: float | None,
) -> dict[str, Any]:
    return {
        "method": facing_prior.get("method"),
        "status": "accepted" if facing_prior.get("available") else "unavailable",
        "reason": facing_prior.get("reason"),
        "loss_weight": float(loss_weight if facing_prior.get("available") else 0.0),
        "target_gltf": facing_prior.get("target_gltf"),
        "desired_direction_xz": facing_prior.get("desired_direction_xz"),
        "local_front_xz": facing_prior.get("local_front_xz"),
        "candidate_yaws": [float(value) for value in facing_prior.get("candidate_yaws") or []],
        "asymmetry": facing_prior.get("asymmetry"),
        "initial": facing_transform_report(facing_prior, initial_transform, initial_loss),
        "optimized": facing_transform_report(facing_prior, optimized_transform, optimized_loss),
        "candidate": facing_transform_report(facing_prior, candidate_transform_value, candidate_loss),
    }


def facing_transform_report(facing_prior: dict[str, Any], transform: np.ndarray, loss: float | None) -> dict[str, Any]:
    local_front = np.asarray(facing_prior.get("local_front_xz"), dtype=np.float64)
    direction = transform_local_xz_direction(transform, local_front) if facing_prior.get("available") else None
    return {
        "yaw": transform_yaw_gltf(transform),
        "front_direction_xz": [float(value) for value in direction] if direction is not None else None,
        "loss": float(loss) if loss is not None else None,
    }


def transform_local_xz_direction(transform: np.ndarray, local_xz: np.ndarray) -> np.ndarray | None:
    if local_xz.shape != (2,) or not np.isfinite(local_xz).all():
        return None
    local = np.asarray([local_xz[0], 0.0, local_xz[1]], dtype=np.float64)
    world = np.asarray(transform, dtype=np.float64)[:3, :3] @ local
    horizontal = np.asarray([world[0], world[2]], dtype=np.float64)
    norm = float(np.linalg.norm(horizontal))
    if norm <= 1e-8:
        return None
    return horizontal / norm


def signed_angle_2d(source: np.ndarray, target: np.ndarray) -> float:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source = source / max(float(np.linalg.norm(source)), 1e-8)
    target = target / max(float(np.linalg.norm(target)), 1e-8)
    cross = float(source[0] * target[1] - source[1] * target[0])
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    return float(np.arctan2(cross, dot))


def vector_or_none(value: Any) -> np.ndarray | None:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if vector.shape != (3,) or not np.isfinite(vector).all():
        return None
    return vector


def transform_yaw_gltf(transform: np.ndarray) -> float:
    matrix = np.asarray(transform, dtype=np.float64)
    forward = matrix[:3, 2]
    horizontal = np.asarray([forward[0], forward[2]], dtype=np.float64)
    if float(np.linalg.norm(horizontal)) <= 1e-8:
        return 0.0
    return normalize_angle(float(np.arctan2(horizontal[0], horizontal[1])))


def normalize_angle(value: float) -> float:
    return float((float(value) + np.pi) % (2.0 * np.pi) - np.pi)


def angle_difference(left: float, right: float) -> float:
    return normalize_angle(float(left) - float(right))


def empty_orientation_search_report() -> dict[str, Any]:
    return {
        "yaw_candidates": [],
        "selected_yaw": None,
        "loss_breakdown": {},
        "fallback_reason": "search_not_run",
    }


def orientation_search_report(
    *,
    yaw_candidates: Any,
    dx_candidates: Any | None = None,
    dz_candidates: Any | None = None,
    selected_yaw: float,
    candidate_count: int,
    accepted_candidate_count: int,
    initial_loss: float | None,
    optimized_loss: float | None,
    bbox_loss: float | None,
    support_loss: float | None,
    scale_delta: float | None,
    vggt_loss: Any,
    yaw_prior_loss: Any = None,
    facing_prior_loss: Any = None,
    physical_size_prior_loss: Any = None,
    object_avoidance_prior_loss: Any = None,
    mask_loss: Any = None,
    support_pivot: dict[str, Any] | None = None,
    translation_prior: dict[str, Any] | None = None,
    projection_translation_prior: dict[str, Any] | None = None,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    scale_prior = abs(float(np.log(float(scale_delta)))) * 0.08 if scale_delta not in (None, 0) else 0.0
    return {
        "yaw_candidates": [float(value) for value in yaw_candidates],
        "dx_candidates": [float(value) for value in (dx_candidates or [])],
        "dz_candidates": [float(value) for value in (dz_candidates or [])],
        "selected_yaw": float(selected_yaw),
        "loss_breakdown": {
            "initial_total": float(initial_loss) if initial_loss is not None else None,
            "optimized_total": float(optimized_loss) if optimized_loss is not None else None,
            "bbox_projection": float(bbox_loss) if bbox_loss is not None else None,
            "support_contact": float(support_loss) if support_loss is not None else None,
            "scale_prior": float(scale_prior),
            "vggt_points": float(vggt_loss) if vggt_loss is not None else None,
            "vggt_yaw_prior": float(yaw_prior_loss) if yaw_prior_loss is not None else None,
            "mesh_facing_prior": float(facing_prior_loss) if facing_prior_loss is not None else None,
            "physical_size_prior": float(physical_size_prior_loss) if physical_size_prior_loss is not None else None,
            "object_avoidance_prior": float(object_avoidance_prior_loss) if object_avoidance_prior_loss is not None else None,
            "mask_silhouette": float(mask_loss) if mask_loss is not None else None,
            "candidate_count": int(candidate_count),
            "accepted_candidate_count": int(accepted_candidate_count),
        },
        "support_pivot": support_pivot,
        "translation_prior": translation_prior,
        "projection_translation_prior": projection_translation_prior,
        "fallback_reason": fallback_reason,
    }


def keep_mask_candidate(
    candidates: list[dict[str, Any]],
    *,
    transform: np.ndarray,
    bbox: np.ndarray,
    loss: float,
    bbox_loss: float,
    vggt: dict[str, Any],
    yaw_prior_loss: float | None,
    facing_prior_loss: float | None,
    physical_size_prior_loss: float | None,
    object_avoidance_prior_loss: float | None,
    delta: np.ndarray,
    yaw: float,
    scale: float,
    quality: dict[str, Any],
) -> None:
    record = {
        "transform": np.asarray(transform, dtype=np.float64).copy(),
        "bbox": np.asarray(bbox, dtype=np.float64).copy(),
        "loss": float(loss),
        "bbox_loss": float(bbox_loss),
        "vggt": dict(vggt),
        "yaw_prior_loss": float(yaw_prior_loss) if yaw_prior_loss is not None else None,
        "facing_prior_loss": float(facing_prior_loss) if facing_prior_loss is not None else None,
        "physical_size_prior_loss": float(physical_size_prior_loss) if physical_size_prior_loss is not None else None,
        "object_avoidance_prior_loss": float(object_avoidance_prior_loss) if object_avoidance_prior_loss is not None else None,
        "delta": np.asarray(delta, dtype=np.float64).copy(),
        "yaw": float(yaw),
        "scale": float(scale),
        "quality": dict(quality),
    }
    candidates.append(record)
    candidates.sort(key=lambda item: float(item["loss"]))
    del candidates[MASK_CANDIDATE_POOL_SIZE:]


def select_mask_candidate(
    fit: dict[str, Any],
    *,
    candidates: list[dict[str, Any]],
    meshes: list[Any],
    source_bounds: np.ndarray,
    vggt_fit: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not fit.get("available"):
        return None, unavailable_mask_candidate_report(fit, fallback_reason=(fit.get("report") or {}).get("reason"), candidate_count=len(candidates))
    if not candidates:
        return None, unavailable_mask_candidate_report(fit, fallback_reason="empty_candidate_pool", candidate_count=0)
    best: dict[str, Any] | None = None
    best_mask: dict[str, Any] | None = None
    best_point_match: dict[str, Any] | None = None
    best_combined = float("inf")
    evaluated = 0
    for candidate in candidates:
        size_prior_loss = candidate.get("physical_size_prior_loss")
        if size_prior_loss is not None and float(size_prior_loss) > PHYSICAL_SIZE_PRIOR_CANDIDATE_REJECT_LOSS:
            continue
        mask_loss = mask_candidate_transform_loss(
            fit,
            meshes=meshes,
            source_bounds=source_bounds,
            transform=np.asarray(candidate["transform"], dtype=np.float64),
        )
        if mask_loss.get("status") != "accepted" or mask_loss.get("loss") is None:
            continue
        evaluated += 1
        point_match = vggt_point_match_transform_loss(
            vggt_fit or unavailable_vggt_candidate_fit("missing_vggt_fit"),
            meshes=meshes,
            source_bounds=source_bounds,
            transform=np.asarray(candidate["transform"], dtype=np.float64),
        )
        point_loss = point_match.get("loss")
        combined = float(candidate["loss"]) + float(mask_loss["loss"]) * MASK_CANDIDATE_LOSS_WEIGHT
        if point_loss is not None:
            combined += float(point_loss) * VGGT_POINT_MATCH_LOSS_WEIGHT
        if combined < best_combined:
            best_combined = combined
            best = candidate
            best_mask = mask_loss
            best_point_match = point_match
    if best is None or best_mask is None:
        return None, unavailable_mask_candidate_report(
            fit,
            fallback_reason="no_rendered_mask_candidate",
            candidate_count=len(candidates),
            evaluated_count=evaluated,
        )
    selected = dict(best)
    report = dict(best_mask)
    report.update(
        status="accepted",
        candidate_count=len(candidates),
        evaluated_count=evaluated,
        selected_yaw=float(best["yaw"]),
        selected_scale=float(best["scale"]),
        base_loss=float(best["loss"]),
        combined_loss=float(best_combined),
        loss_weight=float(MASK_CANDIDATE_LOSS_WEIGHT),
        vggt_point_match=best_point_match,
        vggt_point_match_loss_weight=float(VGGT_POINT_MATCH_LOSS_WEIGHT if (best_point_match or {}).get("loss") is not None else 0.0),
        selected_projected_bbox_xyxy=np.asarray(best["bbox"], dtype=np.float64).tolist(),
        fallback_reason=None,
    )
    selected["mask"] = report
    selected["vggt_point_match"] = best_point_match
    selected["combined_loss"] = best_combined
    return selected, report


def triangle_area_2d(points: np.ndarray) -> float:
    return float(
        0.5
        * (
            (points[1, 0] - points[0, 0]) * (points[2, 1] - points[0, 1])
            - (points[2, 0] - points[0, 0]) * (points[1, 1] - points[0, 1])
        )
    )


def candidate_transform(
    *,
    best_transform: np.ndarray,
    delta: np.ndarray,
    yaw: float,
    scale: float,
    pivot_local: np.ndarray | None = None,
) -> np.ndarray:
    transform = np.asarray(best_transform, dtype=np.float64).copy()
    old_linear = transform[:3, :3].copy()
    old_translation = transform[:3, 3].copy()
    linear = old_linear * float(scale)
    rotation = yaw_rotation_gltf(float(yaw))
    new_linear = rotation @ linear
    pivot = np.asarray(pivot_local if pivot_local is not None else np.zeros(3, dtype=np.float64), dtype=np.float64)
    if pivot.shape != (3,) or not np.isfinite(pivot).all():
        pivot = np.zeros(3, dtype=np.float64)
    pivot_world = old_linear @ pivot + old_translation
    transform[:3, :3] = new_linear
    transform[:3, 3] = pivot_world + np.asarray(delta, dtype=np.float64) - new_linear @ pivot
    return transform


def support_plane_pivot_local(source_bounds: np.ndarray, support_target: dict[str, Any] | None) -> dict[str, Any]:
    support_kind = normalized_support_kind((support_target or {}).get("support_kind"))
    bounds = np.asarray(source_bounds, dtype=np.float64)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or support_kind not in {"floor", "tabletop"}:
        return {
            "method": "normalized_center_pivot",
            "support_kind": support_kind,
            "pivot_source": None,
            "pivot_local": [0.0, 0.0, 0.0],
        }
    pivot_source = np.array(
        [
            (bounds[0, 0] + bounds[1, 0]) / 2.0,
            bounds[0, 1],
            (bounds[0, 2] + bounds[1, 2]) / 2.0,
        ],
        dtype=np.float64,
    )
    pivot_local = transform_points(pivot_source.reshape(1, 3), normalization_transform(bounds))[0]
    return {
        "method": "bottom_center_support_pivot_v1",
        "support_kind": support_kind,
        "pivot_source": pivot_source.tolist(),
        "pivot_local": pivot_local.tolist(),
    }


def support_penalty(source_bounds: np.ndarray, transform: np.ndarray, support_y: float) -> float:
    transformed = transformed_bounds_from_source_bounds(source_bounds, transform)
    vertical_error = abs(float(transformed[0, 1]) - float(support_y))
    return float(vertical_error)


def objective_transform_loss(
    *,
    bbox_loss: float,
    support_loss: float,
    scale_loss: float,
    vggt_loss: Any,
    vggt_loss_weight: float,
    yaw_prior_loss: Any = None,
    facing_prior_loss: Any = None,
    physical_size_prior_loss: Any = None,
    object_avoidance_prior_loss: Any = None,
) -> float:
    total = float(bbox_loss) + float(support_loss) + float(scale_loss)
    if vggt_loss is not None and vggt_loss_weight > 0:
        total += float(vggt_loss) * float(vggt_loss_weight)
    if yaw_prior_loss is not None:
        total += float(yaw_prior_loss) * VGGT_YAW_PRIOR_WEIGHT
    if facing_prior_loss is not None:
        total += float(facing_prior_loss) * MESH_FACING_PRIOR_WEIGHT
    if physical_size_prior_loss is not None:
        total += float(physical_size_prior_loss) * REPEATED_INSTANCE_SIZE_PRIOR_WEIGHT
    if object_avoidance_prior_loss is not None:
        total += float(object_avoidance_prior_loss) * OBJECT_AVOIDANCE_PRIOR_WEIGHT
    return float(total)


def scale_candidates_for_target(
    target_bbox: np.ndarray,
    coordinate_contract: dict[str, Any] | None,
    *,
    initial_bbox: np.ndarray | None = None,
    vggt_fit: dict[str, Any] | None = None,
    source_bounds: np.ndarray | None = None,
    transform: np.ndarray | None = None,
    physical_size_prior: dict[str, Any] | None = None,
) -> tuple[tuple[float, ...], dict[str, Any]]:
    contract = coordinate_contract or {}
    image_height = float(contract.get("image_height") or 0.0)
    target_height = bbox_height(target_bbox) if target_bbox is not None else 0.0
    target_height_ratio = float(target_height / image_height) if image_height > 0 else 0.0
    minimum_scale = min(DEFAULT_UNIFORM_SCALE_CANDIDATES)
    reason = "default"
    if target_height_ratio >= LARGE_TARGET_HEIGHT_RATIO:
        minimum_scale = LARGE_TARGET_MIN_UNIFORM_SCALE
        reason = "large_image_target"
    evidence_scales = evidence_scale_candidates(
        target_bbox=target_bbox,
        initial_bbox=initial_bbox,
        vggt_fit=vggt_fit,
        source_bounds=source_bounds,
        transform=transform,
    )
    if physical_size_prior and physical_size_prior.get("available"):
        scale = physical_size_prior.get("scale_candidate")
        if scale is not None:
            evidence_scales = tuple(list(evidence_scales) + [float(scale)])
    candidate_values = {float(scale) for scale in DEFAULT_UNIFORM_SCALE_CANDIDATES if float(scale) >= minimum_scale}
    for scale in evidence_scales:
        for neighbor in EVIDENCE_SCALE_NEIGHBORS:
            value = float(scale) * float(neighbor)
            if minimum_scale <= value <= MAX_EVIDENCE_SCALE_CANDIDATE:
                candidate_values.add(round(value, 4))
    candidates = tuple(sorted(candidate_values))
    if not candidates:
        candidates = (float(minimum_scale),)
    return candidates, {
        "minimum_scale_delta": float(minimum_scale),
        "reason": reason,
        "target_height_ratio": target_height_ratio,
        "evidence_scale_candidates": [float(value) for value in evidence_scales],
    }


def evidence_scale_candidates(
    *,
    target_bbox: np.ndarray,
    initial_bbox: np.ndarray | None,
    vggt_fit: dict[str, Any] | None,
    source_bounds: np.ndarray | None,
    transform: np.ndarray | None,
) -> tuple[float, ...]:
    scales: list[float] = []
    if initial_bbox is not None:
        target_width = max(float(target_bbox[2] - target_bbox[0]), 1.0)
        target_height = max(float(target_bbox[3] - target_bbox[1]), 1.0)
        initial_width = max(float(initial_bbox[2] - initial_bbox[0]), 1.0)
        initial_height = max(float(initial_bbox[3] - initial_bbox[1]), 1.0)
        scales.extend([target_width / initial_width, target_height / initial_height, np.sqrt((target_width * target_height) / max(initial_width * initial_height, 1.0))])
    if vggt_fit and vggt_fit.get("available") and source_bounds is not None and transform is not None:
        candidate_bounds = transformed_bounds_from_source_bounds(source_bounds, transform)
        candidate_extent = np.maximum(candidate_bounds[1] - candidate_bounds[0], 1e-6)
        visible_extent = np.asarray(vggt_fit.get("extent_gltf"), dtype=np.float64)
        if visible_extent.shape == (3,) and np.isfinite(visible_extent).all():
            active = visible_extent > max(float(np.linalg.norm(visible_extent)) * 0.025, 1e-5)
            ratios = visible_extent[active] / candidate_extent[active]
            if len(ratios) > 0 and np.isfinite(ratios).all():
                scales.append(float(np.median(ratios)))
    clean = []
    for scale in scales:
        value = float(scale)
        if np.isfinite(value) and MIN_EVIDENCE_SCALE_CANDIDATE <= value <= MAX_EVIDENCE_SCALE_CANDIDATE:
            clean.append(round(value, 4))
    return tuple(sorted(set(clean)))


def load_vggt_candidate_fit(placement: dict[str, Any]) -> dict[str, Any]:
    points_path_value = placement.get("visible_points_scene_path")
    if not points_path_value:
        return unavailable_vggt_candidate_fit("missing_visible_points")
    points_path = Path(str(points_path_value))
    if not points_path.is_file():
        return unavailable_vggt_candidate_fit("missing_visible_points_file", points_path=points_path)
    try:
        visible_points = np.load(points_path, allow_pickle=False)
    except Exception:
        return unavailable_vggt_candidate_fit("invalid_visible_points_file", points_path=points_path)
    visible_points = np.asarray(visible_points, dtype=np.float64)
    if visible_points.ndim != 2 or visible_points.shape[1] != 3:
        return unavailable_vggt_candidate_fit("invalid_visible_points_shape", points_path=points_path)
    visible_points = visible_points[np.isfinite(visible_points).all(axis=1)]
    if len(visible_points) == 0:
        return unavailable_vggt_candidate_fit("empty_visible_points", points_path=points_path)
    visible_gltf = scene_points_to_gltf_points(sample_point_rows(visible_points, VGGT_CANDIDATE_POINT_SAMPLE_COUNT))
    visible_bounds = np.stack([visible_gltf.min(axis=0), visible_gltf.max(axis=0)], axis=0)
    visible_extent = visible_bounds[1] - visible_bounds[0]
    visible_diag = max(float(np.linalg.norm(visible_extent)), 1e-6)
    return {
        "available": True,
        "points_gltf": visible_gltf,
        "bounds_gltf": visible_bounds,
        "center_gltf": (visible_bounds[0] + visible_bounds[1]) / 2.0,
        "extent_gltf": visible_extent,
        "diagonal_gltf": visible_diag,
        "report": {
            "method": "visible_vggt_point_aabb_objective_v1",
            "status": "accepted",
            "reason": None,
            "visible_points_scene_path": str(points_path),
            "visible_point_sample_count": int(len(visible_gltf)),
            "visible_bounds_gltf": visible_bounds.tolist(),
        },
    }


def unavailable_vggt_candidate_fit(reason: str, *, points_path: Path | None = None) -> dict[str, Any]:
    return {
        "available": False,
        "points_gltf": None,
        "bounds_gltf": None,
        "center_gltf": None,
        "extent_gltf": None,
        "diagonal_gltf": None,
        "report": {
            "method": "visible_vggt_point_aabb_objective_v1",
            "status": "unavailable",
            "reason": reason,
            "visible_points_scene_path": str(points_path) if points_path is not None else None,
            "visible_point_sample_count": 0,
            "visible_bounds_gltf": None,
        },
    }


def vggt_candidate_transform_loss(
    fit: dict[str, Any],
    source_bounds: np.ndarray,
    transform: np.ndarray,
) -> dict[str, Any]:
    if not fit.get("available"):
        return {
            "status": "unavailable",
            "reason": (fit.get("report") or {}).get("reason"),
            "loss": None,
            "center_loss": None,
            "extent_loss": None,
            "outside_median_loss": None,
            "outside_p90_loss": None,
            "candidate_bounds_gltf": None,
        }
    visible_points = np.asarray(fit["points_gltf"], dtype=np.float64)
    candidate_bounds = transformed_bounds_from_source_bounds(source_bounds, transform)
    visible_center = np.asarray(fit["center_gltf"], dtype=np.float64)
    visible_extent = np.asarray(fit["extent_gltf"], dtype=np.float64)
    diagonal = max(float(fit["diagonal_gltf"]), 1e-6)
    candidate_center = (candidate_bounds[0] + candidate_bounds[1]) / 2.0
    candidate_extent = np.maximum(candidate_bounds[1] - candidate_bounds[0], 1e-6)
    center_loss = float(np.linalg.norm(candidate_center - visible_center) / diagonal)
    active_axes = visible_extent > max(diagonal * 0.025, 1e-5)
    if bool(np.any(active_axes)):
        extent_loss = float(np.mean(np.abs(np.log(candidate_extent[active_axes] / np.maximum(visible_extent[active_axes], 1e-6)))))
    else:
        extent_loss = 0.0
    outside_distances = point_aabb_outside_distances(visible_points, candidate_bounds)
    outside_median = float(np.median(outside_distances) / diagonal)
    outside_p90 = float(np.percentile(outside_distances, 90.0) / diagonal)
    loss = 0.42 * center_loss + 0.43 * extent_loss + 0.15 * outside_p90
    return {
        "status": "accepted",
        "reason": None,
        "loss": float(loss),
        "center_loss": center_loss,
        "extent_loss": extent_loss,
        "outside_median_loss": outside_median,
        "outside_p90_loss": outside_p90,
        "candidate_bounds_gltf": candidate_bounds.tolist(),
    }


def vggt_point_match_transform_loss(
    fit: dict[str, Any],
    *,
    meshes: list[Any],
    source_bounds: np.ndarray,
    transform: np.ndarray,
) -> dict[str, Any]:
    if not fit.get("available"):
        return {
            "status": "unavailable",
            "reason": (fit.get("report") or {}).get("reason"),
            "loss": None,
            "visible_to_mesh_median": None,
            "visible_to_mesh_p90": None,
            "mesh_sample_count": 0,
            "visible_point_sample_count": 0,
        }
    visible_points = sample_point_rows(
        np.asarray(fit["points_gltf"], dtype=np.float64),
        VGGT_POINT_MATCH_TARGET_SAMPLE_COUNT,
    )
    mesh_points = sample_mesh_vertices_for_projection(meshes, VGGT_POINT_MATCH_MESH_SAMPLE_COUNT)
    if len(visible_points) == 0 or len(mesh_points) == 0:
        return {
            "status": "unavailable",
            "reason": "empty_point_sample",
            "loss": None,
            "visible_to_mesh_median": None,
            "visible_to_mesh_p90": None,
            "mesh_sample_count": int(len(mesh_points)),
            "visible_point_sample_count": int(len(visible_points)),
        }
    asset_transform = np.asarray(transform, dtype=np.float64) @ normalization_transform(source_bounds)
    mesh_points = transform_points(mesh_points, asset_transform)
    mesh_points = mesh_points[np.isfinite(mesh_points).all(axis=1)]
    visible_points = visible_points[np.isfinite(visible_points).all(axis=1)]
    if len(visible_points) == 0 or len(mesh_points) == 0:
        return {
            "status": "unavailable",
            "reason": "nonfinite_point_sample",
            "loss": None,
            "visible_to_mesh_median": None,
            "visible_to_mesh_p90": None,
            "mesh_sample_count": int(len(mesh_points)),
            "visible_point_sample_count": int(len(visible_points)),
        }
    diagonal = max(float(fit.get("diagonal_gltf") or 0.0), 1e-6)
    nearest = nearest_point_distances(visible_points, mesh_points)
    median = float(np.median(nearest) / diagonal)
    p90 = float(np.percentile(nearest, 90.0) / diagonal)
    loss = 0.60 * median + 0.40 * p90
    return {
        "status": "accepted",
        "reason": None,
        "method": "visible_vggt_to_mesh_vertex_distance_v1",
        "loss": float(loss),
        "visible_to_mesh_median": median,
        "visible_to_mesh_p90": p90,
        "mesh_sample_count": int(len(mesh_points)),
        "visible_point_sample_count": int(len(visible_points)),
    }


def vggt_candidate_fit_report(
    fit: dict[str, Any],
    *,
    loss_weight: float,
    initial: dict[str, Any],
    optimized: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    report = dict(fit.get("report") or unavailable_vggt_candidate_fit("missing_report")["report"])
    report["loss_weight"] = float(loss_weight)
    report["initial"] = vggt_candidate_loss_summary(initial)
    report["optimized"] = vggt_candidate_loss_summary(optimized)
    report["candidate"] = vggt_candidate_loss_summary(candidate)
    return report


def vggt_candidate_loss_summary(loss: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "status",
        "reason",
        "loss",
        "center_loss",
        "extent_loss",
        "outside_median_loss",
        "outside_p90_loss",
        "candidate_bounds_gltf",
    )
    return {key: loss.get(key) for key in keys}


def load_mask_candidate_fit(placement: dict[str, Any], coordinate_contract: dict[str, Any] | None) -> dict[str, Any]:
    mask_path_value = placement.get("mask_path")
    if not mask_path_value:
        return unavailable_mask_candidate_fit("missing_mask_path")
    mask_path = Path(str(mask_path_value))
    if not mask_path.is_file():
        return unavailable_mask_candidate_fit("missing_mask_file", mask_path=mask_path)
    contract = coordinate_contract or {}
    source_width = int(contract.get("image_width") or 0)
    source_height = int(contract.get("image_height") or 0)
    if source_width <= 0 or source_height <= 0:
        return unavailable_mask_candidate_fit("missing_image_size", mask_path=mask_path)
    try:
        mask_image = Image.open(mask_path).convert("L")
    except Exception:
        return unavailable_mask_candidate_fit("invalid_mask_file", mask_path=mask_path)
    if mask_image.size != (source_width, source_height):
        mask_image = mask_image.resize((source_width, source_height), Image.Resampling.NEAREST)
    render_scale = min(1.0, MASK_CANDIDATE_RENDER_MAX_SIZE / float(max(source_width, source_height)))
    render_width = max(1, int(round(source_width * render_scale)))
    render_height = max(1, int(round(source_height * render_scale)))
    if (render_width, render_height) != mask_image.size:
        mask_image = mask_image.resize((render_width, render_height), Image.Resampling.NEAREST)
    mask = np.asarray(mask_image, dtype=np.uint8) > 127
    if not bool(mask.any()):
        return unavailable_mask_candidate_fit("empty_mask", mask_path=mask_path)
    render_contract = dict(contract)
    render_contract["image_width"] = render_width
    render_contract["image_height"] = render_height
    return {
        "available": True,
        "mask": mask,
        "coordinate_contract": render_contract,
        "width": render_width,
        "height": render_height,
        "report": {
            "method": "software_projected_mesh_mask_candidate_objective_v1",
            "status": "accepted",
            "reason": None,
            "mask_path": str(mask_path),
            "source_image_size": [source_width, source_height],
            "render_size": [render_width, render_height],
            "target_area_px": int(mask.sum()),
        },
    }


def unavailable_mask_candidate_fit(reason: str, *, mask_path: Path | None = None) -> dict[str, Any]:
    return {
        "available": False,
        "mask": None,
        "coordinate_contract": None,
        "width": None,
        "height": None,
        "report": {
            "method": "software_projected_mesh_mask_candidate_objective_v1",
            "status": "unavailable",
            "reason": reason,
            "mask_path": str(mask_path) if mask_path is not None else None,
            "source_image_size": None,
            "render_size": None,
            "target_area_px": None,
        },
    }


def object_avoidance_prior_from_bounds(value: Any) -> dict[str, Any]:
    bounds_items = value if isinstance(value, list) else []
    records: list[dict[str, Any]] = []
    for item in bounds_items:
        bounds_value = item.get("bounds_gltf") if isinstance(item, dict) else item
        bounds = bounds_array(bounds_value)
        if bounds is None:
            continue
        records.append(
            {
                "bounds_gltf": bounds.tolist(),
                "detection_id": item.get("detection_id") if isinstance(item, dict) else None,
                "detector_label": item.get("detector_label") if isinstance(item, dict) else None,
                "reason": item.get("reason") if isinstance(item, dict) else None,
            }
        )
    return {
        "available": bool(records),
        "method": "occupied_bounds_avoidance_prior_v1",
        "avoid_bounds": records,
    }


def object_avoidance_prior_loss(prior: dict[str, Any], source_bounds: np.ndarray, transform: np.ndarray) -> float | None:
    if not prior.get("available"):
        return None
    candidate_bounds = transformed_bounds_from_source_bounds(source_bounds, transform)
    candidate_extent = np.maximum(candidate_bounds[1] - candidate_bounds[0], 1e-8)
    candidate_volume = float(np.prod(candidate_extent))
    candidate_footprint = float(candidate_extent[0] * candidate_extent[2])
    losses = []
    for item in prior.get("avoid_bounds") or []:
        avoid_bounds = bounds_array(item.get("bounds_gltf"))
        if avoid_bounds is None:
            continue
        overlap_extent = np.maximum(0.0, np.minimum(candidate_bounds[1], avoid_bounds[1]) - np.maximum(candidate_bounds[0], avoid_bounds[0]))
        overlap_volume = float(np.prod(overlap_extent))
        overlap_footprint = float(overlap_extent[0] * overlap_extent[2])
        volume_ratio = overlap_volume / max(candidate_volume, 1e-8)
        footprint_ratio = overlap_footprint / max(candidate_footprint, 1e-8)
        if volume_ratio > 0.0 or footprint_ratio > 0.0:
            losses.append(0.70 * volume_ratio + 0.30 * footprint_ratio)
    if not losses:
        return 0.0
    return float(sum(losses))


def object_avoidance_prior_report(
    prior: dict[str, Any],
    *,
    initial: float | None,
    optimized: float | None,
    candidate: float | None,
) -> dict[str, Any]:
    if not prior.get("available"):
        return {
            "status": "unavailable",
            "reason": "missing_avoid_bounds",
            "method": "occupied_bounds_avoidance_prior_v1",
            "avoid_count": 0,
            "loss_weight": 0.0,
            "initial_loss": None,
            "optimized_loss": None,
            "candidate_loss": None,
        }
    return {
        "status": "accepted",
        "reason": None,
        "method": prior.get("method"),
        "avoid_count": len(prior.get("avoid_bounds") or []),
        "avoid_bounds": prior.get("avoid_bounds"),
        "loss_weight": float(OBJECT_AVOIDANCE_PRIOR_WEIGHT),
        "initial_loss": float(initial) if initial is not None else None,
        "optimized_loss": float(optimized) if optimized is not None else None,
        "candidate_loss": float(candidate) if candidate is not None else None,
    }


def translation_candidates_for_avoidance(
    source_bounds: np.ndarray,
    transform: np.ndarray,
    prior: dict[str, Any],
) -> tuple[tuple[float, ...], tuple[float, ...], dict[str, Any]]:
    dx_values = {float(value) for value in DEFAULT_TRANSLATION_X_CANDIDATES}
    dz_values = {float(value) for value in DEFAULT_TRANSLATION_Z_CANDIDATES}
    added: list[dict[str, Any]] = []
    if prior.get("available"):
        candidate_bounds = transformed_bounds_from_source_bounds(source_bounds, transform)
        extent = np.maximum(candidate_bounds[1] - candidate_bounds[0], 1e-8)
        clearance = max(min(float(extent[0]), float(extent[2])) * 0.08, 0.015)
        for item in prior.get("avoid_bounds") or []:
            avoid_bounds = bounds_array(item.get("bounds_gltf"))
            if avoid_bounds is None:
                continue
            overlap_extent = np.maximum(0.0, np.minimum(candidate_bounds[1], avoid_bounds[1]) - np.maximum(candidate_bounds[0], avoid_bounds[0]))
            if float(overlap_extent[1]) <= 1e-8 or float(overlap_extent[0] * overlap_extent[2]) <= 1e-8:
                continue
            suggestions = [
                ("x_min", float(avoid_bounds[0, 0] - candidate_bounds[1, 0] - clearance), dx_values),
                ("x_max", float(avoid_bounds[1, 0] - candidate_bounds[0, 0] + clearance), dx_values),
                ("z_min", float(avoid_bounds[0, 2] - candidate_bounds[1, 2] - clearance), dz_values),
                ("z_max", float(avoid_bounds[1, 2] - candidate_bounds[0, 2] + clearance), dz_values),
            ]
            for axis, value, bucket in suggestions:
                rounded = round(value, 4)
                if np.isfinite(rounded):
                    bucket.add(float(rounded))
                    added.append(
                        {
                            "axis": axis,
                            "delta": float(rounded),
                            "avoid_detection_id": item.get("detection_id"),
                            "reason": "clear_occupied_bounds_overlap",
                        }
                    )
    return tuple(sorted(dx_values)), tuple(sorted(dz_values)), {
        "method": "occupied_bounds_clearance_translation_candidates_v1",
        "added_candidate_count": len(added),
        "added_candidates": added,
    }


def translation_candidates_for_projection_residual(
    *,
    source_bounds: np.ndarray,
    projection_vertices: np.ndarray,
    transform: np.ndarray,
    target_bbox: np.ndarray,
    coordinate_contract: dict[str, Any] | None,
    support_y: float,
    pivot_local: Any = None,
) -> tuple[tuple[float, ...], tuple[float, ...], dict[str, Any]]:
    base_bbox = projected_mesh_sample_bbox(projection_vertices, source_bounds, transform, coordinate_contract)
    if base_bbox is None:
        base_bbox = projected_transform_bbox(source_bounds, transform, coordinate_contract)
    if base_bbox is None:
        return (), (), unavailable_projection_translation_report("missing_base_projection")

    candidate_bounds = transformed_bounds_from_source_bounds(source_bounds, transform)
    extent = np.maximum(candidate_bounds[1] - candidate_bounds[0], 1e-8)
    horizontal_extent = max(float(extent[0]), float(extent[2]), 1e-8)
    step = float(np.clip(horizontal_extent * PROJECTION_TRANSLATION_STEP_RATIO, PROJECTION_TRANSLATION_MIN_STEP, PROJECTION_TRANSLATION_MAX_STEP))
    base_center = bbox_center_2d(base_bbox)
    target_center = bbox_center_2d(target_bbox)
    residual = target_center - base_center
    axes: list[tuple[str, np.ndarray, np.ndarray]] = []
    for axis, delta in (
        ("x", np.array([step, 0.0, 0.0], dtype=np.float64)),
        ("z", np.array([0.0, 0.0, step], dtype=np.float64)),
    ):
        candidate = candidate_transform(
            best_transform=transform,
            delta=delta,
            yaw=0.0,
            scale=1.0,
            pivot_local=pivot_local,
        )
        candidate, _snap_delta = snap_transform_to_support_bounds(source_bounds, candidate, support_y)
        bbox = projected_mesh_sample_bbox(projection_vertices, source_bounds, candidate, coordinate_contract)
        if bbox is None:
            bbox = projected_transform_bbox(source_bounds, candidate, coordinate_contract)
        if bbox is None:
            continue
        response = (bbox_center_2d(bbox) - base_center) / step
        if np.isfinite(response).all() and float(np.linalg.norm(response)) > 1e-6:
            axes.append((axis, delta, response))
    if len(axes) < 2:
        return (), (), unavailable_projection_translation_report(
            "insufficient_projection_jacobian",
            base_bbox=base_bbox,
            residual=residual,
        )

    jacobian = np.column_stack([item[2] for item in axes])
    try:
        solution, *_unused = np.linalg.lstsq(jacobian, residual, rcond=None)
    except np.linalg.LinAlgError:
        return (), (), unavailable_projection_translation_report(
            "singular_projection_jacobian",
            base_bbox=base_bbox,
            residual=residual,
            jacobian=jacobian,
        )
    if solution.shape != (2,) or not np.isfinite(solution).all():
        return (), (), unavailable_projection_translation_report(
            "invalid_projection_translation_solution",
            base_bbox=base_bbox,
            residual=residual,
            jacobian=jacobian,
        )

    max_delta = max(horizontal_extent * 2.5, step * 2.0)
    if bool(np.any(np.abs(solution) > max_delta)):
        return (), (), unavailable_projection_translation_report(
            "projection_translation_solution_exceeds_local_window",
            base_bbox=base_bbox,
            residual=residual,
            jacobian=jacobian,
            solution=solution,
            max_delta=max_delta,
        )
    clipped = np.clip(solution, -max_delta, max_delta)
    dx_values: set[float] = set()
    dz_values: set[float] = set()
    added: list[dict[str, Any]] = []
    for axis_index, (axis, _delta, _response) in enumerate(axes):
        value = float(clipped[axis_index])
        bucket = dx_values if axis == "x" else dz_values
        for neighbor in PROJECTION_TRANSLATION_NEIGHBORS:
            candidate_value = round(value * float(neighbor), 4)
            if not np.isfinite(candidate_value) or abs(candidate_value) <= 1e-8:
                continue
            bucket.add(float(candidate_value))
            added.append(
                {
                    "axis": axis,
                    "delta": float(candidate_value),
                    "reason": "projected_bbox_center_residual",
                    "neighbor": float(neighbor),
                }
            )
    return tuple(sorted(dx_values)), tuple(sorted(dz_values)), {
        "method": "projected_bbox_residual_planar_translation_candidates_v1",
        "status": "accepted",
        "reason": None,
        "step_gltf": step,
        "base_projected_bbox_xyxy": base_bbox.tolist(),
        "target_center_px": target_center.tolist(),
        "base_center_px": base_center.tolist(),
        "residual_px": residual.tolist(),
        "jacobian_px_per_gltf": jacobian.tolist(),
        "solution_delta_gltf": [float(value) for value in solution],
        "clipped_delta_gltf": [float(value) for value in clipped],
        "added_candidate_count": len(added),
        "added_candidates": added,
    }


def unavailable_projection_translation_report(
    reason: str,
    *,
    base_bbox: np.ndarray | None = None,
    residual: np.ndarray | None = None,
    jacobian: np.ndarray | None = None,
    solution: np.ndarray | None = None,
    max_delta: float | None = None,
) -> dict[str, Any]:
    return {
        "method": "projected_bbox_residual_planar_translation_candidates_v1",
        "status": "unavailable",
        "reason": reason,
        "base_projected_bbox_xyxy": base_bbox.tolist() if base_bbox is not None else None,
        "residual_px": residual.tolist() if residual is not None else None,
        "jacobian_px_per_gltf": jacobian.tolist() if jacobian is not None else None,
        "solution_delta_gltf": solution.tolist() if solution is not None else None,
        "max_delta_gltf": float(max_delta) if max_delta is not None else None,
        "added_candidate_count": 0,
        "added_candidates": [],
    }


def mask_candidate_transform_loss(
    fit: dict[str, Any],
    *,
    meshes: list[Any],
    source_bounds: np.ndarray,
    transform: np.ndarray,
) -> dict[str, Any]:
    if not fit.get("available"):
        return {
            "status": "unavailable",
            "reason": (fit.get("report") or {}).get("reason"),
            "loss": None,
            "iou": None,
            "false_positive_area_ratio": None,
            "false_negative_area_ratio": None,
            "rendered_area_px": None,
            "target_area_px": None,
            "intersection_area_px": None,
            "union_area_px": None,
            "rendered_face_count": 0,
        }
    target = np.asarray(fit["mask"], dtype=bool)
    rendered, face_count = render_mesh_candidate_mask(
        meshes=meshes,
        source_bounds=source_bounds,
        transform=transform,
        coordinate_contract=fit["coordinate_contract"],
        width=int(fit["width"]),
        height=int(fit["height"]),
    )
    rendered_area = int(rendered.sum())
    target_area = int(target.sum())
    if rendered_area == 0:
        return {
            "status": "unavailable",
            "reason": "empty_rendered_mask",
            "loss": None,
            "iou": None,
            "false_positive_area_ratio": None,
            "false_negative_area_ratio": None,
            "rendered_area_px": 0,
            "target_area_px": target_area,
            "intersection_area_px": 0,
            "union_area_px": int(target_area),
            "rendered_face_count": int(face_count),
        }
    intersection = int(np.logical_and(rendered, target).sum())
    union = int(np.logical_or(rendered, target).sum())
    false_positive = int(np.logical_and(rendered, ~target).sum())
    false_negative = int(np.logical_and(~rendered, target).sum())
    iou = float(intersection / union) if union else 0.0
    return {
        "status": "accepted",
        "reason": None,
        "loss": float(1.0 - iou),
        "iou": iou,
        "false_positive_area_ratio": float(false_positive / max(rendered_area, 1)),
        "false_negative_area_ratio": float(false_negative / max(target_area, 1)),
        "rendered_area_px": rendered_area,
        "target_area_px": target_area,
        "intersection_area_px": intersection,
        "union_area_px": union,
        "rendered_face_count": int(face_count),
    }


def render_mesh_candidate_mask(
    *,
    meshes: list[Any],
    source_bounds: np.ndarray,
    transform: np.ndarray,
    coordinate_contract: dict[str, Any],
    width: int,
    height: int,
) -> tuple[np.ndarray, int]:
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    asset_transform = np.asarray(transform, dtype=np.float64) @ normalization_transform(source_bounds)
    face_count = 0
    for mesh in meshes:
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0 or len(faces) == 0:
            continue
        projected, valid = project_gltf_points_to_pixels(transform_points(vertices, asset_transform), coordinate_contract)
        for face in sample_point_rows(faces, MASK_CANDIDATE_FACE_SAMPLE_COUNT):
            indices = np.asarray(face, dtype=np.int64)
            if indices.shape != (3,) or not bool(valid[indices].all()):
                continue
            points = projected[indices]
            min_x = float(points[:, 0].min())
            max_x = float(points[:, 0].max())
            min_y = float(points[:, 1].min())
            max_y = float(points[:, 1].max())
            if max_x < 0 or max_y < 0 or min_x >= width or min_y >= height:
                continue
            if abs(triangle_area_2d(points)) < 0.05:
                continue
            draw.polygon([(float(x), float(y)) for x, y in points], fill=255)
            face_count += 1
    return np.asarray(image, dtype=np.uint8) > 0, face_count


def unavailable_mask_candidate_report(
    fit: dict[str, Any],
    *,
    fallback_reason: str | None,
    candidate_count: int,
    evaluated_count: int = 0,
) -> dict[str, Any]:
    base = dict((fit.get("report") or unavailable_mask_candidate_fit("missing_report")["report"]))
    base.update(
        status="unavailable",
        candidate_count=int(candidate_count),
        evaluated_count=int(evaluated_count),
        selected_yaw=None,
        selected_scale=None,
        base_loss=None,
        combined_loss=None,
        loss_weight=float(MASK_CANDIDATE_LOSS_WEIGHT if fit.get("available") else 0.0),
        selected_projected_bbox_xyxy=None,
        fallback_reason=fallback_reason,
        candidate=None,
    )
    return base


def mask_candidate_fit_report(
    fit: dict[str, Any],
    *,
    loss_weight: float,
    initial: dict[str, Any],
    optimized: dict[str, Any],
    candidate: dict[str, Any],
    candidate_count: int,
    selected_yaw: float,
    selected_scale: float,
    fallback_reason: str | None,
) -> dict[str, Any]:
    report = dict(fit.get("report") or unavailable_mask_candidate_fit("missing_report")["report"])
    report["loss_weight"] = float(loss_weight)
    report["candidate_count"] = int(candidate_count)
    report["selected_yaw"] = float(selected_yaw)
    report["selected_scale"] = float(selected_scale)
    report["initial"] = mask_candidate_loss_summary(initial)
    report["optimized"] = mask_candidate_loss_summary(optimized)
    report["candidate"] = mask_candidate_loss_summary(candidate)
    report["fallback_reason"] = fallback_reason
    return report


def mask_candidate_loss_summary(loss: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "status",
        "reason",
        "loss",
        "iou",
        "false_positive_area_ratio",
        "false_negative_area_ratio",
        "rendered_area_px",
        "target_area_px",
        "intersection_area_px",
        "union_area_px",
        "rendered_face_count",
        "selected_projected_bbox_xyxy",
        "base_loss",
        "combined_loss",
        "vggt_point_match",
        "vggt_point_match_loss_weight",
    )
    return {key: loss.get(key) for key in keys}


def point_aabb_outside_distances(points: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    lower = np.maximum(bounds[0] - points, 0.0)
    upper = np.maximum(points - bounds[1], 0.0)
    offsets = np.maximum(lower, upper)
    return np.sqrt(np.sum(offsets * offsets, axis=1))


def nearest_point_distances(source: np.ndarray, target: np.ndarray, *, chunk_size: int = 128) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if len(source) == 0 or len(target) == 0:
        return np.empty((0,), dtype=np.float64)
    distances: list[np.ndarray] = []
    for start in range(0, len(source), chunk_size):
        chunk = source[start : start + chunk_size]
        delta = chunk[:, None, :] - target[None, :, :]
        squared = np.sum(delta * delta, axis=2)
        distances.append(np.sqrt(np.min(squared, axis=1)))
    return np.concatenate(distances, axis=0)


def sample_point_rows(values: np.ndarray, max_count: int) -> np.ndarray:
    if len(values) <= max_count:
        return np.asarray(values, dtype=np.float64)
    indices = np.linspace(0, len(values) - 1, max_count, dtype=np.int64)
    return np.asarray(values[indices], dtype=np.float64)


def scene_points_to_gltf_points(points: np.ndarray) -> np.ndarray:
    return np.asarray([[x, z, -y] for x, y, z in points], dtype=np.float64)


def sample_mesh_vertices_for_projection(meshes: list[Any], max_count: int) -> np.ndarray:
    vertices: list[np.ndarray] = []
    for mesh in meshes:
        mesh_vertices = np.asarray(mesh.vertices, dtype=np.float64)
        if mesh_vertices.ndim == 2 and mesh_vertices.shape[1] == 3 and len(mesh_vertices) > 0:
            vertices.append(mesh_vertices[np.isfinite(mesh_vertices).all(axis=1)])
    vertices = [item for item in vertices if len(item) > 0]
    if not vertices:
        return np.empty((0, 3), dtype=np.float64)
    combined = np.concatenate(vertices, axis=0)
    return sample_point_rows(combined, max_count)


def projected_mesh_sample_bbox(
    vertices: np.ndarray,
    source_bounds: np.ndarray,
    transform: np.ndarray,
    coordinate_contract: dict[str, Any] | None,
) -> np.ndarray | None:
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        return None
    asset_transform = transform @ normalization_transform(source_bounds)
    transformed = transform_points(vertices, asset_transform)
    pixels, valid = project_gltf_points_to_pixels(transformed, coordinate_contract)
    pixels = pixels[valid]
    if len(pixels) == 0:
        return None
    return np.asarray([pixels[:, 0].min(), pixels[:, 1].min(), pixels[:, 0].max(), pixels[:, 1].max()], dtype=np.float64)


def project_gltf_points_to_pixels(points: np.ndarray, coordinate_contract: dict[str, Any] | None) -> tuple[np.ndarray, np.ndarray]:
    contract = coordinate_contract or {}
    width = int(contract.get("image_width") or 0)
    height = int(contract.get("image_height") or 0)
    if width <= 0 or height <= 0:
        return np.zeros((len(points), 2), dtype=np.float64), np.zeros((len(points),), dtype=bool)
    fov = float(contract.get("fov_degrees", DEFAULT_FOV_DEGREES))
    points = np.asarray(points, dtype=np.float64)
    x = points[:, 0]
    scene_z = points[:, 1]
    depth = -points[:, 2]
    valid = (depth > 1e-6) & np.isfinite(points).all(axis=1)
    safe_depth = np.where(valid, depth, 1.0)
    focal = (width / 2.0) / np.tan(np.deg2rad(fov) / 2.0)
    pixels = np.zeros((len(points), 2), dtype=np.float64)
    pixels[:, 0] = width / 2.0 + (x / safe_depth) * focal
    pixels[:, 1] = height / 2.0 - (scene_z / safe_depth) * focal
    return pixels, valid


def projected_transform_bbox(
    source_bounds: np.ndarray,
    transform: np.ndarray,
    coordinate_contract: dict[str, Any] | None,
) -> np.ndarray | None:
    asset_transform = transform @ normalization_transform(source_bounds)
    transformed = transform_points(bounds_corners(source_bounds), asset_transform)
    pixels = [project_gltf_point_to_pixel(point, coordinate_contract) for point in transformed]
    pixels = [pixel for pixel in pixels if pixel is not None]
    if not pixels:
        return None
    array = np.asarray(pixels, dtype=np.float64)
    return np.asarray([array[:, 0].min(), array[:, 1].min(), array[:, 0].max(), array[:, 1].max()], dtype=np.float64)


def project_gltf_point_to_pixel(point: np.ndarray, coordinate_contract: dict[str, Any] | None) -> tuple[float, float] | None:
    contract = coordinate_contract or {}
    width = int(contract.get("image_width") or 0)
    height = int(contract.get("image_height") or 0)
    if width <= 0 or height <= 0:
        return None
    fov = float(contract.get("fov_degrees", DEFAULT_FOV_DEGREES))
    x = float(point[0])
    scene_z = float(point[1])
    depth = float(-point[2])
    if depth <= 1e-6 or not np.isfinite([x, scene_z, depth]).all():
        return None
    focal = (width / 2.0) / np.tan(np.deg2rad(fov) / 2.0)
    pixel_x = width / 2.0 + (x / depth) * focal
    pixel_y = height / 2.0 - (scene_z / depth) * focal
    return (float(pixel_x), float(pixel_y))


def bbox_projection_loss(projected: np.ndarray, target: np.ndarray) -> float:
    iou = bbox_iou(projected, target)
    projected_center = bbox_center_2d(projected)
    target_center = bbox_center_2d(target)
    diagonal = max(float(np.linalg.norm([target[2] - target[0], target[3] - target[1]])), 1.0)
    center_loss = float(np.linalg.norm(projected_center - target_center) / diagonal)
    projected_area = max(float((projected[2] - projected[0]) * (projected[3] - projected[1])), 1.0)
    target_area = max(float((target[2] - target[0]) * (target[3] - target[1])), 1.0)
    area_loss = abs(np.log(projected_area / target_area))
    target_height = max(float(target[3] - target[1]), 1.0)
    edge_loss = (abs(float(projected[1] - target[1])) + abs(float(projected[3] - target[3]))) / target_height
    return float((1.0 - iou) + 0.55 * center_loss + 0.35 * area_loss + 0.45 * edge_loss)


def bbox_center_2d(bbox: np.ndarray) -> np.ndarray:
    array = np.asarray(bbox, dtype=np.float64)
    return np.array([(array[0] + array[2]) / 2.0, (array[1] + array[3]) / 2.0], dtype=np.float64)


def projection_quality_report(
    projected: np.ndarray | None,
    target: np.ndarray | None,
    *,
    accepted: bool | None = None,
    allow_occluded_bottom: bool = False,
) -> dict[str, Any]:
    if projected is None or target is None:
        return {
            "status": "unavailable",
            "accepted": None,
            "reason": "missing_projection_or_target",
            "vertical_edge_error_ratio": None,
            "threshold": PROJECTION_VERTICAL_EDGE_REJECT_RATIO,
            "occluded_bottom_threshold": PROJECTION_OCCLUDED_BOTTOM_EDGE_REVIEW_RATIO,
        }
    target_width = max(float(target[2] - target[0]), 1.0)
    target_height = max(float(target[3] - target[1]), 1.0)
    projected_width = max(float(projected[2] - projected[0]), 1.0)
    projected_height = max(float(projected[3] - projected[1]), 1.0)
    projected_area = projected_width * projected_height
    target_area = target_width * target_height
    top_error = abs(float(projected[1] - target[1]))
    bottom_error = abs(float(projected[3] - target[3]))
    left_error = abs(float(projected[0] - target[0]))
    right_error = abs(float(projected[2] - target[2]))
    center_x_error = abs(float((projected[0] + projected[2] - target[0] - target[2]) / 2.0))
    center_y_error = abs(float((projected[1] + projected[3] - target[1] - target[3]) / 2.0))
    top_ratio = top_error / target_height
    bottom_ratio = bottom_error / target_height
    left_ratio = left_error / target_width
    right_ratio = right_error / target_width
    center_x_ratio = center_x_error / target_width
    center_y_ratio = center_y_error / target_height
    width_error_ratio = abs(projected_width - target_width) / target_width
    height_error_ratio = abs(projected_height - target_height) / target_height
    area_ratio = projected_area / max(target_area, 1.0)
    edge_ratio = max(top_error, bottom_error) / target_height
    horizontal_edge_ratio = max(left_ratio, right_ratio, width_error_ratio)
    occluded_bottom_accepted = (
        accepted is None
        and allow_occluded_bottom
        and top_ratio <= PROJECTION_OCCLUDED_BOTTOM_TOP_EDGE_REVIEW_RATIO
        and float(projected[3]) > float(target[3])
        and bottom_ratio <= PROJECTION_OCCLUDED_BOTTOM_EDGE_REVIEW_RATIO
        and bottom_ratio > PROJECTION_VERTICAL_EDGE_REJECT_RATIO
        and horizontal_edge_ratio <= PROJECTION_HORIZONTAL_EDGE_REJECT_RATIO
        and center_x_ratio <= PROJECTION_CENTER_REJECT_RATIO
        and center_y_ratio <= PROJECTION_CENTER_REJECT_RATIO
        and area_ratio <= PROJECTION_OCCLUDED_BOTTOM_AREA_REJECT_RATIO
    )
    area_error_ratio = abs(area_ratio - 1.0)
    rejected = (
        (
            edge_ratio > PROJECTION_VERTICAL_EDGE_REJECT_RATIO
            or horizontal_edge_ratio > PROJECTION_HORIZONTAL_EDGE_REJECT_RATIO
            or area_error_ratio > PROJECTION_AREA_ERROR_REJECT_RATIO
            or center_y_ratio > PROJECTION_CENTER_Y_REJECT_RATIO
        )
        and not occluded_bottom_accepted
        if accepted is None
        else not bool(accepted)
    )
    if occluded_bottom_accepted:
        status = "accepted_occluded_bottom"
        reason = "occluded_bottom_edge_tolerated"
    else:
        status = "rejected" if rejected else "accepted"
        if rejected and area_error_ratio > PROJECTION_AREA_ERROR_REJECT_RATIO:
            reason = "area_error"
        elif rejected and edge_ratio > PROJECTION_VERTICAL_EDGE_REJECT_RATIO:
            reason = "vertical_edge_error"
        elif rejected and center_y_ratio > PROJECTION_CENTER_Y_REJECT_RATIO:
            reason = "vertical_center_error"
        elif rejected:
            reason = "horizontal_edge_error"
        else:
            reason = "within_threshold"
    return {
        "status": status,
        "accepted": not rejected,
        "reason": reason,
        "vertical_edge_error_ratio": float(edge_ratio),
        "horizontal_edge_error_ratio": float(horizontal_edge_ratio),
        "top_error_ratio": float(top_ratio),
        "bottom_error_ratio": float(bottom_ratio),
        "left_error_ratio": float(left_ratio),
        "right_error_ratio": float(right_ratio),
        "center_x_error_ratio": float(center_x_ratio),
        "center_y_error_ratio": float(center_y_ratio),
        "width_error_ratio": float(width_error_ratio),
        "height_error_ratio": float(height_error_ratio),
        "area_ratio": float(area_ratio),
        "area_error_ratio": float(area_error_ratio),
        "top_error_px": float(top_error),
        "bottom_error_px": float(bottom_error),
        "left_error_px": float(left_error),
        "right_error_px": float(right_error),
        "center_x_error_px": float(center_x_error),
        "center_y_error_px": float(center_y_error),
        "target_height_px": float(target_height),
        "target_width_px": float(target_width),
        "threshold": PROJECTION_VERTICAL_EDGE_REJECT_RATIO,
        "occluded_bottom_threshold": PROJECTION_OCCLUDED_BOTTOM_EDGE_REVIEW_RATIO,
        "occluded_bottom_top_threshold": PROJECTION_OCCLUDED_BOTTOM_TOP_EDGE_REVIEW_RATIO,
        "horizontal_threshold": PROJECTION_HORIZONTAL_EDGE_REJECT_RATIO,
        "center_threshold": PROJECTION_CENTER_REJECT_RATIO,
        "center_y_threshold": PROJECTION_CENTER_Y_REJECT_RATIO,
        "occluded_bottom_area_threshold": PROJECTION_OCCLUDED_BOTTOM_AREA_REJECT_RATIO,
        "area_error_threshold": PROJECTION_AREA_ERROR_REJECT_RATIO,
    }


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = bbox_overlap_area(a, b)
    area_a = max(0.0, float((a[2] - a[0]) * (a[3] - a[1])))
    area_b = max(0.0, float((b[2] - b[0]) * (b[3] - b[1])))
    union = area_a + area_b - intersection
    if union <= 1e-8:
        return 0.0
    return float(intersection / union)


def write_projection_overlay(source_image_path: Path, records: list[dict[str, Any]], output_path: Path) -> None:
    if not source_image_path.is_file():
        return
    image = Image.open(source_image_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    for record in records:
        optimization = record.get("render_to_input_optimization") or {}
        target = optimization.get("target_bbox_xyxy")
        projected = optimization.get("optimized_projected_bbox_xyxy")
        if target:
            draw.rectangle([float(v) for v in target], outline=(255, 220, 0, 220), width=3)
        if projected:
            draw.rectangle([float(v) for v in projected], outline=(0, 220, 255, 220), width=3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def suppressed_objects_report(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    suppressed: list[dict[str, Any]] = []
    for record in records:
        if record.get("reason") != "suppressed_by_composite":
            continue
        suppressed.append(
            {
                "detection_id": record.get("detection_id"),
                "detector_label": record.get("detector_label"),
                "composite_id": record.get("composite_id"),
                "suppressed_by_composite": record.get("suppressed_by_composite"),
            }
        )
    return suppressed


def projection_quality_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    rejected = [
        {
            "detection_id": record.get("detection_id"),
            "detector_label": record.get("detector_label"),
            "reason": (record.get("projection_quality") or {}).get("reason"),
        }
        for record in records
        if (record.get("projection_quality") or {}).get("status") == "rejected"
    ]
    return {
        "rejected_count": len(rejected),
        "rejected": rejected,
    }


def object_overlap_warnings_report(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    composed = [record for record in records if record.get("status") == "composed" and record.get("transformed_bounds")]
    warnings: list[dict[str, Any]] = []
    for left_index, left in enumerate(composed):
        left_bounds = bounds_array(left.get("transformed_bounds"))
        if left_bounds is None:
            continue
        for right in composed[left_index + 1 :]:
            right_bounds = bounds_array(right.get("transformed_bounds"))
            if right_bounds is None:
                continue
            overlap_min = np.maximum(left_bounds[0], right_bounds[0])
            overlap_max = np.minimum(left_bounds[1], right_bounds[1])
            overlap_extent = np.maximum(0.0, overlap_max - overlap_min)
            overlap_volume = float(np.prod(overlap_extent))
            if overlap_volume <= 1e-8:
                continue
            warnings.append(
                {
                    "detection_ids": [left.get("detection_id"), right.get("detection_id")],
                    "labels": [left.get("detector_label"), right.get("detector_label")],
                    "overlap_extent_gltf": [float(value) for value in overlap_extent],
                    "overlap_volume_gltf": overlap_volume,
                    "tabletop_pair": bool(
                        is_tabletop_object_label(str(left.get("detector_label") or ""))
                        or is_tabletop_object_label(str(right.get("detector_label") or ""))
                    ),
                }
            )
    return warnings


def object_support_targets(
    placements: list[dict[str, Any]],
    *,
    object_dirs: dict[int, Path],
    object_mesh_name: str,
    include_review: bool,
    placement_orientation: str,
    object_scale_factor: float,
    floor_y: float | None,
) -> dict[int, dict[str, Any]]:
    if floor_y is None:
        return {}

    targets: dict[int, dict[str, Any]] = {}
    table_candidates: list[dict[str, Any]] = []
    for placement in placements:
        detection_id = int(placement.get("detection_id", 0))
        if not placement_is_composable(placement, include_review=include_review):
            continue
        object_dir = object_dirs.get(object_dir_id_for_placement(placement))
        if object_dir is None:
            continue
        mesh_path = resolve_object_mesh_path(object_dir, object_mesh_name)
        if mesh_path is None:
            continue

        label = str(placement.get("detector_label") or "")
        transform = placement_transform_to_gltf(
            placement,
            placement_orientation=placement_orientation,
            object_scale_factor=object_scale_factor,
        )
        support_kind = "floor"
        support_detection_id = None
        support_y = float(floor_y)
        if is_table_support_label(label):
            snapped, _delta = snap_transform_to_support(
                mesh_path,
                transform,
                floor_y,
                support_kind="floor",
            )
            table_bounds = transformed_mesh_bounds(mesh_path, snapped)
            table_candidates.append(
                {
                    "detection_id": detection_id,
                    "bbox_xyxy": placement.get("bbox_xyxy"),
                    "support_y": float(table_bounds[1, 1]),
                }
            )
        targets[detection_id] = {
            "support_kind": support_kind,
            "support_detection_id": support_detection_id,
            "support_y": support_y,
        }

    for placement in placements:
        detection_id = int(placement.get("detection_id", 0))
        if detection_id not in targets:
            continue
        if placement.get("suppressed_by_composite"):
            continue
        label = str(placement.get("detector_label") or "")
        if not is_tabletop_object_label(label):
            continue
        table = best_table_support(placement.get("bbox_xyxy"), table_candidates)
        if table is None:
            continue
        targets[detection_id] = {
            "support_kind": "tabletop",
            "support_detection_id": int(table["detection_id"]),
            "support_y": float(table["support_y"]),
        }
    return targets


def yaw_rotation_gltf(yaw: float) -> np.ndarray:
    cos_value = float(np.cos(yaw))
    sin_value = float(np.sin(yaw))
    return np.asarray(
        [
            [cos_value, 0.0, sin_value],
            [0.0, 1.0, 0.0],
            [-sin_value, 0.0, cos_value],
        ],
        dtype=np.float64,
    )


def object_dir_id_for_placement(placement: dict[str, Any]) -> int:
    return int(placement.get("source_object_dir_id") or placement.get("detection_id", 0))


def placement_is_composable(placement: dict[str, Any], *, include_review: bool) -> bool:
    if placement.get("box_type") == "failed":
        return False
    if placement.get("suppressed_by_composite"):
        return False
    if bool(placement.get("needs_review", False)) and not include_review:
        return False
    return True


def is_table_support_label(label: str) -> bool:
    normalized = label.lower()
    return any(token in normalized for token in TABLE_SUPPORT_LABELS)


def is_tabletop_object_label(label: str) -> bool:
    normalized = label.lower()
    return any(token in normalized for token in TABLETOP_OBJECT_LABELS)


def best_table_support(bbox_xyxy: Any, table_candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    bbox = bbox_array(bbox_xyxy)
    if bbox is None:
        return None
    best: dict[str, Any] | None = None
    best_score = 0.0
    for table in table_candidates:
        table_bbox = bbox_array(table.get("bbox_xyxy"))
        if table_bbox is None:
            continue
        score = bbox_overlap_area(bbox, table_bbox)
        if score <= 0.0:
            # A tabletop object mask can sit slightly above the visible support,
            # so allow a center projection match when horizontal overlap is clear.
            center_x = float((bbox[0] + bbox[2]) / 2.0)
            horizontal_inside = float(table_bbox[0]) <= center_x <= float(table_bbox[2])
            vertically_near = float(bbox[3]) >= float(table_bbox[1]) - max(24.0, bbox_height(bbox) * 0.35)
            score = 1.0 if horizontal_inside and vertically_near else 0.0
        if score > best_score:
            best = table
            best_score = score
    return best


def bbox_array(value: Any) -> np.ndarray | None:
    try:
        bbox = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if bbox.shape != (4,) or not np.isfinite(bbox).all():
        return None
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    return bbox


def bounds_array(value: Any) -> np.ndarray | None:
    try:
        bounds = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        return None
    if np.any(bounds[1] <= bounds[0]):
        return None
    return bounds


def bbox_overlap_area(a: np.ndarray, b: np.ndarray) -> float:
    width = max(0.0, float(min(a[2], b[2]) - max(a[0], b[0])))
    height = max(0.0, float(min(a[3], b[3]) - max(a[1], b[1])))
    return width * height


def bbox_height(bbox: np.ndarray) -> float:
    return float(bbox[3] - bbox[1])


def infer_background_vggt_dir(background_path: Path) -> Path | None:
    parent = background_path.parent
    if (parent / "vggt_points.npy").is_file():
        return parent
    return None


def infer_source_image_path(geometry: dict[str, Any]) -> Path | None:
    for key in ("image_path", "source_image_path"):
        value = geometry.get(key)
        if value:
            return Path(str(value))
    detections_path = geometry.get("detections_path")
    if detections_path:
        path = Path(str(detections_path))
        if path.is_file():
            try:
                detections = load_json(path)
            except (OSError, json.JSONDecodeError):
                return None
            image_path = detections.get("image_path")
            if image_path:
                return Path(str(image_path))
    return None


def room_corner_plane_texture_path(background_vggt_dir: Path | None) -> Path | None:
    if background_vggt_dir is None:
        return None
    candidate = background_vggt_dir / "empty_room.png"
    return candidate if candidate.is_file() else None


def add_vggt_fitted_room_background(
    scene: Any,
    *,
    vggt_dir: Path,
    placement_bounds: np.ndarray | None,
    margin: float,
    depth_offset: float,
    coordinate_contract: dict[str, Any] | None,
) -> dict[str, Any]:
    if placement_bounds is None:
        return add_vggt_background_mesh(
            scene,
            vggt_dir=vggt_dir,
            objects_dir=Path(),
            object_dirs={},
            stride=16,
            clip_masks=False,
            clip_dilation_px=0,
            placement_bounds=None,
            margin=margin,
            depth_offset=depth_offset,
            coordinate_contract=coordinate_contract,
        )
    plane_path = vggt_dir / "plane_detections.json"
    planes = load_json(plane_path).get("planes", []) if plane_path.is_file() else []
    plane_overrides = vggt_plane_room_overrides(planes)
    stats = add_room_corner_background(
        scene,
        placement_bounds=placement_bounds,
        margin=margin,
        depth_offset=depth_offset,
        texture_image_path=room_corner_plane_texture_path(vggt_dir),
        coordinate_contract=coordinate_contract,
        floor_y_override=plane_overrides.get("floor_y"),
        z_back_override=plane_overrides.get("z_back"),
        side_x_override=plane_overrides.get("side_x"),
    )
    stats["source"] = "vggt_textured_fitted_room_planes_expanded"
    stats["plane_detections_path"] = str(plane_path) if plane_path.is_file() else None
    stats["vggt_plane_overrides"] = plane_overrides
    return stats


def vggt_plane_room_overrides(planes: list[dict[str, Any]]) -> dict[str, float]:
    overrides: dict[str, float] = {}
    floor = first_room_plane(planes, "floor")
    back_wall = first_room_plane(planes, "back_wall")
    right_wall = first_room_plane(planes, "right_wall")
    if floor is not None:
        vertices = plane_vertices_gltf(floor)
        if len(vertices) > 0:
            overrides["floor_y"] = float(np.median(vertices[:, 1]))
    if back_wall is not None:
        vertices = plane_vertices_gltf(back_wall)
        if len(vertices) > 0:
            overrides["z_back"] = float(np.median(vertices[:, 2]))
    if right_wall is not None:
        vertices = plane_vertices_gltf(right_wall)
        if len(vertices) > 0:
            overrides["side_x"] = float(np.median(vertices[:, 0]))
    return overrides


def first_room_plane(planes: list[dict[str, Any]], plane_id: str) -> dict[str, Any] | None:
    for plane in planes:
        if str(plane.get("id") or "") == plane_id:
            return plane
    return None


def plane_vertices_gltf(plane: dict[str, Any]) -> np.ndarray:
    converted: list[tuple[float, float, float]] = []
    for vertex in plane.get("vertices_xyz") or []:
        if isinstance(vertex, list) and len(vertex) == 3:
            converted.append(scene_point_to_gltf_vertex(vertex))
    return np.asarray(converted, dtype=np.float64)


def add_vggt_background_mesh(
    scene: Any,
    *,
    vggt_dir: Path,
    objects_dir: Path,
    object_dirs: dict[int, Path],
    stride: int,
    clip_masks: bool,
    clip_dilation_px: int,
    placement_bounds: np.ndarray | None = None,
    margin: float = 1.0,
    depth_offset: float = 0.12,
    coordinate_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        import trimesh
        from trimesh.visual.material import PBRMaterial
        from trimesh.visual.texture import TextureVisuals
    except Exception as exc:
        raise RuntimeError("Scene composition requires trimesh from requirements.txt.") from exc

    points_path = vggt_dir / "vggt_points.npy"
    image_path = vggt_dir / "empty_room.png"
    if not points_path.is_file():
        raise FileNotFoundError(f"Background VGGT point map does not exist: {points_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Background empty-room image does not exist: {image_path}")

    points = np.load(points_path).astype(np.float32)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"Expected background VGGT points with shape HxWx3, got {points.shape}")
    height, width = points.shape[:2]
    image = Image.open(image_path).convert("RGB")
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    rgb = np.asarray(image, dtype=np.uint8)
    mask = combined_object_mask(object_dirs, image_size=(width, height), dilation_px=clip_dilation_px) if clip_masks else np.zeros((height, width), dtype=bool)

    stride = max(1, int(stride))
    row_indices = list(range(0, height, stride))
    col_indices = list(range(0, width, stride))
    vertex_indices = np.full((len(row_indices), len(col_indices)), -1, dtype=np.int32)
    vertices: list[tuple[float, float, float]] = []
    colors: list[tuple[int, int, int, int]] = []
    uvs: list[tuple[float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for row_out, y in enumerate(row_indices):
        for col_out, x in enumerate(col_indices):
            point = points[y, x]
            if mask[y, x] or not np.isfinite(point).all():
                continue
            vertex_indices[row_out, col_out] = len(vertices)
            vertices.append(scene_point_to_gltf_vertex(point))
            color = rgb[y, x]
            colors.append((int(color[0]), int(color[1]), int(color[2]), 255))
            u = float(x / max(width - 1, 1))
            v = float(1.0 - y / max(height - 1, 1))
            uvs.append((u, v))

    for row in range(len(row_indices) - 1):
        for col in range(len(col_indices) - 1):
            v00 = int(vertex_indices[row, col])
            v10 = int(vertex_indices[row, col + 1])
            v01 = int(vertex_indices[row + 1, col])
            v11 = int(vertex_indices[row + 1, col + 1])
            if min(v00, v10, v01, v11) < 0:
                continue
            faces.append((v00, v11, v10))
            faces.append((v00, v01, v11))

    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int64),
        vertex_colors=np.asarray(colors, dtype=np.uint8),
        process=False,
    )
    mesh.visual = TextureVisuals(
        uv=np.asarray(uvs, dtype=np.float32),
        material=PBRMaterial(
            name="empty_room_vggt_projected_texture",
            baseColorTexture=image.copy(),
            baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            emissiveTexture=image.copy(),
            emissiveFactor=[0.12, 0.12, 0.12],
            roughnessFactor=0.9,
            metallicFactor=0.0,
            doubleSided=True,
        ),
    )
    raw_source_bounds = np.asarray(mesh.bounds, dtype=np.float64)
    plane_path = vggt_dir / "plane_detections.json"
    orientation_transform, orientation_report = vggt_room_orientation_transform(plane_path)
    mesh.apply_transform(orientation_transform)
    source_bounds = np.asarray(mesh.bounds, dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    alignment = "raw_vggt_camera_space"
    room_alignment = unavailable_room_alignment("missing_placement_bounds")
    if placement_bounds is not None and len(vertices) > 0:
        fit_transform, room_alignment = vggt_room_alignment_transform(
            source_bounds=source_bounds,
            placement_bounds=placement_bounds,
            plane_path=plane_path,
            orientation_transform=orientation_transform,
            coordinate_contract=coordinate_contract,
            margin=margin,
            depth_offset=depth_offset,
        )
        transform = fit_transform @ orientation_transform
        room_alignment["applied_transform_gltf"] = transform.tolist()
        mesh.apply_transform(fit_transform)
        floor_regularization = regularize_vggt_floor_vertices(mesh)
        alignment = "plane_camera_guided_room_alignment"
    else:
        floor_regularization = {"status": "skipped", "reason": "missing_placement_bounds"}
    scene.add_geometry(mesh, geom_name="background_camera_clipped_000", node_name="background_camera_clipped_000")
    transformed_bounds = np.asarray(mesh.bounds, dtype=np.float64)
    return {
        "path": str(points_path),
        "image_path": str(image_path),
        "mesh_count": 1,
        "source": "vggt_points_camera_clipped",
        "alignment": alignment,
        "orientation": orientation_report,
        "room_alignment": room_alignment,
        "floor_regularization": floor_regularization,
        "stride": stride,
        "clip_masks": bool(clip_masks),
        "clip_dilation_px": int(clip_dilation_px),
        "masked_pixel_ratio": float(mask.mean()) if mask.size else 0.0,
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
        "texture_source": "empty_room_image_uv_projected",
        "texture_image_path": str(image_path),
        "uv_count": int(len(uvs)),
        "vertex_colors": "sampled_empty_room_image_fallback",
        "raw_source_bounds": raw_source_bounds.tolist(),
        "source_bounds": source_bounds.tolist(),
        "transform_gltf": transform.tolist(),
        "transformed_bounds": transformed_bounds.tolist(),
    }


def regularize_vggt_floor_vertices(mesh: Any) -> dict[str, Any]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        return {"status": "skipped", "reason": "empty_mesh"}
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    height = float(bounds[1, 1] - bounds[0, 1])
    if height <= 1e-8:
        return {"status": "skipped", "reason": "degenerate_height"}
    floor_y = float(bounds[0, 1])
    band = max(0.18, height * 0.30)
    floor_mask = vertices[:, 1] <= floor_y + band
    affected = int(floor_mask.sum())
    if affected == 0:
        return {"status": "skipped", "reason": "no_floor_band_vertices", "floor_y": floor_y, "band": band}
    vertices[floor_mask, 1] = floor_y
    mesh.vertices = vertices
    return {
        "status": "applied",
        "method": "lower_vggt_band_to_fitted_floor_y",
        "floor_y": floor_y,
        "band": float(band),
        "affected_vertex_count": affected,
        "affected_vertex_ratio": float(affected / len(vertices)),
    }


def vggt_room_orientation_transform(plane_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    if not plane_path.is_file():
        return np.eye(4, dtype=np.float64), {"method": "identity", "reason": "missing_plane_detections"}
    try:
        planes = load_json(plane_path).get("planes", [])
    except (OSError, json.JSONDecodeError):
        return np.eye(4, dtype=np.float64), {"method": "identity", "reason": "invalid_plane_detections"}
    floor = first_room_plane(planes, "floor")
    back_wall = first_room_plane(planes, "back_wall")
    if floor is None or back_wall is None:
        return np.eye(4, dtype=np.float64), {"method": "identity", "reason": "missing_floor_or_back_wall_plane"}
    source_up = scene_normal_to_gltf(floor.get("fitted_normal_xyz"))
    source_back = scene_normal_to_gltf(back_wall.get("fitted_normal_xyz"))
    target_up = scene_normal_to_gltf(floor.get("normal_xyz"))
    target_back = scene_normal_to_gltf(back_wall.get("normal_xyz"))
    if source_up is None or source_back is None or target_up is None or target_back is None:
        return np.eye(4, dtype=np.float64), {"method": "identity", "reason": "missing_plane_normals"}
    if float(np.dot(source_up, target_up)) < 0.0:
        source_up = -source_up
    if float(np.dot(source_back, target_back)) < 0.0:
        source_back = -source_back
    source_basis = room_basis_from_up_and_back(source_up, source_back)
    target_basis = room_basis_from_up_and_back(target_up, target_back)
    if source_basis is None or target_basis is None:
        return np.eye(4, dtype=np.float64), {"method": "identity", "reason": "degenerate_plane_basis"}
    rotation = target_basis @ source_basis.T
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    return transform, {
        "method": "fitted_floor_back_wall_normals_to_regularized_axes",
        "plane_detections_path": str(plane_path),
        "source_up_gltf": source_up.tolist(),
        "source_back_gltf": source_back.tolist(),
        "target_up_gltf": target_up.tolist(),
        "target_back_gltf": target_back.tolist(),
        "rotation_gltf": rotation.tolist(),
    }


def scene_normal_to_gltf(value: Any) -> np.ndarray | None:
    normal = np.asarray(value, dtype=np.float64)
    if normal.shape != (3,) or not np.isfinite(normal).all():
        return None
    mapped = np.asarray([normal[0], normal[2], -normal[1]], dtype=np.float64)
    norm = float(np.linalg.norm(mapped))
    if norm <= 1e-8:
        return None
    return mapped / norm


def room_basis_from_up_and_back(up: np.ndarray, back: np.ndarray) -> np.ndarray | None:
    up = np.asarray(up, dtype=np.float64)
    back = np.asarray(back, dtype=np.float64)
    up_norm = float(np.linalg.norm(up))
    if up_norm <= 1e-8:
        return None
    up = up / up_norm
    back = back - up * float(np.dot(back, up))
    back_norm = float(np.linalg.norm(back))
    if back_norm <= 1e-8:
        return None
    back = back / back_norm
    right = np.cross(up, back)
    right_norm = float(np.linalg.norm(right))
    if right_norm <= 1e-8:
        return None
    right = right / right_norm
    back = np.cross(right, up)
    return np.stack([right, up, back], axis=1)


def combined_object_mask(object_dirs: dict[int, Path], *, image_size: tuple[int, int], dilation_px: int) -> np.ndarray:
    width, height = image_size
    combined = Image.new("L", (width, height), 0)
    for object_dir in object_dirs.values():
        mask_path = first_existing_mask_path(object_dir)
        if mask_path is None:
            continue
        mask = Image.open(mask_path).convert("L")
        if mask.size != (width, height):
            mask = mask.resize((width, height), Image.Resampling.NEAREST)
        combined = Image.fromarray(np.maximum(np.asarray(combined, dtype=np.uint8), np.asarray(mask, dtype=np.uint8)), mode="L")
    if dilation_px > 0:
        size = max(3, int(dilation_px) * 2 + 1)
        if size % 2 == 0:
            size += 1
        combined = combined.filter(ImageFilter.MaxFilter(size))
    return np.asarray(combined, dtype=np.uint8) > 0


def first_existing_mask_path(object_dir: Path) -> Path | None:
    for mask_path in (object_dir / "full_mask.png", object_dir / "artifacts" / "segmentation" / "full_mask.png"):
        if mask_path.is_file():
            return mask_path
    return None


def background_floor_y(background_stats: dict[str, Any]) -> float | None:
    bounds = background_stats.get("transformed_bounds")
    if not bounds:
        return None
    array = np.asarray(bounds, dtype=np.float64)
    if array.shape != (2, 3) or not np.isfinite(array).all():
        return None
    return float(array[0, 1])


def snap_transform_to_floor(mesh_path: Path, transform: np.ndarray, floor_y: float) -> tuple[np.ndarray, float]:
    return snap_transform_to_support(mesh_path, transform, floor_y)


def snap_transform_to_support(
    mesh_path: Path,
    transform: np.ndarray,
    support_y: float,
    *,
    support_kind: Any = None,
) -> tuple[np.ndarray, float]:
    meshes = load_meshes(mesh_path)
    source_bounds = combined_bounds(meshes)
    contact_y = transformed_mesh_contact_y(meshes, source_bounds, transform, support_kind=support_kind)
    delta = float(support_y - contact_y)
    snapped = np.asarray(transform, dtype=np.float64).copy()
    snapped[1, 3] += delta
    return snapped, delta


def snap_transform_to_support_bounds(source_bounds: np.ndarray, transform: np.ndarray, support_y: float) -> tuple[np.ndarray, float]:
    transformed = transformed_bounds_from_source_bounds(source_bounds, transform)
    delta = float(support_y - transformed[0, 1])
    snapped = np.asarray(transform, dtype=np.float64).copy()
    snapped[1, 3] += delta
    return snapped, delta


def room_boundary_adjustment_report(mesh_path: Path, transform: np.ndarray, room_bounds: np.ndarray | None) -> dict[str, Any]:
    if room_bounds is None:
        return {
            "status": "unavailable",
            "reason": "missing_room_bounds",
            "translation_delta": [0.0, 0.0, 0.0],
            "transform_gltf": np.asarray(transform, dtype=np.float64).tolist(),
        }
    room_bounds = np.asarray(room_bounds, dtype=np.float64)
    if room_bounds.shape != (2, 3) or not np.isfinite(room_bounds).all():
        return {
            "status": "unavailable",
            "reason": "invalid_room_bounds",
            "translation_delta": [0.0, 0.0, 0.0],
            "transform_gltf": np.asarray(transform, dtype=np.float64).tolist(),
        }
    original = np.asarray(transform, dtype=np.float64).copy()
    before = transformed_mesh_bounds(mesh_path, original)
    delta = np.zeros(3, dtype=np.float64)
    room_extent = room_bounds[1] - room_bounds[0]
    padding = np.array(
        [
            max(float(room_extent[0]) * 0.04, 0.03),
            0.0,
            max(float(room_extent[2]) * 0.04, 0.03),
        ],
        dtype=np.float64,
    )
    inner_min = room_bounds[0] + padding
    inner_max = room_bounds[1] - padding
    for axis in (0, 2):
        object_extent = float(before[1, axis] - before[0, axis])
        inner_extent = float(inner_max[axis] - inner_min[axis])
        if object_extent > inner_extent:
            delta[axis] = float((inner_min[axis] + inner_max[axis]) / 2.0 - (before[0, axis] + before[1, axis]) / 2.0)
        elif before[0, axis] < inner_min[axis]:
            delta[axis] = float(inner_min[axis] - before[0, axis])
        elif before[1, axis] > inner_max[axis]:
            delta[axis] = float(inner_max[axis] - before[1, axis])
    adjusted = original.copy()
    adjusted[:3, 3] += delta
    after = transformed_mesh_bounds(mesh_path, adjusted)
    changed = bool(np.any(np.abs(delta) > 1e-8))
    return {
        "status": "would_adjust" if changed else "inside_room_bounds",
        "method": "x_z_room_bounds_clamp",
        "translation_delta": [float(value) for value in delta],
        "room_bounds": room_bounds.tolist(),
        "inner_room_bounds": np.stack([inner_min, inner_max], axis=0).tolist(),
        "bounds_before": before.tolist(),
        "bounds_after": after.tolist(),
        "applied": False,
        "transform_gltf": original.tolist(),
    }


def transformed_mesh_contact_y(
    meshes: list[Any],
    source_bounds: np.ndarray,
    transform: np.ndarray,
    *,
    support_kind: Any = None,
) -> float:
    points = transformed_mesh_vertices(meshes, source_bounds, transform)
    if len(points) == 0:
        return float(transformed_bounds_from_source_bounds(source_bounds, transform)[0, 1])
    return float(support_contact_estimate(points, support_kind=support_kind)["contact_y"])


def mesh_support_contact_report(mesh_path: Path, transform: np.ndarray, support_kind: Any) -> dict[str, Any] | None:
    try:
        meshes = load_meshes(mesh_path)
        source_bounds = combined_bounds(meshes)
        points = transformed_mesh_vertices(meshes, source_bounds, transform)
    except Exception:
        return None
    if len(points) == 0:
        return None
    estimate = support_contact_estimate(points, support_kind=support_kind)
    layer = estimate.get("selected_layer") or {}
    return {
        "method": "stable_bottom_footprint_v2",
        "support_kind": normalized_support_kind(support_kind),
        "selection_method": estimate.get("selection_method"),
        "raw_bottom_y": float(points[:, 1].min()),
        "contact_y": float(estimate["contact_y"]),
        "selected_quantile": estimate["selected_quantile"],
        "selection_reason": estimate["selection_reason"],
        "vertex_ratio": layer.get("vertex_ratio"),
        "footprint_span": layer.get("footprint_span"),
        "area_ratio": layer.get("area_ratio"),
        "selected_layer": estimate["selected_layer"],
    }


def transformed_mesh_vertices(meshes: list[Any], source_bounds: np.ndarray, transform: np.ndarray) -> np.ndarray:
    asset_transform = np.asarray(transform, dtype=np.float64) @ normalization_transform(source_bounds)
    vertices: list[np.ndarray] = []
    for mesh in meshes:
        mesh_vertices = np.asarray(mesh.vertices, dtype=np.float64)
        if mesh_vertices.ndim == 2 and mesh_vertices.shape[1] == 3 and len(mesh_vertices) > 0:
            transformed = transform_points(mesh_vertices, asset_transform)
            transformed = transformed[np.isfinite(transformed).all(axis=1)]
            if len(transformed) > 0:
                vertices.append(transformed)
    if not vertices:
        return np.empty((0, 3), dtype=np.float64)
    return np.concatenate(vertices, axis=0)


def support_contact_candidates(points: np.ndarray, *, support_kind: Any = None) -> list[dict[str, Any]]:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        return []
    support_kind = normalized_support_kind(support_kind)
    quantiles = SUPPORT_CONTACT_TABLETOP_QUANTILES if support_kind == "tabletop" else SUPPORT_CONTACT_FLOOR_QUANTILES
    x_span = max(float(points[:, 0].max() - points[:, 0].min()), 1e-8)
    z_span = max(float(points[:, 2].max() - points[:, 2].min()), 1e-8)
    y_span = max(float(points[:, 1].max() - points[:, 1].min()), 1e-8)
    layer_tolerance = max(y_span * 0.025, 1e-5)
    candidates: list[dict[str, Any]] = []
    for quantile in quantiles:
        contact_y = float(np.percentile(points[:, 1], quantile))
        layer = points[points[:, 1] <= contact_y + layer_tolerance]
        accepted, report = support_contact_layer_quality(
            layer,
            total_count=len(points),
            x_span=x_span,
            z_span=z_span,
            support_kind=support_kind,
        )
        candidates.append(
            {
                "contact_y": contact_y,
                "quantile": float(quantile),
                "accepted": accepted,
                **report,
            }
        )
    return candidates


def support_contact_estimate(points: np.ndarray, *, support_kind: Any = None) -> dict[str, Any]:
    y_values = np.asarray(points, dtype=np.float64)[:, 1]
    candidates = support_contact_candidates(points, support_kind=support_kind)
    for candidate in candidates:
        if candidate["accepted"]:
            return {
                "contact_y": float(candidate["contact_y"]),
                "selected_quantile": float(candidate["quantile"]),
                "selection_method": "stable_footprint",
                "selection_reason": "stable_contact_layer",
                "selected_layer": contact_layer_report(candidate),
            }
    raw_y = float(y_values.min())
    raw_layer = raw_bottom_layer_report(points, candidates)
    return {
        "contact_y": raw_y,
        "selected_quantile": 0.0,
        "selection_method": "raw_bottom",
        "selection_reason": "stable_contact_unavailable_raw_bottom",
        "selected_layer": raw_layer,
    }


def contact_layer_report(candidate: dict[str, Any]) -> dict[str, Any]:
    x_span_ratio = float(candidate.get("x_span_ratio", 0.0))
    z_span_ratio = float(candidate.get("z_span_ratio", 0.0))
    return {
        "quantile": float(candidate["quantile"]),
        "accepted": bool(candidate["accepted"]),
        "reason": candidate.get("reason"),
        "vertex_ratio": float(candidate.get("vertex_ratio", 0.0)),
        "x_span_ratio": x_span_ratio,
        "z_span_ratio": z_span_ratio,
        "footprint_span": {"x_ratio": x_span_ratio, "z_ratio": z_span_ratio},
        "area_ratio": float(candidate.get("area_ratio", 0.0)),
    }


def raw_bottom_layer_report(points: np.ndarray, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if candidates:
        first = min(candidates, key=lambda item: float(item.get("quantile", 0.0)))
        report = contact_layer_report(first)
        report["accepted"] = False
        report["reason"] = "raw_bottom_fallback"
        return report
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        return None
    raw_y = float(points[:, 1].min())
    layer = points[np.isclose(points[:, 1], raw_y, rtol=0.0, atol=max(float(np.ptp(points[:, 1])) * 0.005, 1e-5))]
    if len(layer) == 0:
        return None
    return {
        "quantile": 0.0,
        "accepted": False,
        "reason": "raw_bottom_fallback",
        "vertex_ratio": float(len(layer) / len(points)),
        "x_span_ratio": 0.0,
        "z_span_ratio": 0.0,
        "footprint_span": {"x_ratio": 0.0, "z_ratio": 0.0},
        "area_ratio": 0.0,
    }


def support_contact_layer_quality(
    layer: np.ndarray,
    *,
    total_count: int,
    x_span: float,
    z_span: float,
    support_kind: str | None,
) -> tuple[bool, dict[str, Any]]:
    if len(layer) == 0 or total_count <= 0:
        return False, {
            "reason": "empty_contact_layer",
            "vertex_ratio": 0.0,
            "x_span_ratio": 0.0,
            "z_span_ratio": 0.0,
            "area_ratio": 0.0,
        }
    layer_x = float(layer[:, 0].max() - layer[:, 0].min())
    layer_z = float(layer[:, 2].max() - layer[:, 2].min())
    x_ratio = float(np.clip(layer_x / x_span, 0.0, 1.0))
    z_ratio = float(np.clip(layer_z / z_span, 0.0, 1.0))
    area_ratio = x_ratio * z_ratio
    vertex_ratio = float(len(layer) / total_count)
    is_tabletop = support_kind == "tabletop"
    min_vertex_ratio = TABLETOP_CONTACT_MIN_VERTEX_RATIO if is_tabletop else SUPPORT_CONTACT_MIN_VERTEX_RATIO
    min_area_ratio = SUPPORT_CONTACT_MIN_AREA_RATIO if is_tabletop else 0.025
    min_span_ratio = SUPPORT_CONTACT_MIN_SPAN_RATIO if is_tabletop else 0.08
    max_span_ratio = 0.45 if is_tabletop else 0.35
    excellent_footprint = area_ratio >= SUPPORT_CONTACT_EXCELLENT_AREA_RATIO and min(x_ratio, z_ratio) >= 0.45
    stable_footprint = (
        area_ratio >= min_area_ratio
        and max(x_ratio, z_ratio) >= max_span_ratio
        and min(x_ratio, z_ratio) >= min_span_ratio
    )
    accepted = excellent_footprint or (stable_footprint and vertex_ratio >= min_vertex_ratio)
    reason = "accepted" if accepted else "unstable_or_tiny_contact_layer"
    return accepted, {
        "reason": reason,
        "vertex_ratio": vertex_ratio,
        "x_span_ratio": x_ratio,
        "z_span_ratio": z_ratio,
        "area_ratio": area_ratio,
    }


def transformed_mesh_bounds(mesh_path: Path, transform: np.ndarray) -> np.ndarray:
    source_bounds = combined_bounds(load_meshes(mesh_path))
    return transformed_bounds_from_source_bounds(source_bounds, transform)


def transformed_bounds_from_source_bounds(source_bounds: np.ndarray, transform: np.ndarray) -> np.ndarray:
    asset_transform = transform @ normalization_transform(source_bounds)
    transformed = transform_points(bounds_corners(source_bounds), asset_transform)
    return np.stack([transformed.min(axis=0), transformed.max(axis=0)], axis=0)


def bounds_corners(bounds: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [x, y, z]
            for x in (bounds[0, 0], bounds[1, 0])
            for y in (bounds[0, 1], bounds[1, 1])
            for z in (bounds[0, 2], bounds[1, 2])
        ],
        dtype=np.float64,
    )


def add_room_corner_background(
    scene: Any,
    *,
    placement_bounds: np.ndarray,
    margin: float,
    depth_offset: float,
    texture_image_path: Path | None = None,
    coordinate_contract: dict[str, Any] | None = None,
    texture_grid_steps: int = 36,
    floor_y_override: float | None = None,
    z_back_override: float | None = None,
    side_x_override: float | None = None,
) -> dict[str, Any]:
    try:
        import trimesh
        from trimesh.visual.material import PBRMaterial
        from trimesh.visual.texture import TextureVisuals
    except Exception as exc:
        raise RuntimeError("Scene composition requires trimesh from requirements.txt.") from exc

    extent = placement_bounds[1] - placement_bounds[0]
    x_pad = max(float(extent[0]) * (margin - 1.0), 0.08)
    z_pad = max(float(extent[2]) * (margin - 1.0), 0.08)
    y_pad = max(float(extent[1]) * 0.10, 0.05)
    x_min = float(placement_bounds[0, 0] - x_pad)
    x_max = float(placement_bounds[1, 0] + x_pad)
    floor_y = float(floor_y_override if floor_y_override is not None else placement_bounds[0, 1] - y_pad)
    z_back = float(z_back_override if z_back_override is not None else placement_bounds[0, 2] - max(depth_offset, z_pad))
    z_front = float(placement_bounds[1, 2] + z_pad)
    camera_frustum_wall_top = max(0.0, -z_back) * 0.56
    wall_top_y = float(
        max(
            placement_bounds[1, 1] + max(float(extent[1]) * 1.60, 0.65),
            camera_frustum_wall_top,
        )
    )
    side_x = float(side_x_override if side_x_override is not None else x_max)

    texture_image = Image.open(texture_image_path).convert("RGB") if texture_image_path is not None else None
    texture_contract = room_corner_texture_contract(texture_image, coordinate_contract)
    texture_rgb = np.asarray(texture_image, dtype=np.uint8) if texture_image is not None else None

    specs = [
        (
            "background_floor_000",
            np.array(
                [
                    [x_min, floor_y, z_front],
                    [x_max, floor_y, z_front],
                    [x_max, floor_y, z_back],
                    [x_min, floor_y, z_back],
                ],
                dtype=np.float32,
            ),
            [218, 216, 208, 255],
        ),
        (
            "background_back_wall_000",
            np.array(
                [
                    [x_min, floor_y, z_back],
                    [x_max, floor_y, z_back],
                    [x_max, wall_top_y, z_back],
                    [x_min, wall_top_y, z_back],
                ],
                dtype=np.float32,
            ),
            [238, 236, 230, 255],
        ),
        (
            "background_side_wall_000",
            np.array(
                [
                    [side_x, floor_y, z_front],
                    [side_x, wall_top_y, z_front],
                    [side_x, wall_top_y, z_back],
                    [side_x, floor_y, z_back],
                ],
                dtype=np.float32,
            ),
            [232, 230, 224, 255],
        ),
    ]
    bounds: list[np.ndarray] = []
    total_vertices = 0
    total_faces = 0
    for name, vertices, color in specs:
        vertices_out, faces, colors, uvs = room_corner_plane_geometry(
            vertices=vertices,
            fallback_color=color,
            texture_rgb=texture_rgb,
            coordinate_contract=texture_contract,
            grid_steps=texture_grid_steps if texture_image is not None else 1,
        )
        mesh = trimesh.Trimesh(vertices=vertices_out, faces=faces, vertex_colors=colors, process=False)
        rgb = [float(value) / 255.0 for value in color[:3]]
        if texture_image is not None:
            mesh.visual = TextureVisuals(
                uv=uvs,
                material=PBRMaterial(
                    name=f"{name}_projected_empty_room_mat",
                    baseColorTexture=texture_image.copy(),
                    baseColorFactor=[1.0, 1.0, 1.0, 1.0],
                    emissiveTexture=texture_image.copy(),
                    emissiveFactor=[0.25, 0.25, 0.25],
                    roughnessFactor=0.9,
                    metallicFactor=0.0,
                    doubleSided=True,
                ),
            )
        else:
            mesh.visual = TextureVisuals(
                material=PBRMaterial(
                    name=f"{name}_mat",
                    baseColorFactor=[rgb[0], rgb[1], rgb[2], 1.0],
                    emissiveFactor=[rgb[0] * 0.35, rgb[1] * 0.35, rgb[2] * 0.35],
                    roughnessFactor=0.9,
                    metallicFactor=0.0,
                    doubleSided=True,
                )
            )
        scene.add_geometry(mesh, geom_name=name, node_name=name)
        bounds.append(np.asarray(mesh.bounds, dtype=np.float64))
        total_vertices += int(len(vertices_out))
        total_faces += int(len(faces))
    merged = merge_bounds(bounds)
    return {
        "path": None,
        "mesh_count": len(specs),
        "source": "procedural_room_corner_from_placement_bounds",
        "room_alignment": {
            "method": "procedural_placement_bounds_room_corner",
            "floor_plane_id": None,
            "wall_plane_ids": [],
            "usable_floor_bounds": [[x_min, z_back], [x_max, z_front]],
            "applied_transform_gltf": np.eye(4, dtype=np.float64).tolist(),
        },
        "placement_bounds": placement_bounds.tolist(),
        "source_bounds": merged.tolist(),
        "transform_gltf": np.eye(4, dtype=np.float64).tolist(),
        "transformed_bounds": merged.tolist(),
        "vertex_count": total_vertices,
        "face_count": total_faces,
        "texture_source": "empty_room_image_camera_projected" if texture_image_path is not None else None,
        "texture_image_path": str(texture_image_path) if texture_image_path is not None else None,
        "texture_grid_steps": int(texture_grid_steps) if texture_image_path is not None else None,
        "vertex_colors": "projected_empty_room_image_fallback" if texture_image_path is not None else "plane_fallback_color",
        "floor_y": floor_y,
        "wall_top_y": wall_top_y,
        "z_back": z_back,
        "z_front": z_front,
    }


def room_corner_texture_contract(image: Image.Image | None, coordinate_contract: dict[str, Any] | None) -> dict[str, Any] | None:
    if image is None:
        return None
    contract = dict(coordinate_contract or {})
    contract["image_width"] = int(image.size[0])
    contract["image_height"] = int(image.size[1])
    contract.setdefault("fov_degrees", DEFAULT_FOV_DEGREES)
    return contract


def room_corner_plane_geometry(
    *,
    vertices: np.ndarray,
    fallback_color: list[int],
    texture_rgb: np.ndarray | None,
    coordinate_contract: dict[str, Any] | None,
    grid_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    corners = np.asarray(vertices, dtype=np.float64)
    steps = max(1, int(grid_steps))
    out_vertices: list[np.ndarray] = []
    colors: list[tuple[int, int, int, int]] = []
    uvs: list[tuple[float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for row in range(steps + 1):
        v = row / steps
        left = corners[0] * (1.0 - v) + corners[3] * v
        right = corners[1] * (1.0 - v) + corners[2] * v
        for col in range(steps + 1):
            u = col / steps
            point = left * (1.0 - u) + right * u
            out_vertices.append(point)
            colors.append(room_corner_vertex_color(point, fallback_color, texture_rgb, coordinate_contract))
            uvs.append(room_corner_vertex_uv(point, texture_rgb, coordinate_contract))

    row_width = steps + 1
    for row in range(steps):
        for col in range(steps):
            v00 = row * row_width + col
            v10 = v00 + 1
            v01 = (row + 1) * row_width + col
            v11 = v01 + 1
            faces.append((v00, v10, v11))
            faces.append((v00, v11, v01))
    return (
        np.asarray(out_vertices, dtype=np.float32),
        np.asarray(faces, dtype=np.int64),
        np.asarray(colors, dtype=np.uint8),
        np.asarray(uvs, dtype=np.float32),
    )


def room_corner_vertex_color(
    point: np.ndarray,
    fallback_color: list[int],
    texture_rgb: np.ndarray | None,
    coordinate_contract: dict[str, Any] | None,
) -> tuple[int, int, int, int]:
    color = np.asarray(fallback_color, dtype=np.float64)
    pixel = room_corner_projected_pixel(point, texture_rgb, coordinate_contract)
    if pixel is not None and texture_rgb is not None:
        x, y = pixel
        color[:3] = np.asarray(texture_rgb[y, x], dtype=np.float64)
    return tuple(int(np.clip(value, 0, 255)) for value in color)


def room_corner_vertex_uv(
    point: np.ndarray,
    texture_rgb: np.ndarray | None,
    coordinate_contract: dict[str, Any] | None,
) -> tuple[float, float]:
    pixel = room_corner_projected_pixel(point, texture_rgb, coordinate_contract)
    if pixel is None or texture_rgb is None:
        return (0.0, 0.0)
    x, y = pixel
    u = x / max(texture_rgb.shape[1] - 1, 1)
    v = 1.0 - y / max(texture_rgb.shape[0] - 1, 1)
    return (float(np.clip(u, 0.0, 1.0)), float(np.clip(v, 0.0, 1.0)))


def room_corner_projected_pixel(
    point: np.ndarray,
    texture_rgb: np.ndarray | None,
    coordinate_contract: dict[str, Any] | None,
) -> tuple[int, int] | None:
    if texture_rgb is None:
        return None
    projected = project_gltf_point_to_pixel(point, coordinate_contract)
    if projected is None:
        return None
    x, y = projected
    if not np.isfinite([x, y]).all():
        return None
    return (
        int(np.clip(round(x), 0, texture_rgb.shape[1] - 1)),
        int(np.clip(round(y), 0, texture_rgb.shape[0] - 1)),
    )


def placement_transform_to_gltf(
    placement: dict[str, Any],
    *,
    placement_orientation: str = "upright",
    object_scale_factor: float = 1.0,
) -> np.ndarray:
    if placement_orientation == "obb":
        return obb_placement_transform_to_gltf(placement, object_scale_factor=object_scale_factor)
    if placement_orientation == "upright":
        return upright_placement_transform_to_gltf(placement, object_scale_factor=object_scale_factor)
    raise ValueError(f"Unsupported placement orientation: {placement_orientation}")


def upright_placement_transform_to_gltf(placement: dict[str, Any], *, object_scale_factor: float) -> np.ndarray:
    center, scene_extent = placement_scene_center_and_extent(placement)
    center_gltf = np.asarray(scene_point_to_gltf_vertex(center), dtype=np.float64)
    target_extent = np.asarray(
        [
            scene_extent[0],
            scene_extent[2],
            scene_extent[1],
        ],
        dtype=np.float64,
    ) * float(object_scale_factor)
    uniform_extent = float(np.median(target_extent))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.eye(3, dtype=np.float64) * uniform_extent
    transform[:3, 3] = center_gltf
    return transform


def obb_placement_transform_to_gltf(placement: dict[str, Any], *, object_scale_factor: float) -> np.ndarray:
    center = required_vector(placement.get("center_xyz"), "center_xyz")
    extent = required_vector(placement.get("extent_xyz"), "extent_xyz")
    rotation = np.asarray(placement.get("rotation_matrix"), dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("rotation_matrix must be a finite 3x3 matrix")
    if np.any(extent <= 0):
        raise ValueError("extent_xyz must be positive")

    center_gltf = np.asarray(scene_point_to_gltf_vertex(center), dtype=np.float64)
    scene_to_gltf = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )
    axes_gltf = scene_to_gltf @ rotation
    transform = np.eye(4, dtype=np.float64)
    target_extent = extent * float(object_scale_factor)
    uniform_extent = float(np.median(target_extent))
    transform[:3, :3] = axes_gltf @ (np.eye(3, dtype=np.float64) * uniform_extent)
    transform[:3, 3] = center_gltf
    return transform


def placement_scene_center_and_extent(placement: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    center = required_vector(placement.get("center_xyz"), "center_xyz")
    extent = required_vector(placement.get("extent_xyz"), "extent_xyz")
    if np.any(extent <= 0):
        raise ValueError("extent_xyz must be positive")
    rotation = np.asarray(placement.get("rotation_matrix"), dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("rotation_matrix must be a finite 3x3 matrix")
    offsets = np.array(
        [
            [sx, sy, sz]
            for sx in (-0.5, 0.5)
            for sy in (-0.5, 0.5)
            for sz in (-0.5, 0.5)
        ],
        dtype=np.float64,
    ) * extent
    corners = center + offsets @ rotation.T
    bounds = np.stack([corners.min(axis=0), corners.max(axis=0)], axis=0)
    return (bounds[0] + bounds[1]) / 2.0, bounds[1] - bounds[0]


def required_vector(value: Any, field: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{field} must be a finite 3-vector")
    return vector


def placement_bounds_gltf(
    placements: list[dict[str, Any]],
    *,
    placement_orientation: str,
    object_scale_factor: float,
) -> np.ndarray | None:
    bounds: list[np.ndarray] = []
    for placement in placements:
        if not placement_is_composable(placement, include_review=False):
            continue
        transform = placement_transform_to_gltf(
            placement,
            placement_orientation=placement_orientation,
            object_scale_factor=object_scale_factor,
        )
        corners = unit_box_corners()
        transformed = transform_points(corners, transform)
        bounds.append(np.stack([transformed.min(axis=0), transformed.max(axis=0)], axis=0))
    if not bounds:
        return None
    return merge_bounds(bounds)


def unavailable_room_alignment(reason: str) -> dict[str, Any]:
    return {
        "method": "unavailable",
        "reason": reason,
        "floor_plane_id": None,
        "wall_plane_ids": [],
        "usable_floor_bounds": None,
        "camera_floor_bounds": None,
        "source_floor_bounds": None,
        "applied_transform_gltf": np.eye(4, dtype=np.float64).tolist(),
    }


def vggt_room_alignment_transform(
    *,
    source_bounds: np.ndarray,
    placement_bounds: np.ndarray,
    plane_path: Path,
    orientation_transform: np.ndarray,
    coordinate_contract: dict[str, Any] | None,
    margin: float,
    depth_offset: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    planes = load_json(plane_path).get("planes", []) if plane_path.is_file() else []
    floor = first_room_plane(planes, "floor")
    walls = [plane for plane in planes if str(plane.get("id") or "") in {"back_wall", "right_wall", "left_wall"}]
    source_floor_bounds = oriented_plane_xz_bounds(floor, orientation_transform) if floor is not None else None
    camera_bounds = camera_floor_bounds_for_placement(placement_bounds, coordinate_contract, margin=margin, depth_offset=depth_offset)
    usable_floor_bounds = merge_xz_bounds(
        [bounds for bounds in (placement_xz_bounds(placement_bounds, margin), camera_bounds) if bounds is not None]
    )
    if source_floor_bounds is None:
        fit = vggt_room_background_fit_transform(
            source_bounds=source_bounds,
            placement_bounds=placement_bounds,
            margin=margin,
            depth_offset=depth_offset,
        )
        return fit, {
            "method": "camera_placement_uniform_fit_without_floor_plane",
            "reason": "missing_floor_plane",
            "floor_plane_id": None,
            "wall_plane_ids": [str(plane.get("id")) for plane in walls],
            "usable_floor_bounds": usable_floor_bounds.tolist() if usable_floor_bounds is not None else None,
            "camera_floor_bounds": camera_bounds.tolist() if camera_bounds is not None else None,
            "source_floor_bounds": None,
            "applied_transform_gltf": fit.tolist(),
        }
    if usable_floor_bounds is None:
        usable_floor_bounds = placement_xz_bounds(placement_bounds, margin)
    source_extent = source_floor_bounds[1] - source_floor_bounds[0]
    target_extent = usable_floor_bounds[1] - usable_floor_bounds[0]
    if np.any(source_extent <= 1e-8) or np.any(target_extent <= 1e-8):
        fit = vggt_room_background_fit_transform(
            source_bounds=source_bounds,
            placement_bounds=placement_bounds,
            margin=margin,
            depth_offset=depth_offset,
        )
        return fit, {
            "method": "fallback_placement_bounds_uniform_fit",
            "reason": "degenerate_floor_bounds",
            "floor_plane_id": floor.get("id") if floor else None,
            "wall_plane_ids": [str(plane.get("id")) for plane in walls],
            "usable_floor_bounds": usable_floor_bounds.tolist(),
            "camera_floor_bounds": camera_bounds.tolist() if camera_bounds is not None else None,
            "source_floor_bounds": source_floor_bounds.tolist(),
            "applied_transform_gltf": fit.tolist(),
        }
    scale_value = float(max(target_extent[0] / source_extent[0], target_extent[1] / source_extent[1]))
    source_floor_center = (source_floor_bounds[0] + source_floor_bounds[1]) / 2.0
    target_floor_center = (usable_floor_bounds[0] + usable_floor_bounds[1]) / 2.0
    floor_y = float(placement_bounds[0, 1])
    source_floor_y = oriented_floor_y(floor, orientation_transform, fallback=float(source_bounds[0, 1]))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.eye(3, dtype=np.float64) * scale_value
    transform[:3, 3] = np.asarray(
        [
            float(target_floor_center[0] - source_floor_center[0] * scale_value),
            float(floor_y - source_floor_y * scale_value),
            float(target_floor_center[1] - source_floor_center[1] * scale_value),
        ],
        dtype=np.float64,
    )
    return transform, {
        "method": "plane_camera_floor_uniform_alignment_v1",
        "floor_plane_id": floor.get("id") if floor else None,
        "wall_plane_ids": [str(plane.get("id")) for plane in walls],
        "usable_floor_bounds": usable_floor_bounds.tolist(),
        "camera_floor_bounds": camera_bounds.tolist() if camera_bounds is not None else None,
        "source_floor_bounds": source_floor_bounds.tolist(),
        "source_floor_y": float(source_floor_y),
        "target_floor_y": floor_y,
        "uniform_scale": scale_value,
        "applied_transform_gltf": transform.tolist(),
    }


def oriented_plane_xz_bounds(plane: dict[str, Any] | None, orientation_transform: np.ndarray) -> np.ndarray | None:
    if plane is None:
        return None
    vertices = plane_vertices_gltf(plane)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        return None
    oriented = transform_points(vertices, orientation_transform)
    if not np.isfinite(oriented).all():
        return None
    xz = oriented[:, [0, 2]]
    return np.stack([xz.min(axis=0), xz.max(axis=0)], axis=0)


def oriented_floor_y(plane: dict[str, Any] | None, orientation_transform: np.ndarray, *, fallback: float) -> float:
    if plane is None:
        return fallback
    vertices = plane_vertices_gltf(plane)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        return fallback
    oriented = transform_points(vertices, orientation_transform)
    if not np.isfinite(oriented).all():
        return fallback
    return float(np.median(oriented[:, 1]))


def placement_xz_bounds(placement_bounds: np.ndarray, margin: float) -> np.ndarray:
    xz = np.asarray([[placement_bounds[0, 0], placement_bounds[0, 2]], [placement_bounds[1, 0], placement_bounds[1, 2]]], dtype=np.float64)
    center = (xz[0] + xz[1]) / 2.0
    extent = np.maximum(xz[1] - xz[0], 1e-6)
    padded_extent = np.maximum(extent * max(float(margin), 1.0), extent + np.array([0.80, 1.00], dtype=np.float64))
    return np.stack([center - padded_extent / 2.0, center + padded_extent / 2.0], axis=0)


def camera_floor_bounds_for_placement(
    placement_bounds: np.ndarray,
    coordinate_contract: dict[str, Any] | None,
    *,
    margin: float,
    depth_offset: float,
) -> np.ndarray | None:
    contract = coordinate_contract or {}
    width = float(contract.get("image_width") or 0.0)
    height = float(contract.get("image_height") or 0.0)
    if width <= 0.0 or height <= 0.0:
        return None
    fov = float(contract.get("fov_degrees", DEFAULT_FOV_DEGREES))
    aspect = width / height
    depth_values = [-float(placement_bounds[0, 2]), -float(placement_bounds[1, 2])]
    back_depth = max(max(depth_values), 0.75) + max(float(depth_offset), 0.0)
    front_depth = max(min(depth_values), 0.35)
    half_width = np.tan(np.deg2rad(fov) / 2.0) * back_depth
    half_depth = max(back_depth - front_depth, 0.50)
    center_x = float((placement_bounds[0, 0] + placement_bounds[1, 0]) / 2.0)
    center_z = float((placement_bounds[0, 2] + placement_bounds[1, 2]) / 2.0)
    x_pad = half_width * max(float(margin), 1.0)
    z_pad = half_depth * max(0.5, min(aspect, 2.0))
    return np.asarray([[center_x - x_pad, center_z - z_pad], [center_x + x_pad, center_z + z_pad]], dtype=np.float64)


def merge_xz_bounds(bounds_items: list[np.ndarray]) -> np.ndarray | None:
    valid = [np.asarray(bounds, dtype=np.float64) for bounds in bounds_items if bounds is not None]
    valid = [bounds for bounds in valid if bounds.shape == (2, 2) and np.isfinite(bounds).all()]
    if not valid:
        return None
    return np.stack([np.min([bounds[0] for bounds in valid], axis=0), np.max([bounds[1] for bounds in valid], axis=0)], axis=0)


def background_fit_transform(
    *,
    source_bounds: np.ndarray,
    placement_bounds: np.ndarray,
    margin: float,
    depth_offset: float,
) -> np.ndarray:
    source_extent = source_bounds[1] - source_bounds[0]
    placement_extent = placement_bounds[1] - placement_bounds[0]
    if np.any(source_extent <= 1e-8) or np.any(placement_extent <= 1e-8):
        raise ValueError("Cannot fit background with degenerate bounds")

    target_extent = np.array(
        [
            placement_extent[0] * margin,
            placement_extent[1] * margin,
            placement_extent[2] * margin,
        ],
        dtype=np.float64,
    )
    target_center = (placement_bounds[0] + placement_bounds[1]) / 2.0
    # GLB Z is back toward the camera for SceneForge exports, so smaller Z is farther away.
    target_center[2] = placement_bounds[1, 2] - depth_offset - target_extent[2] / 2.0

    scale = target_extent / source_extent
    source_center = (source_bounds[0] + source_bounds[1]) / 2.0
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.diag(scale)
    transform[:3, 3] = target_center - source_center * scale
    return transform


def vggt_room_background_fit_transform(
    *,
    source_bounds: np.ndarray,
    placement_bounds: np.ndarray,
    margin: float,
    depth_offset: float,
) -> np.ndarray:
    source_extent = source_bounds[1] - source_bounds[0]
    placement_extent = placement_bounds[1] - placement_bounds[0]
    if np.any(source_extent <= 1e-8) or np.any(placement_extent <= 1e-8):
        raise ValueError("Cannot fit VGGT background with degenerate bounds")

    visual_margin = max(float(margin), 2.25)
    requested_extent = np.array(
        [
            max(float(placement_extent[0]) * visual_margin, float(placement_extent[0]) + 0.80),
            max(float(placement_extent[1]) * 1.90, 1.10),
            max(float(placement_extent[2]) * visual_margin, float(placement_extent[2]) + 1.00),
        ],
        dtype=np.float64,
    )
    horizontal_scale = np.array(
        [
            requested_extent[0] / source_extent[0],
            requested_extent[2] / source_extent[2],
        ],
        dtype=np.float64,
    )
    scale_value = float(np.max(horizontal_scale))
    target_extent = source_extent * scale_value
    target_center = (placement_bounds[0] + placement_bounds[1]) / 2.0
    target_center[1] = float(placement_bounds[0, 1] + target_extent[1] / 2.0)
    target_center[2] = float((placement_bounds[0, 2] + placement_bounds[1, 2]) / 2.0)

    source_center = (source_bounds[0] + source_bounds[1]) / 2.0
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.eye(3, dtype=np.float64) * scale_value
    transform[:3, 3] = target_center - source_center * scale_value
    return transform


def unit_box_corners() -> np.ndarray:
    return np.array(
        [
            [sx, sy, sz]
            for sx in (-0.5, 0.5)
            for sy in (-0.5, 0.5)
            for sz in (-0.5, 0.5)
        ],
        dtype=np.float64,
    )


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate([points, np.ones((len(points), 1), dtype=np.float64)], axis=1)
    return (homogeneous @ transform.T)[:, :3]


def add_scene_asset(
    scene: Any,
    path: Path,
    *,
    name_prefix: str,
    transform: np.ndarray | None = None,
    normalize: bool = True,
) -> dict[str, Any]:
    meshes = load_meshes(path)
    if not meshes:
        raise ValueError(f"No meshes found in {path}")
    source_bounds = combined_bounds(meshes)
    if transform is not None and normalize:
        asset_transform = transform @ normalization_transform(source_bounds)
    elif transform is not None:
        asset_transform = transform
    else:
        asset_transform = np.eye(4)
    transformed_bounds: list[np.ndarray] = []
    for index, mesh in enumerate(meshes):
        mesh = mesh.copy()
        mesh.apply_transform(asset_transform)
        transformed_bounds.append(np.asarray(mesh.bounds, dtype=np.float64))
        scene.add_geometry(mesh, geom_name=f"{name_prefix}_{index:03d}", node_name=f"{name_prefix}_{index:03d}")
    return {
        "path": str(path),
        "mesh_count": len(meshes),
        "source_bounds": source_bounds.tolist(),
        "transform_gltf": asset_transform.tolist(),
        "transformed_bounds": merge_bounds(transformed_bounds).tolist(),
        "mesh_cleanup": None,
    }


def load_meshes(path: Path) -> list[Any]:
    try:
        import trimesh
    except Exception as exc:
        raise RuntimeError("Scene composition requires trimesh from requirements.txt.") from exc

    loaded = trimesh.load(path, force="scene")
    if isinstance(loaded, trimesh.Trimesh):
        return [loaded]
    if not isinstance(loaded, trimesh.Scene):
        raise ValueError(f"Unsupported mesh asset type for {path}: {type(loaded).__name__}")
    dumped = loaded.dump(concatenate=False)
    if dumped is None:
        return []
    if isinstance(dumped, trimesh.Trimesh):
        return [dumped]
    return [mesh for mesh in dumped if isinstance(mesh, trimesh.Trimesh) and len(mesh.vertices) > 0]


def new_scene() -> Any:
    try:
        import trimesh
    except Exception as exc:
        raise RuntimeError("Scene composition requires trimesh from requirements.txt.") from exc
    return trimesh.Scene()


def combined_bounds(meshes: list[Any]) -> np.ndarray:
    bounds = [np.asarray(mesh.bounds, dtype=np.float64) for mesh in meshes if len(mesh.vertices) > 0]
    if not bounds:
        raise ValueError("Cannot compute bounds for empty mesh list")
    return merge_bounds(bounds)


def merge_bounds(bounds: list[np.ndarray]) -> np.ndarray:
    minimum = np.min([item[0] for item in bounds], axis=0)
    maximum = np.max([item[1] for item in bounds], axis=0)
    return np.stack([minimum, maximum], axis=0)


def normalization_transform(bounds: np.ndarray) -> np.ndarray:
    center = (bounds[0] + bounds[1]) / 2.0
    extent = bounds[1] - bounds[0]
    if np.any(extent <= 1e-8):
        raise ValueError("source mesh has degenerate bounds")
    uniform_extent = float(np.max(extent))
    if uniform_extent <= 1e-8:
        raise ValueError("source mesh has degenerate bounds")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.eye(3, dtype=np.float64) * (1.0 / uniform_extent)
    transform[:3, 3] = -center / uniform_extent
    return transform


def resolve_object_mesh_path(object_dir: Path, object_mesh_name: str) -> Path | None:
    requested = object_dir / object_mesh_name
    if requested.is_file():
        return requested
    metadata_path = object_dir / "hunyuan3d_metadata.json"
    if metadata_path.is_file():
        metadata = load_json(metadata_path)
        for key in ("textured_glb", "glb"):
            value = metadata.get(key)
            if value:
                candidate = object_dir / str(value)
                if candidate.is_file():
                    return candidate
    for name in ("hunyuan3d_textured.glb", "hunyuan3d_mesh.glb", "triposr_mesh.obj"):
        candidate = object_dir / name
        if candidate.is_file():
            return candidate
    return None


def index_object_dirs(objects_dir: Path) -> dict[int, Path]:
    indexed: dict[int, Path] = {}
    if not objects_dir.is_dir():
        return indexed
    for child in sorted(objects_dir.iterdir()):
        if not child.is_dir():
            continue
        metadata_path = child / "metadata.json"
        if metadata_path.is_file():
            metadata = load_json(metadata_path)
            if "id" in metadata:
                indexed[int(metadata["id"])] = child
                continue
        match = re.match(r"^(\d+)_", child.name)
        if match:
            indexed[int(match.group(1))] = child
    return indexed


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "object"


def safe_output_name(value: str) -> str:
    name = Path(value).name
    if not name.lower().endswith(".glb"):
        name = f"{name}.glb"
    return name
