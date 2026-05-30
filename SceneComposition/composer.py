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
PROJECTION_OCCLUDED_BOTTOM_AREA_REJECT_RATIO = 2.30
VGGT_CANDIDATE_POINT_SAMPLE_COUNT = 512
VGGT_CANDIDATE_LOSS_WEIGHT = 0.35
DEFAULT_UNIFORM_SCALE_CANDIDATES = (0.50, 0.60, 0.70, 0.82, 0.92, 1.0, 1.08, 1.25, 1.40)
LARGE_TARGET_HEIGHT_RATIO = 0.45
LARGE_TARGET_MIN_UNIFORM_SCALE = 0.70
FLOOR_SUPPORT_CONTACT_QUANTILE = 0.5
TABLETOP_SUPPORT_CONTACT_QUANTILE = 20.0


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
    background_fit: str = "room-corner",
    background_margin: float = 1.0,
    background_depth_offset: float = 0.12,
    background_vggt_dir: str | Path | None = None,
    background_stride: int = 16,
    clip_background_masks: bool = True,
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
                contact_quantile=support_contact_quantile(target.get("support_kind")),
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
                contact_quantile=support_contact_quantile(support_kind),
            )
        object_stats = add_scene_asset(
            scene,
            mesh_path,
            name_prefix=f"object_{detection_id:02d}_{slugify(label)}",
            transform=transform,
        )
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


def support_contact_quantile(support_kind: Any) -> float:
    if normalized_support_kind(support_kind) == "tabletop":
        return TABLETOP_SUPPORT_CONTACT_QUANTILE
    return FLOOR_SUPPORT_CONTACT_QUANTILE


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
                contact_quantile=support_contact_quantile(base["support_kind"]),
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
                contact_quantile=support_contact_quantile(base["support_kind"]),
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
) -> dict[str, Any]:
    target_bbox = bbox_array(placement.get("bbox_xyxy"))
    meshes = load_meshes(mesh_path)
    source_bounds = combined_bounds(meshes)
    initial_bbox = projected_transform_bbox(source_bounds, transform, coordinate_contract)
    vggt_fit = load_vggt_candidate_fit(placement)
    initial_vggt = vggt_candidate_transform_loss(vggt_fit, source_bounds, transform)
    vggt_loss_weight = VGGT_CANDIDATE_LOSS_WEIGHT if vggt_fit.get("available") else 0.0
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
        "delta_gltf": [0.0, 0.0, 0.0],
        "yaw_delta_radians": 0.0,
        "uniform_scale_delta": 1.0,
        "scale_candidates": list(DEFAULT_UNIFORM_SCALE_CANDIDATES),
        "minimum_scale_delta": min(DEFAULT_UNIFORM_SCALE_CANDIDATES),
        "minimum_scale_reason": "default",
        "candidate_count": 0,
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
    )
    initial_quality = projection_quality_report(initial_bbox, target_bbox, allow_occluded_bottom=allow_occluded_bottom)
    scale_candidates, scale_floor_report = scale_candidates_for_target(target_bbox, coordinate_contract)
    best_candidate_transform = np.asarray(transform, dtype=np.float64)
    best_candidate_bbox = initial_bbox
    best_candidate_loss = initial_loss
    best_candidate_bbox_loss = initial_bbox_loss
    best_candidate_vggt = initial_vggt
    best_candidate_delta = np.zeros(3, dtype=np.float64)
    best_candidate_yaw = 0.0
    best_candidate_scale = 1.0
    best_candidate_quality = initial_quality
    best_accepted_transform = np.asarray(transform, dtype=np.float64)
    best_accepted_bbox = initial_bbox
    best_accepted_loss = initial_loss if initial_quality.get("status") != "rejected" else float("inf")
    best_accepted_bbox_loss = initial_bbox_loss
    best_accepted_vggt = initial_vggt
    best_accepted_delta = np.zeros(3, dtype=np.float64)
    best_accepted_yaw = 0.0
    best_accepted_scale = 1.0
    best_accepted_quality = initial_quality
    candidate_count = 0
    accepted_candidate_count = 0
    for dx in (-0.16, -0.08, -0.04, 0.0, 0.04, 0.08, 0.16):
        for dz in (-0.56, -0.44, -0.32, -0.24, -0.16, -0.08, -0.04, 0.0, 0.04, 0.08, 0.16):
            for yaw in (-0.50, -0.25, 0.0, 0.25, 0.50):
                for scale in scale_candidates:
                    candidate_count += 1
                    candidate = candidate_transform(best_transform=transform, delta=np.array([dx, 0.0, dz]), yaw=yaw, scale=scale)
                    candidate, _delta = snap_transform_to_support_bounds(source_bounds, candidate, support_y)
                    projected = projected_transform_bbox(source_bounds, candidate, coordinate_contract)
                    if projected is None:
                        continue
                    bbox_loss = bbox_projection_loss(projected, target_bbox)
                    support_loss = support_penalty(source_bounds, candidate, support_y)
                    scale_loss = abs(float(np.log(scale))) * 0.08
                    vggt_loss = vggt_candidate_transform_loss(vggt_fit, source_bounds, candidate)
                    loss = objective_transform_loss(
                        bbox_loss=bbox_loss,
                        support_loss=support_loss,
                        scale_loss=scale_loss,
                        vggt_loss=vggt_loss.get("loss"),
                        vggt_loss_weight=vggt_loss_weight,
                    )
                    quality = projection_quality_report(projected, target_bbox, allow_occluded_bottom=allow_occluded_bottom)
                    if loss < best_candidate_loss:
                        best_candidate_loss = loss
                        best_candidate_bbox_loss = bbox_loss
                        best_candidate_vggt = vggt_loss
                        best_candidate_transform = candidate
                        best_candidate_bbox = projected
                        best_candidate_delta = np.array([dx, 0.0, dz], dtype=np.float64)
                        best_candidate_yaw = yaw
                        best_candidate_scale = scale
                        best_candidate_quality = quality
                    candidate_accepted = quality.get("status") != "rejected"
                    if candidate_accepted:
                        accepted_candidate_count += 1
                        if loss < best_accepted_loss:
                            best_accepted_loss = loss
                            best_accepted_bbox_loss = bbox_loss
                            best_accepted_vggt = vggt_loss
                            best_accepted_transform = candidate
                            best_accepted_bbox = projected
                            best_accepted_delta = np.array([dx, 0.0, dz], dtype=np.float64)
                            best_accepted_yaw = yaw
                            best_accepted_scale = scale
                            best_accepted_quality = quality
    accepted = np.isfinite(best_accepted_loss)
    final_transform = best_accepted_transform if accepted else np.asarray(transform, dtype=np.float64)
    final_bbox = best_accepted_bbox if accepted else initial_bbox
    final_loss = best_accepted_loss if accepted else initial_loss
    final_bbox_loss = best_accepted_bbox_loss if accepted else initial_bbox_loss
    final_vggt = best_accepted_vggt if accepted else initial_vggt
    final_quality = best_accepted_quality if accepted else initial_quality
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
        minimum_scale_delta=float(scale_floor_report["minimum_scale_delta"]),
        minimum_scale_reason=scale_floor_report["reason"],
        target_height_ratio=scale_floor_report["target_height_ratio"],
        candidate_count=int(candidate_count),
        accepted_candidate_count=int(accepted_candidate_count),
        candidate_projection_quality=best_candidate_quality,
        candidate_uniform_scale_delta=float(best_candidate_scale),
        candidate_delta_gltf=[float(value) for value in best_candidate_delta],
        candidate_yaw_delta_radians=float(best_candidate_yaw),
        vggt_candidate_fit=vggt_candidate_fit_report(
            vggt_fit,
            loss_weight=vggt_loss_weight,
            initial=initial_vggt,
            optimized=final_vggt,
            candidate=best_candidate_vggt,
        ),
        projection_quality=final_quality,
    )
    return {"transform": final_transform, "report": base_report}


def candidate_transform(*, best_transform: np.ndarray, delta: np.ndarray, yaw: float, scale: float) -> np.ndarray:
    transform = np.asarray(best_transform, dtype=np.float64).copy()
    center = transform[:3, 3].copy()
    linear = transform[:3, :3] * float(scale)
    rotation = yaw_rotation_gltf(float(yaw))
    transform[:3, :3] = rotation @ linear
    transform[:3, 3] = center + delta
    return transform


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
) -> float:
    total = float(bbox_loss) + float(support_loss) + float(scale_loss)
    if vggt_loss is not None and vggt_loss_weight > 0:
        total += float(vggt_loss) * float(vggt_loss_weight)
    return float(total)


def scale_candidates_for_target(
    target_bbox: np.ndarray,
    coordinate_contract: dict[str, Any] | None,
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
    candidates = tuple(float(scale) for scale in DEFAULT_UNIFORM_SCALE_CANDIDATES if float(scale) >= minimum_scale)
    if not candidates:
        candidates = (float(minimum_scale),)
    return candidates, {
        "minimum_scale_delta": float(minimum_scale),
        "reason": reason,
        "target_height_ratio": target_height_ratio,
    }


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
    loss = 0.35 * center_loss + 0.25 * extent_loss + 0.40 * outside_p90
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


def point_aabb_outside_distances(points: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    lower = np.maximum(bounds[0] - points, 0.0)
    upper = np.maximum(points - bounds[1], 0.0)
    offsets = np.maximum(lower, upper)
    return np.sqrt(np.sum(offsets * offsets, axis=1))


def sample_point_rows(values: np.ndarray, max_count: int) -> np.ndarray:
    if len(values) <= max_count:
        return np.asarray(values, dtype=np.float64)
    indices = np.linspace(0, len(values) - 1, max_count, dtype=np.int64)
    return np.asarray(values[indices], dtype=np.float64)


def scene_points_to_gltf_points(points: np.ndarray) -> np.ndarray:
    return np.asarray([[x, z, -y] for x, y, z in points], dtype=np.float64)


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
    projected_center = np.array([(projected[0] + projected[2]) / 2.0, (projected[1] + projected[3]) / 2.0])
    target_center = np.array([(target[0] + target[2]) / 2.0, (target[1] + target[3]) / 2.0])
    diagonal = max(float(np.linalg.norm([target[2] - target[0], target[3] - target[1]])), 1.0)
    center_loss = float(np.linalg.norm(projected_center - target_center) / diagonal)
    projected_area = max(float((projected[2] - projected[0]) * (projected[3] - projected[1])), 1.0)
    target_area = max(float((target[2] - target[0]) * (target[3] - target[1])), 1.0)
    area_loss = abs(np.log(projected_area / target_area))
    target_height = max(float(target[3] - target[1]), 1.0)
    edge_loss = (abs(float(projected[1] - target[1])) + abs(float(projected[3] - target[3]))) / target_height
    return float((1.0 - iou) + 0.55 * center_loss + 0.15 * area_loss + 0.35 * edge_loss)


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
        and area_ratio <= PROJECTION_OCCLUDED_BOTTOM_AREA_REJECT_RATIO
    )
    rejected = (
        (edge_ratio > PROJECTION_VERTICAL_EDGE_REJECT_RATIO or horizontal_edge_ratio > PROJECTION_HORIZONTAL_EDGE_REJECT_RATIO)
        and not occluded_bottom_accepted
        if accepted is None
        else not bool(accepted)
    )
    if occluded_bottom_accepted:
        status = "accepted_occluded_bottom"
        reason = "occluded_bottom_edge_tolerated"
    else:
        status = "rejected" if rejected else "accepted"
        reason = "vertical_edge_error" if rejected else "within_threshold"
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
        "occluded_bottom_area_threshold": PROJECTION_OCCLUDED_BOTTOM_AREA_REJECT_RATIO,
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
                contact_quantile=support_contact_quantile("floor"),
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


def add_vggt_background_mesh(
    scene: Any,
    *,
    vggt_dir: Path,
    objects_dir: Path,
    object_dirs: dict[int, Path],
    stride: int,
    clip_masks: bool,
    clip_dilation_px: int,
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
    scene.add_geometry(mesh, geom_name="background_camera_clipped_000", node_name="background_camera_clipped_000")
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    return {
        "path": str(points_path),
        "image_path": str(image_path),
        "mesh_count": 1,
        "source": "vggt_points_camera_clipped",
        "stride": stride,
        "clip_masks": bool(clip_masks),
        "clip_dilation_px": int(clip_dilation_px),
        "masked_pixel_ratio": float(mask.mean()) if mask.size else 0.0,
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
        "source_bounds": bounds.tolist(),
        "transform_gltf": np.eye(4, dtype=np.float64).tolist(),
        "transformed_bounds": bounds.tolist(),
    }


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
    contact_quantile: float = FLOOR_SUPPORT_CONTACT_QUANTILE,
) -> tuple[np.ndarray, float]:
    meshes = load_meshes(mesh_path)
    source_bounds = combined_bounds(meshes)
    contact_y = transformed_mesh_contact_y(meshes, source_bounds, transform, contact_quantile=contact_quantile)
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


def transformed_mesh_contact_y(
    meshes: list[Any],
    source_bounds: np.ndarray,
    transform: np.ndarray,
    *,
    contact_quantile: float,
) -> float:
    asset_transform = np.asarray(transform, dtype=np.float64) @ normalization_transform(source_bounds)
    vertices: list[np.ndarray] = []
    for mesh in meshes:
        mesh_vertices = np.asarray(mesh.vertices, dtype=np.float64)
        if mesh_vertices.ndim == 2 and mesh_vertices.shape[1] == 3 and len(mesh_vertices) > 0:
            vertices.append(transform_points(mesh_vertices, asset_transform))
    if not vertices:
        return float(transformed_bounds_from_source_bounds(source_bounds, transform)[0, 1])
    y_values = np.concatenate(vertices, axis=0)[:, 1]
    y_values = y_values[np.isfinite(y_values)]
    if len(y_values) == 0:
        return float(transformed_bounds_from_source_bounds(source_bounds, transform)[0, 1])
    quantile = float(np.clip(contact_quantile, 0.0, 100.0))
    return float(np.percentile(y_values, quantile))


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
    floor_y = float(placement_bounds[0, 1] - y_pad)
    z_back = float(placement_bounds[0, 2] - max(depth_offset, z_pad))
    z_front = float(placement_bounds[1, 2] + z_pad)
    camera_frustum_wall_top = max(0.0, -z_back) * 0.56
    wall_top_y = float(
        max(
            placement_bounds[1, 1] + max(float(extent[1]) * 1.60, 0.65),
            camera_frustum_wall_top,
        )
    )
    side_x = x_max

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
    extent_gltf = np.asarray(
        [
            scene_extent[0],
            scene_extent[2],
            scene_extent[1],
        ],
        dtype=np.float64,
    ) * float(object_scale_factor)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.diag(extent_gltf)
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
    transform[:3, :3] = axes_gltf @ np.diag(extent * float(object_scale_factor))
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
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.diag(1.0 / extent)
    transform[:3, 3] = -center / extent
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
