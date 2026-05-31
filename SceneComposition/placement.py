from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from SceneComposition.composer import (
    TABLE_SUPPORT_LABELS,
    TABLETOP_OBJECT_LABELS,
    bbox_array,
    bbox_height,
    bbox_iou,
    bbox_overlap_area,
    bounds_array,
    combined_bounds,
    index_object_dirs,
    is_table_support_label,
    is_tabletop_object_label,
    load_json,
    load_meshes,
    mesh_support_contact_report,
    normalization_transform,
    placement_is_composable,
    placement_transform_to_gltf,
    projection_quality_report,
    resolve_object_mesh_path,
    snap_transform_to_support_bounds,
    support_penalty,
    transformed_bounds_from_source_bounds,
    transform_points,
    optimize_transform_to_input,
)
from SceneGeometry.VGGT.pipeline import scene_point_to_gltf_vertex
from SceneGeometry.coordinate_contract import DEFAULT_FOV_DEGREES


SCHEMA_VERSION = 1
FLOOR_SUPPORT_LABELS = (
    "chair",
    "stool",
    "bench",
    "table",
    "desk",
    "sofa",
    "couch",
    "bed",
    "cabinet",
    "dresser",
    "bookshelf",
)
WALL_OBJECT_LABELS = ("picture", "painting", "poster", "mirror", "wall art", "wall light", "sconce")
SUPPORT_CONTACT_THRESHOLD_RATIO = 0.03
SUPPORT_FOOTPRINT_WARNING_RATIO = 0.25
SUPPORT_FOOTPRINT_REJECT_RATIO = 0.60
BACKGROUND_PENETRATION_EPSILON = 1e-5
UNKNOWN_SUPPORT_REVIEW_REASON = "no_reliable_support_plane"
VGGT_POINT_SAMPLE_COUNT = 2048
SILHOUETTE_FACE_SAMPLE_COUNT = 60000
STRUCTURAL_FLOOR_SUPPORT_MIN_POINTS = 128
STRUCTURAL_FLOOR_SUPPORT_BOUNDS_MARGIN_RATIO = 1.0


def choose_object_supports(
    *,
    object_geometry_path: str | Path,
    planes_path: str | Path,
    detections_path: str | Path | None,
    objects_dir: str | Path,
    output_dir: str | Path,
    object_mesh_name: str = "hunyuan3d_textured.glb",
    include_review: bool = False,
    placement_orientation: str = "upright",
    object_scale_factor: float = 0.85,
) -> dict[str, Any]:
    object_geometry_path = Path(object_geometry_path)
    planes_path = Path(planes_path)
    detections_path = Path(detections_path) if detections_path else None
    objects_dir = Path(objects_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    geometry = load_json(object_geometry_path)
    planes = load_json(planes_path) if planes_path.is_file() else {"planes": []}
    detections = load_json(detections_path) if detections_path and detections_path.is_file() else {}
    object_dirs = index_object_dirs(objects_dir)
    coordinate_contract = geometry.get("coordinate_contract") or detections.get("model_info", {}).get("fusion_contract")

    floor_plane = first_plane_with_subtype(planes, "floor")
    wall_plane = first_plane_with_subtype(planes, "wall")
    floor_y_gltf = plane_support_y_gltf(floor_plane)
    table_candidates = build_table_support_candidates(
        geometry.get("objects", []),
        object_dirs=object_dirs,
        object_mesh_name=object_mesh_name,
        include_review=include_review,
        placement_orientation=placement_orientation,
        object_scale_factor=object_scale_factor,
        floor_y_gltf=floor_y_gltf,
    )

    objects = [
        choose_support_for_object(
            placement=placement,
            object_dirs=object_dirs,
            object_mesh_name=object_mesh_name,
            include_review=include_review,
            floor_plane=floor_plane,
            floor_y_gltf=floor_y_gltf,
            wall_plane=wall_plane,
            table_candidates=table_candidates,
        )
        for placement in geometry.get("objects", [])
    ]
    report = {
        "schema_version": SCHEMA_VERSION,
        "object_geometry_path": str(object_geometry_path),
        "planes_path": str(planes_path),
        "detections_path": str(detections_path) if detections_path else None,
        "objects_dir": str(objects_dir),
        "object_mesh_name": object_mesh_name,
        "coordinate_contract": coordinate_contract,
        "placement_orientation": placement_orientation,
        "object_scale_factor": float(object_scale_factor),
        "support_selection": {
            "floor_labels": list(FLOOR_SUPPORT_LABELS),
            "table_support_labels": list(TABLE_SUPPORT_LABELS),
            "tabletop_object_labels": list(TABLETOP_OBJECT_LABELS),
            "wall_object_labels": list(WALL_OBJECT_LABELS),
            "unknown_support_review_reason": UNKNOWN_SUPPORT_REVIEW_REASON,
        },
        "artifacts": {
            "object_supports": str(output_dir / "object_supports.json"),
        },
        "objects": objects,
        "summary": support_summary(objects),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_dir / "object_supports.json", report)
    return report


def build_object_fit_targets(
    *,
    object_geometry_path: str | Path,
    supports_path: str | Path,
    objects_dir: str | Path,
    output_dir: str | Path,
    object_mesh_name: str = "hunyuan3d_textured.glb",
) -> dict[str, Any]:
    object_geometry_path = Path(object_geometry_path)
    supports_path = Path(supports_path)
    objects_dir = Path(objects_dir)
    output_dir = Path(output_dir)
    points_output_dir = output_dir / "visible_points"
    output_dir.mkdir(parents=True, exist_ok=True)
    points_output_dir.mkdir(parents=True, exist_ok=True)

    geometry = load_json(object_geometry_path)
    supports = load_json(supports_path)
    support_by_id = {int(item["detection_id"]): item for item in supports.get("objects", [])}
    object_dirs = index_object_dirs(objects_dir)

    objects = [
        build_fit_target_for_object(
            placement=placement,
            support_record=support_by_id.get(int(placement.get("detection_id", 0))),
            object_dirs=object_dirs,
            object_mesh_name=object_mesh_name,
            points_output_dir=points_output_dir,
        )
        for placement in geometry.get("objects", [])
    ]
    report = {
        "schema_version": SCHEMA_VERSION,
        "object_geometry_path": str(object_geometry_path),
        "supports_path": str(supports_path),
        "objects_dir": str(objects_dir),
        "object_mesh_name": object_mesh_name,
        "coordinate_contract": geometry.get("coordinate_contract") or supports.get("coordinate_contract"),
        "source_image_path": infer_source_image_path_from_reports(geometry, supports),
        "artifacts": {
            "object_fit_targets": str(output_dir / "object_fit_targets.json"),
            "visible_points_dir": str(points_output_dir),
        },
        "objects": objects,
        "summary": fit_target_summary(objects),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_dir / "object_fit_targets.json", report)
    return report


def fit_object_placements(
    *,
    supports_path: str | Path,
    fit_targets_path: str | Path,
    output_dir: str | Path,
    placement_orientation: str = "upright",
    object_scale_factor: float = 0.85,
    optimize_placements: bool = True,
) -> dict[str, Any]:
    supports_path = Path(supports_path)
    fit_targets_path = Path(fit_targets_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    supports = load_json(supports_path)
    targets = load_json(fit_targets_path)
    support_by_id = {int(item["detection_id"]): item for item in supports.get("objects", [])}
    coordinate_contract = targets.get("coordinate_contract") or supports.get("coordinate_contract")
    objects = [
        fit_placement_for_target(
            target=target,
            support_record=support_by_id.get(int(target.get("detection_id", 0))),
            coordinate_contract=coordinate_contract,
            placement_orientation=placement_orientation,
            object_scale_factor=object_scale_factor,
            optimize_placements=optimize_placements,
        )
        for target in targets.get("objects", [])
    ]
    objects, dependent_resolution = resolve_dependent_support_surfaces(
        objects=objects,
        targets=targets.get("objects", []),
        support_by_id=support_by_id,
        coordinate_contract=coordinate_contract,
        placement_orientation=placement_orientation,
        object_scale_factor=object_scale_factor,
        optimize_placements=optimize_placements,
    )
    objects, size_resolution = resolve_repeated_instance_size_priors(
        objects=objects,
        targets=targets.get("objects", []),
        support_by_id=support_by_id,
        coordinate_contract=coordinate_contract,
        placement_orientation=placement_orientation,
        object_scale_factor=object_scale_factor,
        optimize_placements=optimize_placements,
    )
    objects, facing_resolution = resolve_asymmetric_facing_priors(
        objects=objects,
        targets=targets.get("objects", []),
        support_by_id=support_by_id,
        coordinate_contract=coordinate_contract,
        placement_orientation=placement_orientation,
        object_scale_factor=object_scale_factor,
        optimize_placements=optimize_placements,
    )
    visibility_resolution = annotate_visibility_aware_silhouettes(
        objects=objects,
        targets=targets.get("objects", []),
        coordinate_contract=coordinate_contract,
    )
    review_resolution = reconcile_visibility_explained_projection_reviews(objects)
    annotate_pairwise_collision_metrics(objects)
    quality = placement_quality_report(objects)
    report = {
        "schema_version": SCHEMA_VERSION,
        "coordinate_contract": coordinate_contract,
        "coordinate_contract_labels": {
            "scene_space": "SceneForge camera space: X right, Y depth away, Z up",
            "gltf_space": "GLB X right, Y up, Z toward camera negative depth",
            "pixel_space": "source image pixels, origin at top-left",
        },
        "supports_path": str(supports_path),
        "fit_targets_path": str(fit_targets_path),
        "source_image_path": targets.get("source_image_path"),
        "placement_orientation": placement_orientation,
        "object_scale_factor": float(object_scale_factor),
        "optimize_placements": bool(optimize_placements),
        "thresholds": {
            "support_contact_distance_ratio": SUPPORT_CONTACT_THRESHOLD_RATIO,
            "support_footprint_outside_warning_ratio": SUPPORT_FOOTPRINT_WARNING_RATIO,
            "support_footprint_outside_reject_ratio": SUPPORT_FOOTPRINT_REJECT_RATIO,
            "background_penetration_epsilon": BACKGROUND_PENETRATION_EPSILON,
        },
        "artifacts": {
            "object_placements": str(output_dir / "object_placements.json"),
            "placement_quality": str(output_dir / "placement_quality.json"),
        },
        "objects": objects,
        "quality": quality,
        "dependent_support_resolution": dependent_resolution,
        "repeated_instance_size_resolution": size_resolution,
        "asymmetric_facing_resolution": facing_resolution,
        "visibility_resolution": visibility_resolution,
        "visibility_review_resolution": review_resolution,
        "summary": {
            "placement_count": len(objects),
            "accepted_count": sum(1 for item in objects if item["status"] == "accepted"),
            "failed_count": sum(1 for item in objects if item["status"] == "failed"),
            "needs_review_count": sum(1 for item in objects if item["needs_review"]),
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_dir / "object_placements.json", report)
    write_json(output_dir / "placement_quality.json", quality)
    return report



def choose_support_for_object(
    *,
    placement: dict[str, Any],
    object_dirs: dict[int, Path],
    object_mesh_name: str,
    include_review: bool,
    floor_plane: dict[str, Any] | None,
    floor_y_gltf: float | None,
    wall_plane: dict[str, Any] | None,
    table_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    detection_id = int(placement.get("detection_id", 0))
    label = str(placement.get("detector_label") or "object")
    object_dir = object_dirs.get(int(placement.get("source_object_dir_id") or detection_id))
    mesh_path = resolve_object_mesh_path(object_dir, object_mesh_name) if object_dir else None
    warnings: list[str] = []
    if mesh_path is None:
        warnings.append("missing_object_mesh")
    if placement.get("box_type") == "failed":
        warnings.append("object_geometry_failed")
    if bool(placement.get("needs_review", False)):
        warnings.append("object_geometry_needs_review")

    base = {
        "detection_id": detection_id,
        "detector_label": placement.get("detector_label"),
        "bbox_xyxy_px": placement.get("bbox_xyxy"),
        "mesh_path": str(mesh_path) if mesh_path else None,
        "status": "accepted",
        "needs_review": False,
        "reason": None,
        "support": None,
        "evidence": {
            "mesh_available": mesh_path is not None,
            "box_type": placement.get("box_type"),
            "point_count": int(placement.get("point_count") or 0),
            "valid_point_ratio": float(placement.get("valid_point_ratio") or 0.0),
            "mask_path": placement.get("mask_path"),
        },
        "warnings": warnings,
    }
    if not placement_is_composable(placement, include_review=include_review):
        base.update(
            status="skipped",
            needs_review=True,
            reason="not_composable_from_object_geometry",
            support=unknown_support(),
        )
        return base

    table_match = best_table_candidate_for_object(placement.get("bbox_xyxy"), table_candidates)
    if is_tabletop_object_label(label) and table_match is not None:
        base["support"] = tabletop_support(table_match)
        return base

    if is_wall_object_label(label) and wall_plane is not None:
        base["support"] = plane_support(
            mode="wall_4dof",
            support_kind="wall",
            plane=wall_plane,
            support_y_gltf=None,
            confidence=0.62,
            reason="wall_label_with_wall_plane",
        )
        return base

    if should_use_floor_support(label) and floor_plane is not None and floor_y_gltf is not None:
        base["support"] = plane_support(
            mode="floor_4dof",
            support_kind="floor",
            plane=floor_plane,
            support_y_gltf=floor_y_gltf,
            confidence=0.78 if is_floor_object_label(label) else 0.66,
            reason="floor_compatible_label_with_floor_plane",
        )
        return base

    base.update(
        needs_review=True,
        reason=UNKNOWN_SUPPORT_REVIEW_REASON,
        support=unknown_support(),
    )
    return base


def build_fit_target_for_object(
    *,
    placement: dict[str, Any],
    support_record: dict[str, Any] | None,
    object_dirs: dict[int, Path],
    object_mesh_name: str,
    points_output_dir: Path,
) -> dict[str, Any]:
    detection_id = int(placement.get("detection_id", 0))
    label = str(placement.get("detector_label") or "object")
    object_dir = object_dirs.get(int(placement.get("source_object_dir_id") or detection_id))
    mesh_path = resolve_object_mesh_path(object_dir, object_mesh_name) if object_dir else None
    warnings: list[str] = []
    if object_dir is None:
        warnings.append("missing_object_dir")
    if mesh_path is None:
        warnings.append("missing_object_mesh")
    if bbox_array(placement.get("bbox_xyxy")) is None:
        warnings.append("missing_or_invalid_bbox")
    if placement.get("box_type") == "failed":
        warnings.append("object_geometry_failed")

    visible_points_path = write_visible_points_npy(placement, detection_id, label, points_output_dir)
    mesh_bounds = mesh_bounds_for_path(mesh_path)
    mesh_quality = mesh_quality_report(mesh_path, mesh_bounds)
    support = (support_record or {}).get("support") or unknown_support()
    needs_review = bool((support_record or {}).get("needs_review")) or bool(placement.get("needs_review")) or bool(warnings)
    status = "ready" if mesh_path is not None and placement.get("box_type") != "failed" else "failed"
    return {
        "detection_id": detection_id,
        "detector_label": placement.get("detector_label"),
        "bbox_xyxy_px": placement.get("bbox_xyxy"),
        "mask_path": placement.get("mask_path"),
        "mesh_path": str(mesh_path) if mesh_path else None,
        "visible_points_scene_path": str(visible_points_path) if visible_points_path else None,
        "visible_point_count": int(placement.get("point_count") or 0),
        "visible_point_valid_ratio": float(placement.get("valid_point_ratio") or 0.0),
        "mesh_bounds_gltf": mesh_bounds.tolist() if mesh_bounds is not None else None,
        "mesh_quality": mesh_quality,
        "support": support,
        "placement_geometry": {
            "box_type": placement.get("box_type"),
            "center_xyz": placement.get("center_xyz"),
            "extent_xyz": placement.get("extent_xyz"),
            "rotation_matrix": placement.get("rotation_matrix"),
        },
        "status": status,
        "needs_review": needs_review,
        "warnings": warnings + list((support_record or {}).get("warnings") or []),
    }


def fit_placement_for_target(
    *,
    target: dict[str, Any],
    support_record: dict[str, Any] | None,
    coordinate_contract: dict[str, Any] | None,
    placement_orientation: str,
    object_scale_factor: float,
    optimize_placements: bool,
    facing_target_gltf: Any = None,
    physical_size_target_extent_gltf: Any = None,
) -> dict[str, Any]:
    detection_id = int(target.get("detection_id", 0))
    label = str(target.get("detector_label") or "object")
    support = target.get("support") or (support_record or {}).get("support") or unknown_support()
    mesh_path = Path(str(target.get("mesh_path"))) if target.get("mesh_path") else None
    base = {
        "detection_id": detection_id,
        "detector_label": target.get("detector_label"),
        "mesh_path": str(mesh_path) if mesh_path else None,
        "mask_path": target.get("mask_path"),
        "status": "failed",
        "reason": None,
        "needs_review": True,
        "support": support,
        "degrees_of_freedom": degrees_of_freedom_for_support(support),
        "transform_gltf": None,
        "source_bounds": None,
        "transformed_bounds": None,
        "support_snap_delta": None,
        "render_to_input_optimization": None,
        "orientation_search": None,
        "support_contact": None,
        "losses": empty_losses(),
        "quality": {
            "projection_status": "unavailable",
            "support_status": "unavailable",
            "collision_status": "not_evaluated",
            "warnings": list(target.get("warnings") or []),
        },
    }
    if mesh_path is None or not mesh_path.is_file():
        base.update(reason="missing_object_mesh")
        return base
    if target.get("status") == "failed":
        base.update(reason="fit_target_failed")
        return base

    try:
        placement = placement_record_from_target(target)
        source_bounds = combined_bounds(load_meshes(mesh_path))
        transform = placement_transform_to_gltf(
            placement,
            placement_orientation=placement_orientation,
            object_scale_factor=object_scale_factor,
        )
        support_y = support.get("support_y_gltf")
        support_target = None
        support_snap_delta = 0.0
        if support_y is not None:
            transform, support_snap_delta = snap_transform_to_support_bounds(source_bounds, transform, float(support_y))
            support_target = {
                "support_kind": support.get("support_kind"),
                "support_detection_id": support.get("support_detection_id"),
                "support_y": float(support_y),
            }
        optimization = optimize_transform_to_input(
            mesh_path=mesh_path,
            placement=placement,
            transform=transform,
            support_target=support_target,
            coordinate_contract=coordinate_contract,
            enabled=bool(optimize_placements),
            facing_target_gltf=facing_target_gltf,
            physical_size_target_extent_gltf=physical_size_target_extent_gltf,
        )
        mode = str(support.get("mode") or "unknown_5dof")
        if support_target is None and mode == "unknown_5dof":
            optimization = optimize_unknown_5dof_transform(
                source_bounds=source_bounds,
                placement=placement,
                transform=transform,
                coordinate_contract=coordinate_contract,
                enabled=bool(optimize_placements),
            )
        transform = np.asarray(optimization["transform"], dtype=np.float64)
        transformed_bounds = transformed_bounds_from_source_bounds(source_bounds, transform)
        support_contact_loss = support_contact_distance(source_bounds, transform, support_y)
        support_contact = mesh_support_contact_report(mesh_path, transform, support.get("support_kind"))
        projection_quality = optimization["report"].get("projection_quality") or projection_quality_report(None, None)
        footprint = support_footprint_report(transformed_bounds, support)
        penetration = support_penetration_report(transformed_bounds, support)
        support_status = support_quality_status(support_contact_loss, transformed_bounds, footprint, penetration, support_y)
        collision_status = "rejected" if penetration.get("penetrates_support") else "accepted"
        silhouette = silhouette_proxy_report(optimization["report"])
        silhouette_render = silhouette_render_report(
            target=target,
            mesh_path=mesh_path,
            source_bounds=source_bounds,
            transform=transform,
            coordinate_contract=coordinate_contract,
        )
        silhouette_loss = silhouette_render.get("loss") if silhouette_render.get("loss") is not None else silhouette.get("loss")
        vggt_points = vggt_point_loss_report(
            target=target,
            mesh_path=mesh_path,
            source_bounds=source_bounds,
            transform=transform,
        )
        needs_review = (
            bool(target.get("needs_review"))
            or mode == "unknown_5dof"
            or projection_quality.get("status") == "rejected"
            or projection_quality.get("status") == "accepted_occluded_bottom"
            or support_status == "rejected"
            or collision_status == "rejected"
        )
        losses = placement_losses(
            bbox_loss=optimization["report"].get("optimized_bbox_loss", optimization["report"].get("optimized_loss")),
            silhouette_loss=silhouette_loss,
            vggt_point_loss=vggt_points.get("loss"),
            support_contact=support_contact_loss,
            background_collision=penetration.get("penetration_depth_gltf"),
            scale_delta=optimization["report"].get("uniform_scale_delta"),
            unknown_support=mode == "unknown_5dof",
        )
        base.update(
            status="accepted",
            reason=None if not needs_review else placement_review_reason(mode, projection_quality, support_status),
            needs_review=needs_review,
            transform_gltf=transform.tolist(),
            source_bounds=source_bounds.tolist(),
            transformed_bounds=transformed_bounds.tolist(),
            support_snap_delta=float(support_snap_delta),
            render_to_input_optimization=optimization["report"],
            orientation_search=optimization["report"].get("orientation_search"),
            support_contact=support_contact,
            losses=losses,
            quality={
                "projection_status": projection_quality.get("status"),
                "support_status": support_status,
                "collision_status": collision_status,
                "silhouette_proxy": silhouette,
                "silhouette_render": silhouette_render,
                "vggt_points": vggt_points,
                "support_contact": support_contact,
                "support_footprint": footprint,
                "support_penetration": penetration,
                "warnings": placement_warnings(
                    target,
                    mode,
                    projection_quality,
                    support_status,
                    collision_status,
                    footprint,
                    penetration,
                ),
            },
        )
        return base
    except Exception as exc:
        base.update(reason=f"placement_fit_failed: {exc}")
        return base


def resolve_dependent_support_surfaces(
    *,
    objects: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    support_by_id: dict[int, dict[str, Any]],
    coordinate_contract: dict[str, Any] | None,
    placement_orientation: str,
    object_scale_factor: float,
    optimize_placements: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id = {int(item.get("detection_id", 0)): item for item in objects}
    refreshed = list(objects)
    refreshed_ids: list[int] = []
    for index, target in enumerate(targets):
        detection_id = int(target.get("detection_id", 0))
        support = dict(target.get("support") or (support_by_id.get(detection_id) or {}).get("support") or {})
        support_detection_id = support.get("support_detection_id")
        if support_detection_id is None:
            continue
        support_object = by_id.get(int(support_detection_id))
        support_bounds = bounds_array((support_object or {}).get("transformed_bounds"))
        if support_bounds is None:
            continue
        current_support_y = support.get("support_y_gltf")
        final_support_y = float(support_bounds[1, 1])
        support_changed = current_support_y is None or abs(float(current_support_y) - final_support_y) > 1e-6
        if not support_changed:
            continue
        support["support_y_gltf"] = final_support_y
        support["support_bounds_gltf"] = support_bounds.tolist()
        support["resolved_from_final_support_detection_id"] = int(support_detection_id)
        support["reason"] = f"{support.get('reason') or 'support_object'}_final_surface"
        retargeted = dict(target)
        retargeted["support"] = support
        record = fit_placement_for_target(
            target=retargeted,
            support_record={"support": support},
            coordinate_contract=coordinate_contract,
            placement_orientation=placement_orientation,
            object_scale_factor=object_scale_factor,
            optimize_placements=optimize_placements,
        )
        refreshed[index] = record
        by_id[detection_id] = record
        refreshed_ids.append(detection_id)
    return refreshed, {
        "enabled": True,
        "refit_count": len(refreshed_ids),
        "refit_detection_ids": refreshed_ids,
    }


def resolve_repeated_instance_size_priors(
    *,
    objects: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    support_by_id: dict[int, dict[str, Any]],
    coordinate_contract: dict[str, Any] | None,
    placement_orientation: str,
    object_scale_factor: float,
    optimize_placements: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in objects:
        if item.get("status") != "accepted":
            continue
        bounds = bounds_array(item.get("transformed_bounds"))
        if bounds is None:
            continue
        key = repeated_instance_group_key(item)
        if key is None:
            continue
        groups.setdefault(key, []).append(item)

    target_by_id = {int(target.get("detection_id", 0)): target for target in targets}
    refreshed = list(objects)
    refit_ids: list[int] = []
    group_reports: list[dict[str, Any]] = []
    for (label_key, support_kind), members in sorted(groups.items()):
        if len(members) < 2:
            continue
        extents = []
        for member in members:
            bounds = bounds_array(member.get("transformed_bounds"))
            if bounds is not None:
                extents.append(bounds[1] - bounds[0])
        if len(extents) < 2:
            continue
        extent_array = np.asarray(extents, dtype=np.float64)
        target_extent = np.median(extent_array, axis=0)
        if target_extent.shape != (3,) or not np.isfinite(target_extent).all() or np.any(target_extent <= 1e-8):
            continue
        volumes = np.prod(np.maximum(extent_array, 1e-8), axis=1)
        target_volume = float(np.median(volumes))
        if not np.isfinite(target_volume) or target_volume <= 1e-12:
            continue
        size_target = {
            "target_extent_gltf": [float(value) for value in target_extent],
            "target_volume_gltf": target_volume,
        }
        group_refit_ids: list[int] = []
        for member in members:
            detection_id = int(member.get("detection_id", 0))
            target = target_by_id.get(detection_id)
            if target is None:
                continue
            support = member.get("support") or (support_by_id.get(detection_id) or {}).get("support") or {}
            record = fit_placement_for_target(
                target=target,
                support_record={"support": support},
                coordinate_contract=coordinate_contract,
                placement_orientation=placement_orientation,
                object_scale_factor=object_scale_factor,
                optimize_placements=optimize_placements,
                physical_size_target_extent_gltf=size_target,
            )
            if record.get("status") != "accepted":
                continue
            report = ((record.get("render_to_input_optimization") or {}).get("physical_size_prior") or {})
            if report.get("status") != "accepted":
                continue
            index = next((idx for idx, item in enumerate(refreshed) if int(item.get("detection_id", 0)) == detection_id), None)
            if index is None:
                continue
            refreshed[index] = record
            refit_ids.append(detection_id)
            group_refit_ids.append(detection_id)
        if group_refit_ids:
            group_reports.append(
                {
                    "group_label_key": label_key,
                    "support_kind": support_kind,
                    "target_extent_gltf": [float(value) for value in target_extent],
                    "target_volume_gltf": target_volume,
                    "member_count": len(members),
                    "refit_detection_ids": group_refit_ids,
                }
            )
    return refreshed, {
        "enabled": True,
        "method": "same_label_support_physical_extent_median_prior",
        "group_count": len(group_reports),
        "refit_count": len(refit_ids),
        "refit_detection_ids": refit_ids,
        "groups": group_reports,
    }


def repeated_instance_group_key(item: dict[str, Any]) -> tuple[str, str] | None:
    label = normalized_instance_label(item.get("detector_label"))
    support_kind = str((item.get("support") or {}).get("support_kind") or "")
    if not label or not support_kind:
        return None
    return label, support_kind


def normalized_instance_label(value: Any) -> str:
    label = str(value or "").strip().lower()
    return " ".join(label.split())


def resolve_asymmetric_facing_priors(
    *,
    objects: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    support_by_id: dict[int, dict[str, Any]],
    coordinate_contract: dict[str, Any] | None,
    placement_orientation: str,
    object_scale_factor: float,
    optimize_placements: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    anchors = asymmetric_facing_anchor_candidates(objects)
    if not anchors:
        return objects, {
            "enabled": True,
            "refit_count": 0,
            "refit_detection_ids": [],
            "reason": "no_anchor_candidates",
        }
    refreshed = list(objects)
    by_id = {int(item.get("detection_id", 0)): item for item in objects}
    refit_ids: list[int] = []
    refit_targets: dict[int, dict[str, Any]] = {}
    for index, target in enumerate(targets):
        detection_id = int(target.get("detection_id", 0))
        current = by_id.get(detection_id)
        if current is None or current.get("status") != "accepted":
            continue
        support = current.get("support") or (support_by_id.get(detection_id) or {}).get("support") or {}
        if support.get("support_kind") != "floor":
            continue
        current_bounds = bounds_array(current.get("transformed_bounds"))
        if current_bounds is None:
            continue
        if any(int(anchor["detection_id"]) == detection_id for anchor in anchors):
            continue
        anchor = nearest_facing_anchor(current_bounds, anchors)
        if anchor is None:
            continue
        record = fit_placement_for_target(
            target=target,
            support_record={"support": support},
            coordinate_contract=coordinate_contract,
            placement_orientation=placement_orientation,
            object_scale_factor=object_scale_factor,
            optimize_placements=optimize_placements,
            facing_target_gltf=anchor["center_gltf"],
            physical_size_target_extent_gltf=((current.get("render_to_input_optimization") or {}).get("physical_size_prior") or {}),
        )
        if record.get("status") != "accepted":
            continue
        report = ((record.get("render_to_input_optimization") or {}).get("mesh_facing_prior") or {})
        if report.get("status") != "accepted":
            continue
        refreshed[index] = record
        by_id[detection_id] = record
        refit_ids.append(detection_id)
        refit_targets[detection_id] = {
            "anchor_detection_id": int(anchor["detection_id"]),
            "anchor_center_gltf": anchor["center_gltf"],
            "anchor_reason": anchor["reason"],
            "optimized_loss": (report.get("optimized") or {}).get("loss"),
        }
    return refreshed, {
        "enabled": True,
        "method": "mesh_vertical_asymmetry_to_nearby_low_wide_anchor",
        "anchor_count": len(anchors),
        "anchors": anchors,
        "refit_count": len(refit_ids),
        "refit_detection_ids": refit_ids,
        "refit_targets": refit_targets,
    }


def asymmetric_facing_anchor_candidates(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    for item in objects:
        if item.get("status") != "accepted":
            continue
        support = item.get("support") or {}
        if support.get("support_kind") != "floor":
            continue
        bounds = bounds_array(item.get("transformed_bounds"))
        if bounds is None:
            continue
        extent = bounds[1] - bounds[0]
        horizontal_max = max(float(extent[0]), float(extent[2]))
        height = float(extent[1])
        footprint_area = float(max(extent[0], 0.0) * max(extent[2], 0.0))
        if horizontal_max <= 1e-8 or height <= 1e-8:
            continue
        if horizontal_max < height * 1.10:
            continue
        if footprint_area < 0.035:
            continue
        center = ((bounds[0] + bounds[1]) / 2.0).tolist()
        anchors.append(
            {
                "detection_id": int(item.get("detection_id", 0)),
                "detector_label": item.get("detector_label"),
                "center_gltf": [float(value) for value in center],
                "footprint_radius": float(np.linalg.norm(extent[[0, 2]]) / 2.0),
                "height": height,
                "footprint_area": footprint_area,
                "reason": "low_wide_floor_object",
            }
        )
    return anchors


def nearest_facing_anchor(bounds: np.ndarray, anchors: list[dict[str, Any]]) -> dict[str, Any] | None:
    center = (bounds[0] + bounds[1]) / 2.0
    extent = bounds[1] - bounds[0]
    radius = float(np.linalg.norm(extent[[0, 2]]) / 2.0)
    best: dict[str, Any] | None = None
    best_distance = float("inf")
    for anchor in anchors:
        anchor_center = np.asarray(anchor["center_gltf"], dtype=np.float64)
        distance = float(np.linalg.norm(anchor_center[[0, 2]] - center[[0, 2]]))
        max_distance = max((radius + float(anchor["footprint_radius"])) * 3.0, 0.45)
        if distance > max_distance:
            continue
        if distance < best_distance:
            best = anchor
            best_distance = distance
    return best


def annotate_visibility_aware_silhouettes(
    *,
    objects: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    coordinate_contract: dict[str, Any] | None,
) -> dict[str, Any]:
    target_by_id = {int(target.get("detection_id", 0)): target for target in targets}
    render_cache: dict[int, dict[str, Any]] = {}
    for item in objects:
        detection_id = int(item.get("detection_id", 0))
        if item.get("status") != "accepted":
            continue
        target = target_by_id.get(detection_id)
        mesh_path = Path(str(item.get("mesh_path"))) if item.get("mesh_path") else None
        source_bounds = bounds_array(item.get("source_bounds"))
        transform = matrix_array(item.get("transform_gltf"))
        if target is None or mesh_path is None or source_bounds is None or transform is None:
            continue
        rendered = rendered_mask_for_item(
            target=target,
            mesh_path=mesh_path,
            source_bounds=source_bounds,
            transform=transform,
            coordinate_contract=coordinate_contract,
        )
        if rendered is not None:
            render_cache[detection_id] = rendered

    updated_ids: list[int] = []
    for item in objects:
        detection_id = int(item.get("detection_id", 0))
        rendered = render_cache.get(detection_id)
        if rendered is None:
            ensure_quality(item)["silhouette_visibility"] = unavailable_silhouette_visibility_report("missing_render_cache")
            continue
        target_bounds = bounds_array(item.get("transformed_bounds"))
        occluder_mask = np.zeros(rendered["rendered"].shape, dtype=bool)
        occluder_ids: list[int] = []
        for other_id, other_rendered in render_cache.items():
            if other_id == detection_id:
                continue
            other = next((candidate for candidate in objects if int(candidate.get("detection_id", 0)) == other_id), None)
            if other is None:
                continue
            other_bounds = bounds_array(other.get("transformed_bounds"))
            if target_bounds is None or other_bounds is None:
                continue
            if not likely_front_occluder(target_bounds, other_bounds):
                continue
            overlap = np.logical_and(rendered["rendered"], other_rendered["rendered"])
            if not bool(overlap.any()):
                continue
            occluder_mask |= other_rendered["rendered"]
            occluder_ids.append(other_id)
        visible = np.logical_and(rendered["rendered"], ~occluder_mask)
        report = silhouette_visibility_report(
            rendered=rendered["rendered"],
            visible=visible,
            target_mask=rendered["target"],
            occluder_ids=occluder_ids,
            render_info=rendered,
        )
        ensure_quality(item)["silhouette_visibility"] = report
        item.setdefault("losses", {})["visible_silhouette"] = report.get("loss")
        updated_ids.append(detection_id)
    return {
        "enabled": True,
        "updated_count": len(updated_ids),
        "updated_detection_ids": updated_ids,
    }


def reconcile_visibility_explained_projection_reviews(objects: list[dict[str, Any]]) -> dict[str, Any]:
    resolved_ids: list[int] = []
    for item in objects:
        if item.get("status") != "accepted" or not item.get("needs_review"):
            continue
        quality = ensure_quality(item)
        projection_status = quality.get("projection_status")
        if projection_status != "accepted_occluded_bottom":
            continue
        if quality.get("support_status") != "accepted" or quality.get("collision_status") != "accepted":
            continue
        visibility = quality.get("silhouette_visibility") or {}
        if visibility.get("status") != "accepted":
            continue
        if not visibility.get("occluder_detection_ids"):
            continue
        visible_area = int(visibility.get("visible_area_px") or 0)
        occluded_area = int(visibility.get("occluded_area_px") or 0)
        if visible_area <= 0 or occluded_area <= 0:
            continue
        quality["projection_review_resolution"] = {
            "status": "resolved",
            "reason": "occluded_bottom_explained_by_front_silhouettes",
            "occluder_detection_ids": visibility.get("occluder_detection_ids") or [],
            "occluded_area_ratio": visibility.get("occluded_area_ratio"),
        }
        item["needs_review"] = False
        item["reason"] = None
        resolved_ids.append(int(item.get("detection_id", 0)))
    return {
        "enabled": True,
        "method": "visibility_explained_projection_review_reconciliation",
        "resolved_count": len(resolved_ids),
        "resolved_detection_ids": resolved_ids,
    }


def rendered_mask_for_item(
    *,
    target: dict[str, Any],
    mesh_path: Path,
    source_bounds: np.ndarray,
    transform: np.ndarray,
    coordinate_contract: dict[str, Any] | None,
) -> dict[str, Any] | None:
    contract = coordinate_contract or {}
    width = int(contract.get("image_width") or 0)
    height = int(contract.get("image_height") or 0)
    if width <= 0 or height <= 0:
        return None
    target_mask = load_target_mask(target.get("mask_path"), width, height)
    if target_mask is None:
        return None
    try:
        rendered_image, face_count = render_mesh_silhouette_mask(
            mesh_path=mesh_path,
            source_bounds=source_bounds,
            transform=transform,
            coordinate_contract=contract,
            width=width,
            height=height,
        )
    except Exception:
        return None
    rendered = np.asarray(rendered_image, dtype=np.uint8) > 0
    if not bool(rendered.any()):
        return None
    return {
        "rendered": rendered,
        "target": target_mask,
        "mask_path": target.get("mask_path"),
        "rendered_face_count": int(face_count),
    }


def load_target_mask(mask_path_value: Any, width: int, height: int) -> np.ndarray | None:
    if not mask_path_value:
        return None
    mask_path = Path(str(mask_path_value))
    if not mask_path.is_file():
        return None
    try:
        mask_image = Image.open(mask_path).convert("L")
    except Exception:
        return None
    if mask_image.size != (width, height):
        mask_image = mask_image.resize((width, height), Image.Resampling.NEAREST)
    target_mask = np.asarray(mask_image, dtype=np.uint8) > 127
    return target_mask if bool(target_mask.any()) else None


def likely_front_occluder(target_bounds: np.ndarray, other_bounds: np.ndarray) -> bool:
    target_back_depth = -float(target_bounds[0, 2])
    other_front_depth = -float(other_bounds[1, 2])
    return other_front_depth <= target_back_depth + 0.02


def silhouette_visibility_report(
    *,
    rendered: np.ndarray,
    visible: np.ndarray,
    target_mask: np.ndarray,
    occluder_ids: list[int],
    render_info: dict[str, Any],
) -> dict[str, Any]:
    visible_area = int(visible.sum())
    target_area = int(target_mask.sum())
    rendered_area = int(rendered.sum())
    intersection = int(np.logical_and(visible, target_mask).sum())
    union = int(np.logical_or(visible, target_mask).sum())
    false_positive = int(np.logical_and(visible, ~target_mask).sum())
    false_negative = int(np.logical_and(~visible, target_mask).sum())
    iou = float(intersection / union) if union else 0.0
    occluded_area = max(rendered_area - visible_area, 0)
    return {
        "method": "software_projected_mesh_triangles_with_front_occluder_masks",
        "status": "accepted" if visible_area > 0 else "empty_visible_silhouette",
        "mask_path": render_info.get("mask_path"),
        "occluder_detection_ids": sorted(occluder_ids),
        "rendered_face_count": int(render_info.get("rendered_face_count") or 0),
        "target_area_px": target_area,
        "rendered_area_px": rendered_area,
        "visible_area_px": visible_area,
        "occluded_area_px": occluded_area,
        "occluded_area_ratio": float(occluded_area / max(rendered_area, 1)),
        "intersection_area_px": intersection,
        "union_area_px": union,
        "iou": iou,
        "loss": float(1.0 - iou),
        "false_positive_area_ratio": float(false_positive / max(visible_area, 1)),
        "false_negative_area_ratio": float(false_negative / max(target_area, 1)),
    }


def unavailable_silhouette_visibility_report(reason: str) -> dict[str, Any]:
    return {
        "method": "software_projected_mesh_triangles_with_front_occluder_masks",
        "status": "unavailable",
        "reason": reason,
        "mask_path": None,
        "occluder_detection_ids": [],
        "rendered_face_count": 0,
        "target_area_px": None,
        "rendered_area_px": None,
        "visible_area_px": None,
        "occluded_area_px": None,
        "occluded_area_ratio": None,
        "intersection_area_px": None,
        "union_area_px": None,
        "iou": None,
        "loss": None,
        "false_positive_area_ratio": None,
        "false_negative_area_ratio": None,
    }


def ensure_quality(item: dict[str, Any]) -> dict[str, Any]:
    quality = item.get("quality")
    if not isinstance(quality, dict):
        quality = {}
        item["quality"] = quality
    return quality


def build_table_support_candidates(
    placements: list[dict[str, Any]],
    *,
    object_dirs: dict[int, Path],
    object_mesh_name: str,
    include_review: bool,
    placement_orientation: str,
    object_scale_factor: float,
    floor_y_gltf: float | None,
) -> list[dict[str, Any]]:
    if floor_y_gltf is None:
        return []
    candidates: list[dict[str, Any]] = []
    for placement in placements:
        if not placement_is_composable(placement, include_review=include_review):
            continue
        label = str(placement.get("detector_label") or "")
        if not is_table_support_label(label):
            continue
        detection_id = int(placement.get("detection_id", 0))
        object_dir = object_dirs.get(int(placement.get("source_object_dir_id") or detection_id))
        mesh_path = resolve_object_mesh_path(object_dir, object_mesh_name) if object_dir else None
        if mesh_path is None:
            continue
        try:
            source_bounds = combined_bounds(load_meshes(mesh_path))
            transform = placement_transform_to_gltf(
                placement,
                placement_orientation=placement_orientation,
                object_scale_factor=object_scale_factor,
            )
            transform, _delta = snap_transform_to_support_bounds(source_bounds, transform, floor_y_gltf)
            bounds = transformed_bounds_from_source_bounds(source_bounds, transform)
        except Exception:
            continue
        candidates.append(
            {
                "detection_id": detection_id,
                "detector_label": placement.get("detector_label"),
                "bbox_xyxy_px": placement.get("bbox_xyxy"),
                "support_y_gltf": float(bounds[1, 1]),
                "support_bounds_gltf": bounds.tolist(),
                "support_plane_id": f"object_top_{detection_id:02d}_{slugify(label)}",
                "support_label": placement.get("detector_label"),
            }
        )
    return candidates


def best_table_candidate_for_object(bbox_xyxy: Any, table_candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    bbox = bbox_array(bbox_xyxy)
    if bbox is None:
        return None
    best: dict[str, Any] | None = None
    best_score = 0.0
    for candidate in table_candidates:
        table_bbox = bbox_array(candidate.get("bbox_xyxy_px"))
        if table_bbox is None:
            continue
        score = table_support_score(bbox, table_bbox)
        if score > best_score:
            best = dict(candidate)
            best["support_confidence"] = float(score)
            best_score = score
    return best


def table_support_score(bbox: np.ndarray, table_bbox: np.ndarray) -> float:
    object_area = max(float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])), 1.0)
    overlap_ratio = bbox_overlap_area(bbox, table_bbox) / object_area
    if overlap_ratio > 0.0:
        return float(min(0.95, 0.58 + overlap_ratio * 0.35))
    center_x = float((bbox[0] + bbox[2]) / 2.0)
    horizontal_inside = float(table_bbox[0]) <= center_x <= float(table_bbox[2])
    vertically_near = float(bbox[3]) >= float(table_bbox[1]) - max(24.0, bbox_height(bbox) * 0.35)
    return 0.66 if horizontal_inside and vertically_near else 0.0


def optimize_unknown_5dof_transform(
    *,
    source_bounds: np.ndarray,
    placement: dict[str, Any],
    transform: np.ndarray,
    coordinate_contract: dict[str, Any] | None,
    enabled: bool,
) -> dict[str, Any]:
    from SceneComposition.composer import bbox_projection_loss, projected_transform_bbox

    target_bbox = bbox_array(placement.get("bbox_xyxy"))
    initial_bbox = projected_transform_bbox(source_bounds, transform, coordinate_contract)
    base_report = {
        "enabled": bool(enabled),
        "method": "unknown_support_5dof_discrete_render_proxy_v1",
        "target_bbox_xyxy": target_bbox.tolist() if target_bbox is not None else None,
        "initial_projected_bbox_xyxy": initial_bbox.tolist() if initial_bbox is not None else None,
        "optimized_projected_bbox_xyxy": initial_bbox.tolist() if initial_bbox is not None else None,
        "candidate_projected_bbox_xyxy": None,
        "initial_loss": None,
        "optimized_loss": None,
        "candidate_loss": None,
        "delta_gltf": [0.0, 0.0, 0.0],
        "yaw_delta_radians": 0.0,
        "uniform_scale_delta": 1.0,
        "candidate_count": 0,
        "orientation_search": {
            "yaw_candidates": [],
            "selected_yaw": None,
            "loss_breakdown": {},
            "fallback_reason": "search_not_run",
        },
        "projection_quality": projection_quality_report(initial_bbox, target_bbox, accepted=True),
    }
    if not enabled or target_bbox is None or initial_bbox is None:
        return {"transform": transform, "report": base_report}

    initial_loss = bbox_projection_loss(initial_bbox, target_bbox)
    best_transform = np.asarray(transform, dtype=np.float64)
    best_bbox = initial_bbox
    best_loss = initial_loss
    best_delta = np.zeros(3, dtype=np.float64)
    best_yaw = 0.0
    best_scale = 1.0
    candidate_count = 0
    for dx in (-0.10, -0.05, 0.0, 0.05, 0.10):
        for dy in (-0.08, -0.04, 0.0, 0.04, 0.08):
            for dz in (-0.10, -0.05, 0.0, 0.05, 0.10):
                for yaw in (-0.35, 0.0, 0.35):
                    for scale in (0.85, 1.0, 1.15):
                        candidate_count += 1
                        candidate = candidate_5dof_transform(
                            transform=transform,
                            delta=np.array([dx, dy, dz], dtype=np.float64),
                            yaw=yaw,
                            scale=scale,
                        )
                        projected = projected_transform_bbox(source_bounds, candidate, coordinate_contract)
                        if projected is None:
                            continue
                        loss = bbox_projection_loss(projected, target_bbox) + scale_prior_loss(scale) + 0.12
                        if loss < best_loss:
                            best_loss = loss
                            best_transform = candidate
                            best_bbox = projected
                            best_delta = np.array([dx, dy, dz], dtype=np.float64)
                            best_yaw = yaw
                            best_scale = scale
    quality = projection_quality_report(best_bbox, target_bbox)
    base_report.update(
        initial_loss=float(initial_loss),
        optimized_loss=float(best_loss),
        candidate_loss=float(best_loss),
        optimized_projected_bbox_xyxy=best_bbox.tolist(),
        candidate_projected_bbox_xyxy=best_bbox.tolist(),
        delta_gltf=[float(value) for value in best_delta],
        yaw_delta_radians=float(best_yaw),
        uniform_scale_delta=float(best_scale),
        candidate_count=int(candidate_count),
        orientation_search={
            "yaw_candidates": [-0.35, 0.0, 0.35],
            "selected_yaw": float(best_yaw),
            "loss_breakdown": {
                "initial_total": float(initial_loss),
                "optimized_total": float(best_loss),
                "bbox_projection": float(best_loss),
                "support_contact": 0.0,
                "scale_prior": float(scale_prior_loss(best_scale)),
                "vggt_points": None,
                "candidate_count": int(candidate_count),
                "accepted_candidate_count": int(candidate_count),
            },
            "fallback_reason": None,
        },
        projection_quality=quality,
    )
    return {"transform": best_transform, "report": base_report}


def candidate_5dof_transform(*, transform: np.ndarray, delta: np.ndarray, yaw: float, scale: float) -> np.ndarray:
    from SceneComposition.composer import yaw_rotation_gltf

    adjusted = np.asarray(transform, dtype=np.float64).copy()
    adjusted[:3, :3] = yaw_rotation_gltf(float(yaw)) @ (adjusted[:3, :3] * float(scale))
    adjusted[:3, 3] = adjusted[:3, 3] + np.asarray(delta, dtype=np.float64)
    return adjusted


def scale_prior_loss(scale: float) -> float:
    if scale <= 0:
        return 10.0
    return abs(float(np.log(scale))) * 0.08


def plane_support(
    *,
    mode: str,
    support_kind: str,
    plane: dict[str, Any],
    support_y_gltf: float | None,
    confidence: float,
    reason: str,
) -> dict[str, Any]:
    support_bounds = plane_bounds_gltf(plane)
    if support_kind == "floor":
        support_bounds = expanded_structural_floor_bounds(plane, support_bounds)
    return {
        "mode": mode,
        "support_kind": support_kind,
        "support_plane_id": plane.get("id"),
        "support_detection_id": None,
        "support_label": plane.get("plane_subtype") or support_kind,
        "support_confidence": float(confidence),
        "support_y_gltf": float(support_y_gltf) if support_y_gltf is not None else None,
        "support_bounds_gltf": support_bounds,
        "plane_scene": plane_scene_summary(plane),
        "reason": reason,
    }


def expanded_structural_floor_bounds(plane: dict[str, Any], bounds: list[list[float]] | None) -> list[list[float]] | None:
    if bounds is None or int(plane.get("support_count") or 0) < STRUCTURAL_FLOOR_SUPPORT_MIN_POINTS:
        return bounds
    array = support_bounds_array(bounds)
    if array is None:
        return bounds
    extent = array[1] - array[0]
    margin_x = max(float(extent[0]) * STRUCTURAL_FLOOR_SUPPORT_BOUNDS_MARGIN_RATIO, 0.25)
    margin_z = max(float(extent[2]) * STRUCTURAL_FLOOR_SUPPORT_BOUNDS_MARGIN_RATIO, 0.25)
    expanded = array.copy()
    expanded[0, 0] -= margin_x
    expanded[1, 0] += margin_x
    expanded[0, 2] -= margin_z
    expanded[1, 2] += margin_z
    return expanded.tolist()


def tabletop_support(candidate: dict[str, Any]) -> dict[str, Any]:
    confidence = float(candidate.get("support_confidence") or 0.66)
    return {
        "mode": "tabletop_4dof",
        "support_kind": "tabletop",
        "support_plane_id": candidate.get("support_plane_id"),
        "support_detection_id": int(candidate["detection_id"]),
        "support_label": candidate.get("support_label"),
        "support_confidence": confidence,
        "support_y_gltf": float(candidate["support_y_gltf"]),
        "support_bounds_gltf": candidate.get("support_bounds_gltf"),
        "reason": "tabletop_label_with_2d_support_overlap",
    }


def unknown_support() -> dict[str, Any]:
    return {
        "mode": "unknown_5dof",
        "support_kind": "unknown",
        "support_plane_id": None,
        "support_detection_id": None,
        "support_label": None,
        "support_confidence": 0.0,
        "support_y_gltf": None,
        "reason": UNKNOWN_SUPPORT_REVIEW_REASON,
    }


def degrees_of_freedom_for_support(support: dict[str, Any]) -> dict[str, Any]:
    mode = str(support.get("mode") or "unknown_5dof")
    if mode.endswith("_4dof"):
        return {
            "model": "support_plane_4dof",
            "free_parameters": ["plane_u", "plane_v", "yaw_normal", "uniform_scale"],
            "locked_parameters": ["plane_normal_distance"],
        }
    return {
        "model": "unknown_support_5dof",
        "free_parameters": ["tx", "ty", "tz", "yaw_up", "uniform_scale"],
        "locked_parameters": [],
    }


def placement_record_from_target(target: dict[str, Any]) -> dict[str, Any]:
    geometry = target.get("placement_geometry") or {}
    return {
        "detection_id": target.get("detection_id"),
        "detector_label": target.get("detector_label"),
        "bbox_xyxy": target.get("bbox_xyxy_px"),
        "mask_path": target.get("mask_path"),
        "box_type": geometry.get("box_type"),
        "center_xyz": geometry.get("center_xyz"),
        "extent_xyz": geometry.get("extent_xyz"),
        "rotation_matrix": geometry.get("rotation_matrix"),
        "needs_review": target.get("needs_review"),
        "visible_points_scene_path": target.get("visible_points_scene_path"),
    }


def support_contact_distance(source_bounds: np.ndarray, transform: np.ndarray, support_y: Any) -> float:
    if support_y is None:
        return 0.0
    return support_penalty(source_bounds, transform, float(support_y))


def support_contact_threshold(transformed_bounds: np.ndarray) -> float:
    height = max(float(transformed_bounds[1, 1] - transformed_bounds[0, 1]), 1e-6)
    return max(height * SUPPORT_CONTACT_THRESHOLD_RATIO, 1e-5)


def support_quality_status(
    contact_loss: float,
    transformed_bounds: np.ndarray,
    footprint: dict[str, Any],
    penetration: dict[str, Any],
    support_y: Any,
) -> str:
    if support_y is None:
        return "accepted"
    if contact_loss > support_contact_threshold(transformed_bounds):
        return "rejected"
    if float(footprint.get("outside_ratio") or 0.0) > SUPPORT_FOOTPRINT_REJECT_RATIO:
        return "rejected"
    if penetration.get("penetrates_support"):
        return "rejected"
    return "accepted"


def support_footprint_report(transformed_bounds: np.ndarray, support: dict[str, Any]) -> dict[str, Any]:
    support_bounds = support_bounds_array(support.get("support_bounds_gltf"))
    mode = str(support.get("mode") or "")
    if support_bounds is None or not mode.endswith("_4dof"):
        return {
            "status": "unavailable",
            "outside_ratio": None,
            "footprint_area_gltf": None,
            "supported_area_gltf": None,
            "warning_threshold": SUPPORT_FOOTPRINT_WARNING_RATIO,
            "reject_threshold": SUPPORT_FOOTPRINT_REJECT_RATIO,
        }
    footprint = np.asarray(
        [
            [float(transformed_bounds[0, 0]), float(transformed_bounds[0, 2])],
            [float(transformed_bounds[1, 0]), float(transformed_bounds[1, 2])],
        ],
        dtype=np.float64,
    )
    support_rect = np.asarray(
        [
            [float(support_bounds[0, 0]), float(support_bounds[0, 2])],
            [float(support_bounds[1, 0]), float(support_bounds[1, 2])],
        ],
        dtype=np.float64,
    )
    footprint_area = rect_area_2d(footprint)
    overlap_area = rect_overlap_area_2d(footprint, support_rect)
    outside_ratio = 1.0 - (overlap_area / footprint_area) if footprint_area > 1e-8 else 1.0
    if outside_ratio > SUPPORT_FOOTPRINT_REJECT_RATIO:
        status = "rejected"
    elif outside_ratio > SUPPORT_FOOTPRINT_WARNING_RATIO:
        status = "warning"
    else:
        status = "accepted"
    return {
        "status": status,
        "outside_ratio": float(outside_ratio),
        "footprint_area_gltf": float(footprint_area),
        "supported_area_gltf": float(overlap_area),
        "warning_threshold": SUPPORT_FOOTPRINT_WARNING_RATIO,
        "reject_threshold": SUPPORT_FOOTPRINT_REJECT_RATIO,
    }


def support_penetration_report(transformed_bounds: np.ndarray, support: dict[str, Any]) -> dict[str, Any]:
    support_y = support.get("support_y_gltf")
    if support_y is None:
        return {
            "status": "not_evaluated",
            "penetrates_support": False,
            "penetration_depth_gltf": 0.0,
        }
    penetration = max(0.0, float(support_y) - float(transformed_bounds[0, 1]))
    return {
        "status": "rejected" if penetration > BACKGROUND_PENETRATION_EPSILON else "accepted",
        "penetrates_support": bool(penetration > BACKGROUND_PENETRATION_EPSILON),
        "penetration_depth_gltf": float(penetration),
        "epsilon": BACKGROUND_PENETRATION_EPSILON,
    }


def silhouette_proxy_report(optimization_report: dict[str, Any]) -> dict[str, Any]:
    target = bbox_array(optimization_report.get("target_bbox_xyxy"))
    projected = bbox_array(optimization_report.get("optimized_projected_bbox_xyxy"))
    if target is None or projected is None:
        return {
            "method": "bbox_projection_proxy",
            "status": "unavailable",
            "iou": None,
            "loss": None,
            "false_positive_area_ratio": None,
            "false_negative_area_ratio": None,
        }
    intersection = bbox_overlap_area(projected, target)
    projected_area = max(float((projected[2] - projected[0]) * (projected[3] - projected[1])), 1.0)
    target_area = max(float((target[2] - target[0]) * (target[3] - target[1])), 1.0)
    iou = bbox_iou(projected, target)
    return {
        "method": "bbox_projection_proxy",
        "status": "accepted",
        "iou": float(iou),
        "loss": float(1.0 - iou),
        "false_positive_area_ratio": float(max(0.0, projected_area - intersection) / projected_area),
        "false_negative_area_ratio": float(max(0.0, target_area - intersection) / target_area),
    }


def silhouette_render_report(
    *,
    target: dict[str, Any],
    mesh_path: Path,
    source_bounds: np.ndarray,
    transform: np.ndarray,
    coordinate_contract: dict[str, Any] | None,
) -> dict[str, Any]:
    mask_path_value = target.get("mask_path")
    if not mask_path_value:
        return unavailable_silhouette_render_report("missing_mask_path")
    mask_path = Path(str(mask_path_value))
    if not mask_path.is_file():
        return unavailable_silhouette_render_report("missing_mask_file")
    contract = coordinate_contract or {}
    width = int(contract.get("image_width") or 0)
    height = int(contract.get("image_height") or 0)
    if width <= 0 or height <= 0:
        return unavailable_silhouette_render_report("missing_image_size")
    try:
        mask_image = Image.open(mask_path).convert("L")
    except Exception:
        return unavailable_silhouette_render_report("invalid_mask_file")
    if mask_image.size != (width, height):
        mask_image = mask_image.resize((width, height), Image.Resampling.NEAREST)
    target_mask = np.asarray(mask_image, dtype=np.uint8) > 127
    if not bool(target_mask.any()):
        return unavailable_silhouette_render_report("empty_mask")
    try:
        rendered_mask, face_count = render_mesh_silhouette_mask(
            mesh_path=mesh_path,
            source_bounds=source_bounds,
            transform=transform,
            coordinate_contract=contract,
            width=width,
            height=height,
        )
    except Exception:
        return unavailable_silhouette_render_report("render_failed")
    rendered = np.asarray(rendered_mask, dtype=np.uint8) > 0
    rendered_area = int(rendered.sum())
    if rendered_area == 0:
        return unavailable_silhouette_render_report("empty_rendered_silhouette")
    intersection = int(np.logical_and(rendered, target_mask).sum())
    union = int(np.logical_or(rendered, target_mask).sum())
    target_area = int(target_mask.sum())
    false_positive = int(np.logical_and(rendered, ~target_mask).sum())
    false_negative = int(np.logical_and(~rendered, target_mask).sum())
    iou = float(intersection / union) if union else 0.0
    return {
        "method": "software_projected_mesh_triangles",
        "status": "accepted",
        "mask_path": str(mask_path),
        "rendered_face_count": int(face_count),
        "target_area_px": target_area,
        "rendered_area_px": rendered_area,
        "intersection_area_px": intersection,
        "union_area_px": union,
        "iou": iou,
        "loss": float(1.0 - iou),
        "false_positive_area_ratio": float(false_positive / max(rendered_area, 1)),
        "false_negative_area_ratio": float(false_negative / max(target_area, 1)),
    }


def unavailable_silhouette_render_report(reason: str) -> dict[str, Any]:
    return {
        "method": "software_projected_mesh_triangles",
        "status": "unavailable",
        "reason": reason,
        "mask_path": None,
        "rendered_face_count": 0,
        "target_area_px": None,
        "rendered_area_px": None,
        "intersection_area_px": None,
        "union_area_px": None,
        "iou": None,
        "loss": None,
        "false_positive_area_ratio": None,
        "false_negative_area_ratio": None,
    }


def render_mesh_silhouette_mask(
    *,
    mesh_path: Path,
    source_bounds: np.ndarray,
    transform: np.ndarray,
    coordinate_contract: dict[str, Any],
    width: int,
    height: int,
) -> tuple[Image.Image, int]:
    asset_transform = np.asarray(transform, dtype=np.float64) @ normalization_transform(source_bounds)
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    face_count = 0
    for mesh in load_meshes(mesh_path):
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if len(vertices) == 0 or len(faces) == 0:
            continue
        projected, valid = project_gltf_vertices_to_pixels(transform_points(vertices, asset_transform), coordinate_contract, width, height)
        for face in sample_rows(faces, SILHOUETTE_FACE_SAMPLE_COUNT):
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
            if abs(triangle_area_2d(points)) < 0.25:
                continue
            draw.polygon([(float(x), float(y)) for x, y in points], fill=255)
            face_count += 1
    return image, face_count


def project_gltf_vertices_to_pixels(
    points: np.ndarray,
    coordinate_contract: dict[str, Any],
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    fov = float(coordinate_contract.get("fov_degrees", DEFAULT_FOV_DEGREES))
    focal = (width / 2.0) / np.tan(np.deg2rad(fov) / 2.0)
    x = points[:, 0]
    scene_z = points[:, 1]
    depth = -points[:, 2]
    valid = (depth > 1e-6) & np.isfinite(points).all(axis=1)
    pixels = np.zeros((len(points), 2), dtype=np.float64)
    safe_depth = np.where(valid, depth, 1.0)
    pixels[:, 0] = width / 2.0 + (x / safe_depth) * focal
    pixels[:, 1] = height / 2.0 - (scene_z / safe_depth) * focal
    return pixels, valid


def triangle_area_2d(points: np.ndarray) -> float:
    return float(
        0.5
        * (
            (points[1, 0] - points[0, 0]) * (points[2, 1] - points[0, 1])
            - (points[2, 0] - points[0, 0]) * (points[1, 1] - points[0, 1])
        )
    )


def vggt_point_loss_report(
    *,
    target: dict[str, Any],
    mesh_path: Path,
    source_bounds: np.ndarray,
    transform: np.ndarray,
) -> dict[str, Any]:
    points_path_value = target.get("visible_points_scene_path")
    if not points_path_value:
        return unavailable_vggt_point_report("missing_visible_points")
    points_path = Path(str(points_path_value))
    if not points_path.is_file():
        return unavailable_vggt_point_report("missing_visible_points_file")
    try:
        visible_points = np.load(points_path)
    except Exception:
        return unavailable_vggt_point_report("invalid_visible_points_file")
    visible_points = np.asarray(visible_points, dtype=np.float64)
    if visible_points.ndim != 2 or visible_points.shape[1] != 3:
        return unavailable_vggt_point_report("invalid_visible_points_shape")
    visible_points = visible_points[np.isfinite(visible_points).all(axis=1)]
    if len(visible_points) == 0:
        return unavailable_vggt_point_report("empty_visible_points")
    try:
        meshes = load_meshes(mesh_path)
        vertices = np.concatenate([np.asarray(mesh.vertices, dtype=np.float64) for mesh in meshes if len(mesh.vertices) > 0], axis=0)
    except Exception:
        return unavailable_vggt_point_report("invalid_mesh_vertices")
    if len(vertices) == 0:
        return unavailable_vggt_point_report("empty_mesh_vertices")

    asset_transform = np.asarray(transform, dtype=np.float64) @ normalization_transform(source_bounds)
    transformed_vertices = transform_points(sample_rows(vertices, VGGT_POINT_SAMPLE_COUNT), asset_transform)
    visible_gltf = scene_points_to_gltf(sample_rows(visible_points, VGGT_POINT_SAMPLE_COUNT))
    distances = nearest_distances(visible_gltf, transformed_vertices)
    if len(distances) == 0:
        return unavailable_vggt_point_report("empty_distance_samples")
    median = float(np.median(distances))
    p90 = float(np.percentile(distances, 90.0))
    return {
        "method": "sampled_visible_vggt_points_to_mesh_vertices",
        "status": "accepted",
        "visible_points_scene_path": str(points_path),
        "visible_point_sample_count": int(len(visible_gltf)),
        "mesh_vertex_sample_count": int(len(transformed_vertices)),
        "median_distance_gltf": median,
        "p90_distance_gltf": p90,
        "loss": median,
    }


def unavailable_vggt_point_report(reason: str) -> dict[str, Any]:
    return {
        "method": "sampled_visible_vggt_points_to_mesh_vertices",
        "status": "unavailable",
        "reason": reason,
        "visible_points_scene_path": None,
        "visible_point_sample_count": 0,
        "mesh_vertex_sample_count": 0,
        "median_distance_gltf": None,
        "p90_distance_gltf": None,
        "loss": None,
    }


def sample_rows(values: np.ndarray, max_count: int) -> np.ndarray:
    if len(values) <= max_count:
        return np.asarray(values, dtype=np.float64)
    indices = np.linspace(0, len(values) - 1, max_count, dtype=np.int64)
    return np.asarray(values[indices], dtype=np.float64)


def scene_points_to_gltf(points: np.ndarray) -> np.ndarray:
    return np.asarray([[x, z, -y] for x, y, z in points], dtype=np.float64)


def nearest_distances(points: np.ndarray, samples: np.ndarray, *, chunk_size: int = 256) -> np.ndarray:
    distances: list[np.ndarray] = []
    for start in range(0, len(points), chunk_size):
        chunk = points[start : start + chunk_size]
        diff = chunk[:, None, :] - samples[None, :, :]
        distances.append(np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1)))
    if not distances:
        return np.empty((0,), dtype=np.float64)
    return np.concatenate(distances)


def placement_losses(
    *,
    bbox_loss: Any,
    silhouette_loss: Any,
    vggt_point_loss: Any,
    support_contact: float,
    background_collision: Any,
    scale_delta: Any,
    unknown_support: bool,
) -> dict[str, Any]:
    bbox_value = float(bbox_loss) if bbox_loss is not None else None
    silhouette_value = float(silhouette_loss) if silhouette_loss is not None else None
    vggt_value = float(vggt_point_loss) if vggt_point_loss is not None else None
    collision_value = float(background_collision) if background_collision is not None else None
    scale_value = abs(float(np.log(float(scale_delta)))) if scale_delta not in (None, 0) else 0.0
    unknown_penalty = 0.25 if unknown_support else 0.0
    total = sum(value for value in (bbox_value, silhouette_value, vggt_value, support_contact, collision_value, scale_value, unknown_penalty) if value is not None)
    return {
        "total": float(total),
        "bbox_projection": bbox_value,
        "silhouette": silhouette_value,
        "vggt_points": vggt_value,
        "support_contact": float(support_contact),
        "background_collision": collision_value,
        "scale_prior": float(scale_value),
        "unknown_support_prior": float(unknown_penalty),
    }


def empty_losses() -> dict[str, Any]:
    return {
        "total": None,
        "bbox_projection": None,
        "silhouette": None,
        "vggt_points": None,
        "support_contact": None,
        "background_collision": None,
        "scale_prior": None,
        "unknown_support_prior": None,
    }


def placement_review_reason(mode: str, projection_quality: dict[str, Any], support_status: str) -> str:
    if mode == "unknown_5dof":
        return UNKNOWN_SUPPORT_REVIEW_REASON
    if projection_quality.get("status") == "accepted_occluded_bottom":
        return str(projection_quality.get("reason") or "occluded_bottom_edge_tolerated")
    if projection_quality.get("status") == "rejected":
        return str(projection_quality.get("reason") or "projection_rejected")
    if support_status == "rejected":
        return "support_contact_rejected"
    return "review_required_by_evidence"


def placement_warnings(
    target: dict[str, Any],
    mode: str,
    projection_quality: dict[str, Any],
    support_status: str,
    collision_status: str,
    footprint: dict[str, Any],
    penetration: dict[str, Any],
) -> list[str]:
    warnings = list(target.get("warnings") or [])
    if mode == "unknown_5dof":
        warnings.append(UNKNOWN_SUPPORT_REVIEW_REASON)
    if projection_quality.get("status") == "accepted_occluded_bottom":
        warnings.append(str(projection_quality.get("reason") or "occluded_bottom_edge_tolerated"))
    if projection_quality.get("status") == "rejected":
        warnings.append(str(projection_quality.get("reason") or "projection_rejected"))
    if support_status == "rejected":
        warnings.append("support_contact_rejected")
    if collision_status == "rejected":
        warnings.append("support_penetration")
    if footprint.get("status") == "warning":
        warnings.append("support_footprint_warning")
    if footprint.get("status") == "rejected":
        warnings.append("support_footprint_rejected")
    if penetration.get("penetrates_support"):
        warnings.append("support_penetration")
    return sorted(set(warnings))


def annotate_pairwise_collision_metrics(objects: list[dict[str, Any]]) -> None:
    composed = [item for item in objects if item.get("status") == "accepted" and bounds_array(item.get("transformed_bounds")) is not None]
    overlap_warnings: dict[int, list[dict[str, Any]]] = {int(item["detection_id"]): [] for item in composed}
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
            warning = {
                "detection_ids": [left.get("detection_id"), right.get("detection_id")],
                "labels": [left.get("detector_label"), right.get("detector_label")],
                "overlap_extent_gltf": [float(value) for value in overlap_extent],
                "overlap_volume_gltf": overlap_volume,
                "tabletop_pair": bool(
                    is_tabletop_object_label(str(left.get("detector_label") or ""))
                    or is_tabletop_object_label(str(right.get("detector_label") or ""))
                ),
            }
            overlap_warnings[int(left["detection_id"])].append(warning)
            overlap_warnings[int(right["detection_id"])].append(warning)
    for item in objects:
        warnings = overlap_warnings.get(int(item.get("detection_id", 0)), [])
        quality = item.get("quality") or {}
        quality["object_overlap_warnings"] = warnings
        if warnings:
            quality["warnings"] = sorted(set(list(quality.get("warnings") or []) + ["object_aabb_overlap"]))
            if quality.get("collision_status") == "accepted":
                quality["collision_status"] = "warning"
            losses = item.get("losses") or {}
            losses["object_overlap"] = float(sum(warning["overlap_volume_gltf"] for warning in warnings))
            item["losses"] = losses
        item["quality"] = quality


def rect_area_2d(rect: np.ndarray) -> float:
    return max(0.0, float(rect[1, 0] - rect[0, 0])) * max(0.0, float(rect[1, 1] - rect[0, 1]))


def rect_overlap_area_2d(left: np.ndarray, right: np.ndarray) -> float:
    width = max(0.0, float(min(left[1, 0], right[1, 0]) - max(left[0, 0], right[0, 0])))
    height = max(0.0, float(min(left[1, 1], right[1, 1]) - max(left[0, 1], right[0, 1])))
    return width * height


def support_bounds_array(value: Any) -> np.ndarray | None:
    try:
        bounds = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        return None
    if np.any(bounds[1] < bounds[0]):
        return None
    return bounds


def matrix_array(value: Any) -> np.ndarray | None:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        return None
    return matrix


def write_visible_points_npy(placement: dict[str, Any], detection_id: int, label: str, output_dir: Path) -> Path | None:
    artifacts = placement.get("artifacts") or {}
    points_xyz = artifacts.get("points_xyz")
    if not points_xyz:
        return None
    source = Path(str(points_xyz))
    if not source.is_file():
        return None
    target = output_dir / f"{detection_id:02d}_{slugify(label)}_points.npy"
    try:
        points = np.loadtxt(source, dtype=np.float32)
    except Exception:
        return None
    points = np.asarray(points, dtype=np.float32)
    if points.ndim == 1 and points.shape[0] == 3:
        points = points.reshape(1, 3)
    if points.ndim != 2 or points.shape[1] != 3:
        return None
    np.save(target, points)
    return target


def mesh_bounds_for_path(mesh_path: Path | None) -> np.ndarray | None:
    if mesh_path is None or not mesh_path.is_file():
        return None
    try:
        return combined_bounds(load_meshes(mesh_path))
    except Exception:
        return None


def mesh_quality_report(mesh_path: Path | None, mesh_bounds: np.ndarray | None) -> dict[str, Any]:
    if mesh_path is None:
        return {
            "has_texture": False,
            "has_large_support_sheet": None,
            "bounds_degenerate": True,
        }
    bounds_degenerate = True
    if mesh_bounds is not None:
        bounds_degenerate = bool(np.any((mesh_bounds[1] - mesh_bounds[0]) <= 1e-8))
    return {
        "has_texture": "textured" in mesh_path.name.lower() or mesh_path.suffix.lower() == ".glb",
        "has_large_support_sheet": None,
        "bounds_degenerate": bounds_degenerate,
    }


def first_plane_with_subtype(report: dict[str, Any], subtype: str) -> dict[str, Any] | None:
    for plane in report.get("planes", []):
        if str(plane.get("plane_subtype") or "").lower() == subtype:
            return plane
    return None


def plane_support_y_gltf(plane: dict[str, Any] | None) -> float | None:
    if plane is None:
        return None
    vertices = np.asarray(plane.get("vertices_xyz"), dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        return None
    # SceneForge scene Z maps to GLB Y.
    return float(np.mean(vertices[:, 2]))


def plane_bounds_gltf(plane: dict[str, Any]) -> list[list[float]] | None:
    try:
        vertices = np.asarray([scene_point_to_gltf_vertex(vertex) for vertex in plane.get("vertices_xyz", [])], dtype=np.float64)
    except Exception:
        return None
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all() or len(vertices) == 0:
        return None
    return np.stack([vertices.min(axis=0), vertices.max(axis=0)], axis=0).tolist()


def plane_scene_summary(plane: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": plane.get("id"),
        "plane_subtype": plane.get("plane_subtype"),
        "normal_xyz": plane.get("normal_xyz"),
        "vertices_xyz": plane.get("vertices_xyz"),
        "support_count": plane.get("support_count"),
        "fit_residual": plane.get("fit_residual"),
    }


def is_floor_object_label(label: str) -> bool:
    normalized = label.lower()
    return any(token in normalized for token in FLOOR_SUPPORT_LABELS)


def should_use_floor_support(label: str) -> bool:
    normalized = label.lower()
    if any(token in normalized for token in TABLETOP_OBJECT_LABELS) and not is_floor_object_label(normalized):
        return False
    return is_floor_object_label(normalized) or not is_tabletop_object_label(normalized)


def is_wall_object_label(label: str) -> bool:
    normalized = label.lower()
    return any(token in normalized for token in WALL_OBJECT_LABELS)


def support_summary(objects: list[dict[str, Any]]) -> dict[str, Any]:
    modes: dict[str, int] = {}
    for item in objects:
        mode = str((item.get("support") or {}).get("mode") or "none")
        modes[mode] = modes.get(mode, 0) + 1
    return {
        "object_count": len(objects),
        "accepted_count": sum(1 for item in objects if item["status"] == "accepted"),
        "skipped_count": sum(1 for item in objects if item["status"] == "skipped"),
        "needs_review_count": sum(1 for item in objects if item["needs_review"]),
        "support_modes": modes,
    }


def fit_target_summary(objects: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "object_count": len(objects),
        "ready_count": sum(1 for item in objects if item["status"] == "ready"),
        "failed_count": sum(1 for item in objects if item["status"] == "failed"),
        "needs_review_count": sum(1 for item in objects if item["needs_review"]),
    }


def placement_quality_report(objects: list[dict[str, Any]]) -> dict[str, Any]:
    projection_statuses = status_counts((item.get("quality") or {}).get("projection_status") for item in objects)
    support_statuses = status_counts((item.get("quality") or {}).get("support_status") for item in objects)
    collision_statuses = status_counts((item.get("quality") or {}).get("collision_status") for item in objects)
    silhouette_statuses = status_counts(((item.get("quality") or {}).get("silhouette_render") or {}).get("status") for item in objects)
    silhouette_visibility_statuses = status_counts(
        ((item.get("quality") or {}).get("silhouette_visibility") or {}).get("status") for item in objects
    )
    vggt_point_statuses = status_counts(((item.get("quality") or {}).get("vggt_points") or {}).get("status") for item in objects)
    support_footprint_statuses = status_counts(((item.get("quality") or {}).get("support_footprint") or {}).get("status") for item in objects)
    rejected_projection = [
        {
            "detection_id": item.get("detection_id"),
            "detector_label": item.get("detector_label"),
            "reason": ((item.get("render_to_input_optimization") or {}).get("projection_quality") or {}).get("reason")
            or (item.get("quality") or {}).get("projection_status"),
        }
        for item in objects
        if (item.get("quality") or {}).get("projection_status") == "rejected"
    ]
    unknown_support = [
        {
            "detection_id": item.get("detection_id"),
            "detector_label": item.get("detector_label"),
        }
        for item in objects
        if ((item.get("support") or {}).get("mode") == "unknown_5dof")
    ]
    needs_review = [
        {
            "detection_id": item.get("detection_id"),
            "detector_label": item.get("detector_label"),
            "reason": item.get("reason"),
            "warnings": (item.get("quality") or {}).get("warnings") or [],
        }
        for item in objects
        if item.get("needs_review")
    ]
    overlap_warnings = unique_overlap_warnings(
        warning
        for item in objects
        for warning in ((item.get("quality") or {}).get("object_overlap_warnings") or [])
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "object_count": len(objects),
        "accepted_count": sum(1 for item in objects if item["status"] == "accepted"),
        "failed_count": sum(1 for item in objects if item["status"] == "failed"),
        "needs_review_count": sum(1 for item in objects if item["needs_review"]),
        "needs_review": needs_review,
        "status_counts": {
            "placement": status_counts(item.get("status") for item in objects),
            "projection": projection_statuses,
            "support": support_statuses,
            "collision": collision_statuses,
            "silhouette_render": silhouette_statuses,
            "silhouette_visibility": silhouette_visibility_statuses,
            "vggt_points": vggt_point_statuses,
            "support_footprint": support_footprint_statuses,
        },
        "support_modes": status_counts(((item.get("support") or {}).get("mode")) for item in objects),
        "losses": loss_summaries(objects),
        "projection_rejected_count": len(rejected_projection),
        "projection_rejected": rejected_projection,
        "projection_occluded_bottom_count": projection_statuses.get("accepted_occluded_bottom", 0),
        "unknown_support_count": len(unknown_support),
        "unknown_support": unknown_support,
        "object_overlap_warning_count": len(overlap_warnings),
        "object_overlap_warnings": overlap_warnings,
    }


def status_counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value) if value not in (None, "") else "missing"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def unique_overlap_warnings(warnings: Any) -> list[dict[str, Any]]:
    unique: dict[tuple[tuple[int, ...], float], dict[str, Any]] = {}
    for warning in warnings:
        detection_ids = tuple(sorted(int(value) for value in warning.get("detection_ids", []) if value is not None))
        volume = round(float(warning.get("overlap_volume_gltf") or 0.0), 10)
        unique[(detection_ids, volume)] = warning
    return list(unique.values())


def loss_summaries(objects: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    keys = (
        "total",
        "bbox_projection",
        "silhouette",
        "vggt_points",
        "support_contact",
        "background_collision",
        "scale_prior",
        "unknown_support_prior",
        "object_overlap",
    )
    return {key: numeric_summary((item.get("losses") or {}).get(key) for item in objects) for key in keys}


def numeric_summary(values: Any) -> dict[str, Any]:
    numeric: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            numeric.append(number)
    if not numeric:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    array = np.asarray(numeric, dtype=np.float64)
    return {
        "count": int(len(array)),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "max": float(array.max()),
    }


def infer_source_image_path_from_reports(geometry: dict[str, Any], supports: dict[str, Any]) -> str | None:
    for report in (geometry, supports):
        for key in ("image_path", "source_image_path"):
            if report.get(key):
                return str(report[key])
        detections_path = report.get("detections_path")
        if detections_path and Path(str(detections_path)).is_file():
            try:
                detections = load_json(Path(str(detections_path)))
            except Exception:
                continue
            for key in ("image_path", "source_image_path"):
                if detections.get(key):
                    return str(detections[key])
    return None


def slugify(value: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "object"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
