from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from Input.Depth.depth_loader import load_grayscale_depth
from Input.Image.image_loader import load_rgb_image
from ObjectEnrichment.report_loader import load_enrichment_report
from OutputWriter.metrics_summary import write_fit_metrics_summary
from OutputWriter.overlay import write_overlay
from PrimitiveFitting.blender_exporter import export_fit_report_to_blend
from PrimitiveFitting.camera import PinholeCamera
from PrimitiveFitting.depth_check import write_depth_check
from PrimitiveFitting.depth_refiner import refine_fits_against_depth
from PrimitiveFitting.fitter import fit_primitive
from PrimitiveFitting.masks import polygon_to_mask, sampled_mask_pixels
from PrimitiveFitting.report_loader import load_detection_report
from PrimitiveFitting.report_writer import write_fit_report
from PrimitiveFitting.types import PrimitiveFit, PrimitiveFitReport
from ShapeDetection.report import ObjectShapeDetection
from ObjectEnrichment.types import FUSED_LABELS, FusedState


def fitted_scene_output_path(output_path: Path) -> Path:
    if output_path.name == "fit" and output_path.parent.name == "Latest":
        return output_path.parent / "fitted_scene.blend"
    return output_path / "fitted_scene.blend"


BOX_DIMENSION_CANDIDATE_SCALES = (
    (1.04, 1.00, 1.04),
    (1.08, 1.00, 1.08),
    (1.00, 1.05, 1.00),
    (1.04, 1.05, 1.04),
    (0.96, 1.00, 0.96),
    (1.00, 0.96, 1.00),
)


def run_primitive_fitting(
    image_path: str | Path,
    depth_path: str | Path,
    detections_path: str | Path,
    output_dir: str | Path,
    enrichment_path: str | Path | None = None,
    fov_degrees: float = 70.0,
    sensor_fit: str = "horizontal",
    camera_shift_x: float = 0.0,
    camera_shift_y: float = 0.0,
    near_depth: float = 1.0,
    far_depth: float = 6.0,
    blender_executable: str = "blender",
    reference_blend_path: str | Path | None = None,
    final_layout: str = "camera",
    depth_refinement_enabled: bool = True,
) -> PrimitiveFitReport:
    if final_layout not in {"camera", "ground", "original-camera"}:
        raise ValueError("final_layout must be 'camera', 'ground', or 'original-camera'.")
    if final_layout == "original-camera" and reference_blend_path is None:
        raise ValueError("reference_blend_path is required when final_layout is 'original-camera'.")
    resolved_image_path = Path(image_path)
    resolved_depth_path = Path(depth_path)
    resolved_detections_path = Path(detections_path)
    output_path = Path(output_dir)

    image = load_rgb_image(resolved_image_path)
    depth = load_grayscale_depth(resolved_depth_path, expected_size=image.size)
    detections = load_detection_report(resolved_detections_path)
    if detections.image_width != image.width or detections.image_height != image.height:
        raise ValueError("Detection report image dimensions do not match the input image.")

    camera = PinholeCamera(
        image_width=image.width,
        image_height=image.height,
        fov_degrees=fov_degrees,
        sensor_fit=sensor_fit,
        near_depth=near_depth,
        far_depth=far_depth,
    )
    enrichment_by_id = {}
    if enrichment_path is not None:
        enrichment = load_enrichment_report(enrichment_path)
        if detections.objects and not enrichment.objects:
            raise ValueError("Enrichment report has no objects for non-empty detections.")
        detection_ids = {item.id for item in detections.objects}
        enrichment_ids = {item.id for item in enrichment.objects}
        if detection_ids != enrichment_ids:
            raise ValueError(
                f"Detection/enrichment ids do not match: detections={sorted(detection_ids)}, enrichment={sorted(enrichment_ids)}"
            )
        enrichment_by_id = {item.id: item for item in enrichment.objects}

    fits: list[PrimitiveFit] = []
    fit_detections: list[ObjectShapeDetection] = []
    for detection in detections.objects:
        enrichment_object = enrichment_by_id.get(detection.id)
        if enrichment_object is not None:
            fused_state = _resolve_fitting_contract(enrichment_object)
            if fused_state is None:
                raise ValueError(f"Missing fused contract for object id {detection.id} in enrichment report.")
            detection = replace(
                detection,
                primitive_label=fused_state.fused_label,
                primitive_confidence=fused_state.fused_confidence,
                primitive_label_source="fused",
            )
        fit_detections.append(detection)
        mask = polygon_to_mask(detection.mask_polygon, image.width, image.height)
        pixels = sampled_mask_pixels(mask)
        if len(pixels) > 0:
            depth_values = depth[pixels[:, 1], pixels[:, 0]]
            points = camera.unproject_pixels(pixels, depth_values)
        else:
            points = camera.unproject_pixels(pixels, depth[pixels[:, 1], pixels[:, 0]])
        fit = fit_primitive(detection, points, camera=camera)
        if enrichment_object is not None:
            fit = apply_enrichment_fit_metadata(fit, enrichment_object)
        fits.append(fit)

    model_info = {
        "backend": "depth_mask_primitive_fit",
        "final_blend_layout": final_layout,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    camera_info = camera.to_dict()
    camera_info["shift_x"] = float(camera_shift_x)
    camera_info["shift_y"] = float(camera_shift_y)

    report = PrimitiveFitReport(
        image_path=str(resolved_image_path),
        depth_path=str(resolved_depth_path),
        detections_path=str(resolved_detections_path),
        image_width=image.width,
        image_height=image.height,
        camera=camera_info,
        objects=fits,
        model_info=model_info,
    )

    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / "primitive_fits.json"
    write_fit_report(report, report_path)
    write_overlay(image, fit_detections, output_path / "fit_overlay.png")
    with TemporaryDirectory() as temp_dir:
        camera_space_blend = Path(temp_dir) / "fitted_scene_camera_space.blend"
        export_fit_report_to_blend(
            report_path=report_path,
            output_path=camera_space_blend,
            blender_executable=blender_executable,
            layout="camera",
        )
        initial_depth_metrics = write_depth_check(
            source_depth_path=resolved_depth_path,
            fitted_blend_path=camera_space_blend,
            output_dir=output_path / "depth_check_initial",
            near_depth=near_depth,
            far_depth=far_depth,
            blender_executable=blender_executable,
            detections=detections.objects,
        )
        unrefined_fits = report.objects
        if depth_refinement_enabled:
            refined_fits, refinement_metrics = refine_fits_against_depth(
                fits=report.objects,
                detections=detections.objects,
                source_depth_path=resolved_depth_path,
                fitted_depth_path=initial_depth_metrics["fitted_depth_path"],
                near_depth=near_depth,
                far_depth=far_depth,
            )
            model_info["depth_refinement"] = {
                "strategy": refinement_metrics["strategy"],
                "changed_object_count": refinement_metrics["changed_object_count"],
                "initial_depth_check_dir": str(output_path / "depth_check_initial"),
            }
            report = replace(report, objects=refined_fits, model_info=model_info)
            write_fit_report(report, report_path)
            refined_camera_space_blend = Path(temp_dir) / "fitted_scene_camera_space_refined.blend"
            export_fit_report_to_blend(
                report_path=report_path,
                output_path=refined_camera_space_blend,
                blender_executable=blender_executable,
                layout="camera",
            )
            depth_metrics = write_depth_check(
                source_depth_path=resolved_depth_path,
                fitted_blend_path=refined_camera_space_blend,
                output_dir=output_path / "depth_check",
                near_depth=near_depth,
                far_depth=far_depth,
                blender_executable=blender_executable,
                detections=detections.objects,
            )
            candidate_depth_metrics = depth_metrics
            refinement_metrics["candidate_depth_metrics"] = _depth_metric_summary(candidate_depth_metrics)
            refinement_metrics["initial_depth_score"] = round(_depth_metric_score(initial_depth_metrics), 6)
            refinement_metrics["candidate_depth_score"] = round(_depth_metric_score(candidate_depth_metrics), 6)
            refinement_metrics["accepted"] = _depth_refinement_accepted(initial_depth_metrics, candidate_depth_metrics)
            if not refinement_metrics["accepted"]:
                report = replace(report, objects=unrefined_fits, model_info=model_info)
                write_fit_report(report, report_path)
                depth_metrics = write_depth_check(
                    source_depth_path=resolved_depth_path,
                    fitted_blend_path=camera_space_blend,
                    output_dir=output_path / "depth_check",
                    near_depth=near_depth,
                    far_depth=far_depth,
                    blender_executable=blender_executable,
                    detections=detections.objects,
                )
            refinement_metrics["initial_depth_metrics"] = _depth_metric_summary(initial_depth_metrics)
            refinement_metrics["final_depth_metrics"] = _depth_metric_summary(depth_metrics)
            refinement_metrics["metric_delta"] = _depth_metric_delta(
                refinement_metrics["initial_depth_metrics"],
                refinement_metrics["final_depth_metrics"],
            )
            refinement_metrics["candidate_changed_object_count"] = refinement_metrics["changed_object_count"]
            refinement_metrics["final_changed_object_count"] = (
                refinement_metrics["changed_object_count"] if refinement_metrics["accepted"] else 0
            )
            model_info["depth_refinement"] = {
                "strategy": refinement_metrics["strategy"],
                "enabled": True,
                "accepted": refinement_metrics["accepted"],
                "candidate_changed_object_count": refinement_metrics["candidate_changed_object_count"],
                "final_changed_object_count": refinement_metrics["final_changed_object_count"],
                "initial_depth_score": refinement_metrics["initial_depth_score"],
                "candidate_depth_score": refinement_metrics["candidate_depth_score"],
                "initial_depth_check_dir": str(output_path / "depth_check_initial"),
                "final_depth_check_dir": str(output_path / "depth_check"),
                "metric_delta": refinement_metrics["metric_delta"],
            }
            (output_path / "depth_refinement.json").write_text(
                json.dumps(refinement_metrics, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        else:
            depth_metrics = write_depth_check(
                source_depth_path=resolved_depth_path,
                fitted_blend_path=camera_space_blend,
                output_dir=output_path / "depth_check",
                near_depth=near_depth,
                far_depth=far_depth,
                blender_executable=blender_executable,
                detections=detections.objects,
            )
            initial_summary = _depth_metric_summary(initial_depth_metrics)
            final_summary = _depth_metric_summary(depth_metrics)
            model_info["depth_refinement"] = {
                "enabled": False,
                "reason": "disabled_by_caller",
                "initial_depth_check_dir": str(output_path / "depth_check_initial"),
                "final_depth_check_dir": str(output_path / "depth_check"),
                "metric_delta": _depth_metric_delta(initial_summary, final_summary),
            }
            disabled_refinement_metrics = {
                "schema_version": 1,
                "strategy": "one_pass_depth_residual_refinement",
                "enabled": False,
                "accepted": False,
                "reason": "disabled_by_caller",
                "initial_depth_metrics": initial_summary,
                "final_depth_metrics": final_summary,
                "metric_delta": model_info["depth_refinement"]["metric_delta"],
            }
            (output_path / "depth_refinement.json").write_text(
                json.dumps(disabled_refinement_metrics, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        dimension_report, dimension_depth_metrics, dimension_refinement_metrics = refine_dimensions_with_accepted_candidates(
            report=report,
            depth_metrics=depth_metrics,
            detections=detections.objects,
            source_depth_path=resolved_depth_path,
            output_dir=output_path / "dimension_refinement",
            temp_dir=Path(temp_dir),
            near_depth=near_depth,
            far_depth=far_depth,
            blender_executable=blender_executable,
        )
        if dimension_refinement_metrics["accepted_candidate_count"] > 0:
            report = replace(dimension_report, model_info=model_info)
            depth_metrics = dimension_depth_metrics
            model_info["dimension_refinement"] = {
                "enabled": True,
                "strategy": dimension_refinement_metrics["strategy"],
                "accepted_candidate_count": dimension_refinement_metrics["accepted_candidate_count"],
                "changed_object_count": dimension_refinement_metrics["changed_object_count"],
                "initial_depth_score": dimension_refinement_metrics["initial_depth_score"],
                "final_depth_score": dimension_refinement_metrics["final_depth_score"],
                "output_dir": str(output_path / "dimension_refinement"),
            }
            report = replace(report, model_info=model_info)
            write_fit_report(report, report_path)
        else:
            model_info["dimension_refinement"] = {
                "enabled": True,
                "strategy": dimension_refinement_metrics["strategy"],
                "accepted_candidate_count": 0,
                "changed_object_count": 0,
                "initial_depth_score": dimension_refinement_metrics["initial_depth_score"],
                "final_depth_score": dimension_refinement_metrics["final_depth_score"],
                "output_dir": str(output_path / "dimension_refinement"),
            }
        (output_path / "dimension_refinement.json").write_text(
            json.dumps(dimension_refinement_metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    final_objects = apply_object_depth_metrics(report.objects, depth_metrics.get("objects", []))
    model_info["fit_quality_summary"] = _fit_quality_summary(depth_metrics, final_objects)
    report = replace(
        report,
        objects=final_objects,
        model_info=model_info,
    )
    write_fit_report(report, report_path)
    write_fit_metrics_summary(depth_metrics, output_path / "metrics_summary.json")
    final_blend_path = fitted_scene_output_path(output_path)
    if final_blend_path != output_path / "fitted_scene.blend":
        stale_nested_blend = output_path / "fitted_scene.blend"
        stale_nested_backup = output_path / "fitted_scene.blend1"
        stale_nested_blend.unlink(missing_ok=True)
        stale_nested_backup.unlink(missing_ok=True)
    export_fit_report_to_blend(
        report_path=report_path,
        output_path=final_blend_path,
        blender_executable=blender_executable,
        layout=final_layout,
        reference_blend_path=reference_blend_path if final_layout in {"ground", "original-camera"} else None,
    )
    return report


def refine_dimensions_with_accepted_candidates(
    report: PrimitiveFitReport,
    depth_metrics: dict,
    detections: list[ObjectShapeDetection],
    source_depth_path: Path,
    output_dir: Path,
    temp_dir: Path,
    near_depth: float,
    far_depth: float,
    blender_executable: str,
) -> tuple[PrimitiveFitReport, dict, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    current_report = report
    current_depth_metrics = depth_metrics
    accepted_candidate_count = 0
    changed_object_ids: set[int] = set()
    object_events: list[dict[str, Any]] = []

    initial_score = _depth_metric_score(current_depth_metrics)
    current_score = initial_score
    for fit in list(current_report.objects):
        object_metric = _object_metric_for_id(current_depth_metrics, fit.id)
        if not _dimension_candidate_eligible(fit, object_metric):
            object_events.append(
                {
                    "id": fit.id,
                    "primitive_label": fit.primitive_label,
                    "eligible": False,
                    "reason": _dimension_ineligible_reason(fit, object_metric),
                }
            )
            continue

        object_event: dict[str, Any] = {
            "id": fit.id,
            "primitive_label": fit.primitive_label,
            "eligible": True,
            "accepted_candidates": [],
            "rejected_candidates": [],
            "initial_object_score": round(_object_dimension_score(object_metric), 6),
        }
        for candidate_index, scales in enumerate(BOX_DIMENSION_CANDIDATE_SCALES, start=1):
            candidate_fit = _scaled_fit_dimensions(fit, scales)
            candidate_report = replace(
                current_report,
                objects=[
                    candidate_fit if item.id == fit.id else item
                    for item in current_report.objects
                ],
            )
            candidate_report_path = temp_dir / f"dimension_candidate_{fit.id}_{candidate_index}.json"
            candidate_blend_path = temp_dir / f"dimension_candidate_{fit.id}_{candidate_index}.blend"
            write_fit_report(candidate_report, candidate_report_path)
            export_fit_report_to_blend(
                report_path=candidate_report_path,
                output_path=candidate_blend_path,
                blender_executable=blender_executable,
                layout="camera",
            )
            candidate_metrics = write_depth_check(
                source_depth_path=source_depth_path,
                fitted_blend_path=candidate_blend_path,
                output_dir=output_dir / f"object_{fit.id:02d}_candidate_{candidate_index:02d}",
                near_depth=near_depth,
                far_depth=far_depth,
                blender_executable=blender_executable,
                detections=detections,
            )
            candidate_object_metric = _object_metric_for_id(candidate_metrics, fit.id)
            accepted, reason = _dimension_candidate_accepted(
                current_depth_metrics=current_depth_metrics,
                candidate_depth_metrics=candidate_metrics,
                current_object_metric=object_metric,
                candidate_object_metric=candidate_object_metric,
            )
            candidate_event = {
                "candidate_index": candidate_index,
                "scales_xyz": [round(float(value), 6) for value in scales],
                "accepted": accepted,
                "reason": reason,
                "object_score": round(_object_dimension_score(candidate_object_metric), 6),
                "depth_score": round(_depth_metric_score(candidate_metrics), 6),
                "dimensions_xyz": [round(float(value), 6) for value in candidate_fit.dimensions_xyz],
            }
            if accepted:
                current_report = candidate_report
                current_depth_metrics = candidate_metrics
                object_metric = candidate_object_metric
                current_score = _depth_metric_score(current_depth_metrics)
                accepted_candidate_count += 1
                changed_object_ids.add(fit.id)
                object_event["accepted_candidates"].append(candidate_event)
            else:
                object_event["rejected_candidates"].append(candidate_event)
        object_event["final_object_score"] = round(_object_dimension_score(object_metric), 6)
        object_events.append(object_event)

    final_score = _depth_metric_score(current_depth_metrics)
    diagnostics = {
        "schema_version": 1,
        "strategy": "accepted_box_dimension_candidates",
        "candidate_scales_xyz": [
            [round(float(value), 6) for value in scales]
            for scales in BOX_DIMENSION_CANDIDATE_SCALES
        ],
        "accepted_candidate_count": accepted_candidate_count,
        "changed_object_count": len(changed_object_ids),
        "changed_object_ids": sorted(changed_object_ids),
        "initial_depth_score": round(float(initial_score), 6),
        "final_depth_score": round(float(final_score), 6),
        "metric_delta": _depth_metric_delta(
            _depth_metric_summary(depth_metrics),
            _depth_metric_summary(current_depth_metrics),
        ),
        "objects": object_events,
        "final_depth_metrics": _depth_metric_summary(current_depth_metrics),
    }
    return current_report, current_depth_metrics, diagnostics


def _object_metric_for_id(metrics: dict, object_id: int) -> dict | None:
    for item in metrics.get("objects", []):
        try:
            if int(item.get("id")) == int(object_id):
                return item
        except (TypeError, ValueError):
            continue
    return None


def _dimension_candidate_eligible(fit: PrimitiveFit, object_metric: dict | None) -> bool:
    if fit.primitive_label != "box":
        return False
    if object_metric is None:
        return False
    return (
        float(object_metric.get("depth_mae") or 0.0) > 0.095
        or float(object_metric.get("bad_pixel_ratio_010") or 0.0) > 0.20
        or float(object_metric.get("missing_source_foreground_ratio") or 0.0) > 0.13
    )


def _dimension_ineligible_reason(fit: PrimitiveFit, object_metric: dict | None) -> str:
    if fit.primitive_label != "box":
        return "not_box"
    if object_metric is None:
        return "missing_object_metrics"
    return "metrics_below_dimension_refinement_threshold"


def _scaled_fit_dimensions(
    fit: PrimitiveFit,
    scales: tuple[float, float, float],
) -> PrimitiveFit:
    old_dimensions = tuple(float(value) for value in fit.dimensions_xyz)
    dimensions = tuple(
        max(0.02, float(dimension) * float(scale))
        for dimension, scale in zip(old_dimensions, scales)
    )
    quality = dict(fit.fit_quality)
    quality["dimension_candidate_source"] = "accepted_box_dimension_candidates"
    quality["dimension_candidate_scales_xyz"] = [round(float(value), 6) for value in scales]
    quality["pre_dimension_candidate_dimensions_xyz"] = [
        round(float(value), 6)
        for value in old_dimensions
    ]
    return replace(fit, dimensions_xyz=dimensions, fit_quality=quality)


def _dimension_candidate_accepted(
    current_depth_metrics: dict,
    candidate_depth_metrics: dict,
    current_object_metric: dict | None,
    candidate_object_metric: dict | None,
) -> tuple[bool, str]:
    if current_object_metric is None or candidate_object_metric is None:
        return False, "missing_object_metrics"
    current_score = _object_dimension_score(current_object_metric)
    candidate_score = _object_dimension_score(candidate_object_metric)
    required_improvement = max(0.004, current_score * 0.025)
    if candidate_score > current_score - required_improvement:
        return False, "object_score_not_improved_enough"

    current_iou = float(current_object_metric.get("foreground_iou") or 0.0)
    candidate_iou = float(candidate_object_metric.get("foreground_iou") or 0.0)
    if candidate_iou < current_iou - 0.006:
        return False, "object_iou_regressed"

    current_missing = float(current_object_metric.get("missing_source_foreground_ratio") or 1.0)
    candidate_missing = float(candidate_object_metric.get("missing_source_foreground_ratio") or 1.0)
    if candidate_missing > current_missing + 0.008:
        return False, "object_missing_foreground_regressed"

    current_extra = float(current_object_metric.get("extra_fitted_foreground_ratio") or 1.0)
    candidate_extra = float(candidate_object_metric.get("extra_fitted_foreground_ratio") or 1.0)
    if candidate_extra > current_extra + 0.018:
        return False, "object_extra_foreground_regressed"

    current_global_score = _depth_metric_score(current_depth_metrics)
    candidate_global_score = _depth_metric_score(candidate_depth_metrics)
    if candidate_global_score > current_global_score:
        return False, "global_depth_score_regressed"

    current_global_iou = float(current_depth_metrics.get("foreground_iou", 0.0))
    candidate_global_iou = float(candidate_depth_metrics.get("foreground_iou", 0.0))
    if candidate_global_iou < current_global_iou - 0.004:
        return False, "global_iou_regressed"

    return True, "accepted"


def _object_dimension_score(object_metric: dict | None) -> float:
    if object_metric is None:
        return 999.0
    foreground_iou = float(object_metric.get("foreground_iou") or 0.0)
    return float(
        float(object_metric.get("depth_mae") or 1.0)
        + float(object_metric.get("bad_pixel_ratio_010") or 1.0) * 0.18
        + (1.0 - foreground_iou) * 0.35
        + float(object_metric.get("missing_source_foreground_ratio") or 1.0) * 0.18
        + float(object_metric.get("extra_fitted_foreground_ratio") or 1.0) * 0.08
    )


def _depth_metric_summary(metrics: dict) -> dict[str, float]:
    keys = (
        "mean_abs_error",
        "rmse",
        "p95_abs_error",
        "bad_pixel_ratio_005",
        "bad_pixel_ratio_010",
        "source_coverage_ratio",
        "fitted_coverage_ratio",
        "foreground_iou",
        "missing_source_foreground_ratio",
        "extra_fitted_foreground_ratio",
    )
    summary: dict[str, float] = {}
    for key in keys:
        value = metrics.get(key)
        if value is None:
            continue
        summary[key] = round(float(value), 6)
    return summary


def _depth_metric_delta(initial: dict[str, float], final: dict[str, float]) -> dict[str, float]:
    delta: dict[str, float] = {}
    for key, initial_value in initial.items():
        if key not in final:
            continue
        delta[key] = round(float(final[key]) - float(initial_value), 6)
    return delta


def _depth_refinement_accepted(initial_metrics: dict, candidate_metrics: dict) -> bool:
    initial_score = _depth_metric_score(initial_metrics)
    candidate_score = _depth_metric_score(candidate_metrics)
    initial_iou = float(initial_metrics.get("foreground_iou", 0.0))
    candidate_iou = float(candidate_metrics.get("foreground_iou", 0.0))
    initial_missing = float(initial_metrics.get("missing_source_foreground_ratio", 1.0))
    candidate_missing = float(candidate_metrics.get("missing_source_foreground_ratio", 1.0))
    initial_extra = float(initial_metrics.get("extra_fitted_foreground_ratio", 1.0))
    candidate_extra = float(candidate_metrics.get("extra_fitted_foreground_ratio", 1.0))
    if candidate_iou < initial_iou - 0.012:
        return False
    if candidate_missing > initial_missing + 0.020:
        return False
    if candidate_extra > initial_extra + 0.030:
        return False
    return candidate_score <= initial_score + max(0.003, initial_score * 0.01)


def _depth_metric_score(metrics: dict) -> float:
    source_coverage = float(metrics.get("source_coverage_ratio", 0.0))
    fitted_coverage = float(metrics.get("fitted_coverage_ratio", 0.0))
    return float(
        float(metrics.get("mean_abs_error", 1.0))
        + float(metrics.get("rmse", 1.0)) * 0.35
        + float(metrics.get("p95_abs_error", 1.0)) * 0.15
        + float(metrics.get("bad_pixel_ratio_010", 1.0)) * 0.10
        + abs(fitted_coverage - source_coverage) * 0.20
        + (1.0 - float(metrics.get("foreground_iou", 0.0))) * 0.25
        + float(metrics.get("missing_source_foreground_ratio", 1.0)) * 0.08
        + float(metrics.get("extra_fitted_foreground_ratio", 1.0)) * 0.08
    )


def _fit_quality_summary(metrics: dict, fits: list[PrimitiveFit]) -> dict:
    review_count = sum(1 for item in fits if item.fit_quality.get("status") == "needs_review")
    foreground_iou = float(metrics.get("foreground_iou", 0.0))
    mean_abs_error = float(metrics.get("mean_abs_error", 1.0))
    extra_ratio = float(metrics.get("extra_fitted_foreground_ratio", 1.0))
    missing_ratio = float(metrics.get("missing_source_foreground_ratio", 1.0))
    score = _depth_metric_score(metrics)
    if (
        review_count == 0
        and foreground_iou >= 0.82
        and mean_abs_error <= 0.16
        and extra_ratio <= 0.14
        and missing_ratio <= 0.14
    ):
        verdict = "good"
    elif foreground_iou >= 0.68 and mean_abs_error <= 0.28 and extra_ratio <= 0.28 and missing_ratio <= 0.28:
        verdict = "usable_needs_review"
    else:
        verdict = "needs_review"
    return {
        "schema_version": 1,
        "verdict": verdict,
        "quality_gate_passed": verdict == "good",
        "score_scale": "lower_is_better",
        "depth_score": round(score, 6),
        "object_count": len(fits),
        "needs_review_object_count": review_count,
        "foreground_iou": round(foreground_iou, 6),
        "mean_abs_error": round(mean_abs_error, 6),
        "extra_fitted_foreground_ratio": round(extra_ratio, 6),
        "missing_source_foreground_ratio": round(missing_ratio, 6),
        "good_thresholds": {
            "foreground_iou_min": 0.82,
            "mean_abs_error_max": 0.16,
            "extra_fitted_foreground_ratio_max": 0.14,
            "missing_source_foreground_ratio_max": 0.14,
            "needs_review_object_count_max": 0,
        },
        "usable_thresholds": {
            "foreground_iou_min": 0.68,
            "mean_abs_error_max": 0.28,
            "extra_fitted_foreground_ratio_max": 0.28,
            "missing_source_foreground_ratio_max": 0.28,
        },
    }


def apply_object_depth_metrics(fits: list[PrimitiveFit], metrics: list[dict]) -> list[PrimitiveFit]:
    metrics_by_id = {int(item["id"]): item for item in metrics}
    updated: list[PrimitiveFit] = []
    for fit in fits:
        object_metrics = metrics_by_id.get(fit.id)
        if object_metrics is None:
            updated.append(fit)
            continue
        quality = dict(fit.fit_quality)
        if object_metrics.get("depth_mae") is not None:
            quality["depth_mae"] = object_metrics["depth_mae"]
            quality["depth_rmse"] = object_metrics["depth_rmse"]
            quality["bad_pixel_ratio_010"] = object_metrics["bad_pixel_ratio_010"]
            quality["foreground_iou"] = object_metrics.get("foreground_iou")
            quality["missing_source_foreground_ratio"] = object_metrics.get("missing_source_foreground_ratio")
            quality["extra_fitted_foreground_ratio"] = object_metrics.get("extra_fitted_foreground_ratio")
            if float(object_metrics["bad_pixel_ratio_010"]) > 0.35 or float(object_metrics["depth_mae"]) > 0.18:
                quality["status"] = "needs_review"
            if (
                object_metrics.get("foreground_iou") is not None
                and float(object_metrics["foreground_iou"]) < 0.55
            ):
                quality["status"] = "needs_review"
        quality["mask_pixel_count"] = object_metrics["mask_pixel_count"]
        updated.append(replace(fit, fit_quality=quality))
    return updated


def apply_enrichment_fit_metadata(fit: PrimitiveFit, enrichment_object) -> PrimitiveFit:
    quality = dict(fit.fit_quality)
    quality.setdefault("status", "ok")
    quality["schema_version"] = 2
    quality["score_scale"] = "lower_is_better"
    quality["edge_boundary_agreement"] = enrichment_object.edge.boundary_agreement
    quality["wireframe_status"] = enrichment_object.wireframe.status
    quality["mesh_status"] = enrichment_object.mesh.status
    quality["mesh_candidate_path"] = enrichment_object.mesh.path if enrichment_object.mesh.status == "ok" else None
    quality["original_detector_label"] = enrichment_object.original_detector_label
    quality["geometry_selected_label"] = enrichment_object.geometry.selected_label
    fused_state = _resolve_fitting_contract(enrichment_object)
    if fused_state is None:
        raise ValueError(f"Missing fused contract for object id {fit.id} in enrichment report.")
    quality["fused_label"] = fused_state.fused_label
    quality["fused_confidence"] = fused_state.fused_confidence
    quality["fused_contributions"] = fused_state.fused_contributions
    quality["needs_review"] = bool(fused_state.needs_review)
    quality["needs_review_reason"] = list(fused_state.needs_review_reason)
    quality["label_source"] = "fused"
    if enrichment_object.mesh.status not in {"ok", "missing", "skipped"}:
        quality["status"] = "needs_review"
    return replace(
        fit,
        primitive_label=fused_state.fused_label,
        confidence=fused_state.fused_confidence,
        primitive_label_source="fused",
        fit_quality=quality,
    )


def _resolve_fitting_contract(enrichment_object: Any) -> FusedState | None:
    fused_state = getattr(enrichment_object, "fused_state", None)
    if fused_state is None:
        return None
    fused_label = str(fused_state.fused_label).strip().lower()
    if fused_label not in FUSED_LABELS:
        fused_label = "unknown"
    try:
        fused_confidence = float(fused_state.fused_confidence)
    except (TypeError, ValueError):
        fused_confidence = 0.0
    fused_confidence = max(0.0, min(1.0, fused_confidence))
    if fused_label != fused_state.fused_label or fused_confidence != fused_state.fused_confidence:
        return replace(
            fused_state,
            fused_label=fused_label,
            fused_confidence=fused_confidence,
        )
    return fused_state
