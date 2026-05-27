from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from PrimitiveFitting.masks import polygon_to_mask
from ShapeDetection.report import ObjectShapeDetection
from OutputWriter.depth_colormap import thermal_colormap


class DepthCheckError(RuntimeError):
    pass


def write_depth_check(
    source_depth_path: str | Path,
    fitted_blend_path: str | Path,
    output_dir: str | Path,
    near_depth: float,
    far_depth: float,
    blender_executable: str = "blender",
    detections: list[ObjectShapeDetection] | None = None,
) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fitted_depth_path = output_path / "fitted_depth.png"
    render_fitted_depth(
        fitted_blend_path=fitted_blend_path,
        output_path=fitted_depth_path,
        near_depth=near_depth,
        far_depth=far_depth,
        blender_executable=blender_executable,
    )

    source = _load_depth_image(source_depth_path)
    fitted = _load_depth_image(fitted_depth_path)
    if source.shape != fitted.shape:
        raise DepthCheckError(
            f"Fitted depth size {fitted.shape[1]}x{fitted.shape[0]} does not match "
            f"source depth size {source.shape[1]}x{source.shape[0]}."
        )

    difference = np.abs(source - fitted)
    source_non_far = source > 0.01
    fitted_non_far = fitted > 0.01
    coverage_union = source_non_far | fitted_non_far
    if not np.any(coverage_union):
        coverage_union = np.ones_like(source, dtype=bool)
    foreground_intersection = source_non_far & fitted_non_far
    foreground_union = source_non_far | fitted_non_far
    foreground_union_count = max(1, int(np.count_nonzero(foreground_union)))
    source_foreground_count = max(1, int(np.count_nonzero(source_non_far)))
    fitted_foreground_count = max(1, int(np.count_nonzero(fitted_non_far)))

    metrics = {
        "source_depth_path": str(source_depth_path),
        "fitted_depth_path": str(fitted_depth_path),
        "near_depth": round(float(near_depth), 6),
        "far_depth": round(float(far_depth), 6),
        "mean_abs_error": round(float(np.mean(difference[coverage_union])), 6),
        "rmse": round(float(np.sqrt(np.mean(np.square(difference[coverage_union])))), 6),
        "p95_abs_error": round(float(np.percentile(difference[coverage_union], 95)), 6),
        "bad_pixel_ratio_005": round(float(np.mean(difference[coverage_union] > 0.05)), 6),
        "bad_pixel_ratio_010": round(float(np.mean(difference[coverage_union] > 0.10)), 6),
        "source_coverage_ratio": round(float(np.mean(source_non_far)), 6),
        "fitted_coverage_ratio": round(float(np.mean(fitted_non_far)), 6),
        "foreground_iou": round(float(np.count_nonzero(foreground_intersection)) / foreground_union_count, 6),
        "missing_source_foreground_ratio": round(float(np.count_nonzero(source_non_far & ~fitted_non_far)) / source_foreground_count, 6),
        "extra_fitted_foreground_ratio": round(float(np.count_nonzero(fitted_non_far & ~source_non_far)) / fitted_foreground_count, 6),
    }
    if detections is not None:
        metrics["objects"] = object_depth_metrics(
            detections=detections,
            difference=difference,
            source=source,
            fitted=fitted,
        )

    _save_rgb(output_path / "source_depth_thermal.png", thermal_colormap(source, auto_contrast=True))
    _save_rgb(output_path / "fitted_depth_thermal.png", thermal_colormap(fitted, auto_contrast=True))
    _save_l(output_path / "depth_difference.png", difference)
    _save_rgb(output_path / "depth_difference_thermal.png", thermal_colormap(difference, auto_contrast=True))
    _save_side_by_side(
        output_path / "depth_check_side_by_side.png",
        [
            output_path / "source_depth_thermal.png",
            output_path / "fitted_depth_thermal.png",
            output_path / "depth_difference_thermal.png",
        ],
    )
    (output_path / "depth_check.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def object_depth_metrics(
    detections: list[ObjectShapeDetection],
    difference: np.ndarray,
    source: np.ndarray,
    fitted: np.ndarray,
) -> list[dict]:
    height, width = source.shape
    rows: list[dict] = []
    for detection in detections:
        mask = polygon_to_mask(detection.mask_polygon, width, height)
        source_present = source > 0.01
        fitted_present = fitted > 0.01
        valid = mask & (source_present | fitted_present)
        if not np.any(valid):
            rows.append(
                {
                    "id": detection.id,
                    "depth_mae": None,
                    "depth_rmse": None,
                    "bad_pixel_ratio_010": None,
                    "mask_pixel_count": int(np.count_nonzero(mask)),
                }
            )
            continue
        values = difference[valid]
        object_source_present = mask & source_present
        object_fitted_present = mask & fitted_present
        object_foreground_union = object_source_present | object_fitted_present
        object_foreground_union_count = max(1, int(np.count_nonzero(object_foreground_union)))
        object_source_count = max(1, int(np.count_nonzero(object_source_present)))
        object_fitted_count = max(1, int(np.count_nonzero(object_fitted_present)))
        rows.append(
            {
                "id": detection.id,
                "depth_mae": round(float(np.mean(values)), 6),
                "depth_rmse": round(float(np.sqrt(np.mean(np.square(values)))), 6),
                "bad_pixel_ratio_010": round(float(np.mean(values > 0.10)), 6),
                "foreground_iou": round(float(np.count_nonzero(object_source_present & object_fitted_present)) / object_foreground_union_count, 6),
                "missing_source_foreground_ratio": round(float(np.count_nonzero(object_source_present & ~object_fitted_present)) / object_source_count, 6),
                "extra_fitted_foreground_ratio": round(float(np.count_nonzero(object_fitted_present & ~object_source_present)) / object_fitted_count, 6),
                "mask_pixel_count": int(np.count_nonzero(mask)),
            }
        )
    return rows


def render_fitted_depth(
    fitted_blend_path: str | Path,
    output_path: str | Path,
    near_depth: float,
    far_depth: float,
    blender_executable: str = "blender",
) -> None:
    if shutil.which(blender_executable) is None:
        raise DepthCheckError(f"Blender executable was not found: {blender_executable}")

    script_path = Path(__file__).resolve().parents[1] / "Tools" / "Scripts" / "render_blend_depth.py"
    command = [
        blender_executable,
        "-b",
        str(fitted_blend_path),
        "--python",
        str(script_path),
        "--",
        "--output",
        str(output_path),
        "--near-depth",
        str(near_depth),
        "--far-depth",
        str(far_depth),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise DepthCheckError(
            f"Fitted depth render failed with exit code {result.returncode}: {result.stderr.strip()}"
        )


def _load_depth_image(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def _save_rgb(path: Path, values: np.ndarray) -> None:
    Image.fromarray(values, mode="RGB").save(path)


def _save_l(path: Path, values: np.ndarray) -> None:
    Image.fromarray(np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L").save(path)


def _save_side_by_side(path: Path, image_paths: list[Path]) -> None:
    images = [Image.open(item).convert("RGB") for item in image_paths]
    width = sum(image.width for image in images)
    height = max(image.height for image in images)
    output = Image.new("RGB", (width, height), (0, 0, 0))
    x = 0
    for image in images:
        output.paste(image, (x, 0))
        x += image.width
    output.save(path)
