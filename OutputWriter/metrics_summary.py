from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageStat


METRIC_GROUPS = ("preview", "iso", "ortho", "depth", "normal")
MULTI_VIEW_SCORE_WEIGHTS = {
    "camera_view_depth": 0.45,
    "camera_preview": 0.20,
    "axis_depth": 0.14,
    "axis_normal": 0.09,
    "ortho": 0.07,
    "iso": 0.05,
}


def write_fit_metrics_summary(depth_metrics: dict, output_path: str | Path) -> dict:
    object_rows = [
        item
        for item in depth_metrics.get("objects", [])
        if item.get("depth_mae") is not None
    ]
    object_rows.sort(key=lambda item: float(item.get("bad_pixel_ratio_010", 0.0)), reverse=True)
    summary = {
        "camera_view_depth": {
            "mean_abs_error": depth_metrics.get("mean_abs_error"),
            "rmse": depth_metrics.get("rmse"),
            "p95_abs_error": depth_metrics.get("p95_abs_error"),
            "bad_pixel_ratio_010": depth_metrics.get("bad_pixel_ratio_010"),
            "source_coverage_ratio": depth_metrics.get("source_coverage_ratio"),
            "fitted_coverage_ratio": depth_metrics.get("fitted_coverage_ratio"),
        },
        "object_worst": object_rows[:5],
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def write_metrics_comparison_summary(
    original_metrics_dir: str | Path,
    generated_metrics_dir: str | Path,
    output_dir: str | Path,
    depth_check_path: str | Path | None = None,
) -> dict:
    original_root = Path(original_metrics_dir)
    generated_root = Path(generated_metrics_dir)
    output_root = Path(output_dir)
    comparison_root = output_root / "comparison"
    rows = compare_metric_images(original_root, generated_root, comparison_root)

    depth_rows = [item for item in rows if item["group"] == "depth"]
    normal_rows = [item for item in rows if item["group"] == "normal"]
    preview_rows = [item for item in rows if item["group"] == "preview"]
    ortho_rows = [item for item in rows if item["group"] == "ortho"]
    iso_rows = [item for item in rows if item["group"] == "iso"]
    summary = {
        "camera_preview": summarize_rows(preview_rows),
        "all_axis_depth": summarize_rows(depth_rows),
        "all_axis_normal": summarize_rows(normal_rows),
        "all_axis_ortho": summarize_rows(ortho_rows),
        "all_axis_iso": summarize_rows(iso_rows),
        "top_failing_views": rows[:10],
    }
    if depth_check_path is not None:
        depth_check = json.loads(Path(depth_check_path).read_text(encoding="utf-8"))
        summary["camera_view_depth"] = {
            "mean_abs_error": depth_check.get("mean_abs_error"),
            "rmse": depth_check.get("rmse"),
            "p95_abs_error": depth_check.get("p95_abs_error"),
            "bad_pixel_ratio_010": depth_check.get("bad_pixel_ratio_010"),
        }
        summary["object_worst"] = sorted(
            [
                item
                for item in depth_check.get("objects", [])
                if item.get("depth_mae") is not None
            ],
            key=lambda item: float(item.get("bad_pixel_ratio_010", 0.0)),
            reverse=True,
        )[:5]
    summary["multi_view_quality"] = multi_view_quality_summary(summary)

    output_root.mkdir(parents=True, exist_ok=True)
    write_comparison_csv(comparison_root / "metrics_comparison.csv", rows)
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def multi_view_quality_summary(summary: dict) -> dict:
    components = {
        "camera_view_depth": camera_view_depth_score(summary.get("camera_view_depth", {})),
        "camera_preview": row_summary_score(summary.get("camera_preview", {})),
        "axis_depth": row_summary_score(summary.get("all_axis_depth", {})),
        "axis_normal": row_summary_score(summary.get("all_axis_normal", {})),
        "ortho": row_summary_score(summary.get("all_axis_ortho", {})),
        "iso": row_summary_score(summary.get("all_axis_iso", {})),
    }
    available = {
        key: value
        for key, value in components.items()
        if value is not None
    }
    if not available:
        return {
            "score": None,
            "verdict": "missing",
            "score_scale": "lower_is_better",
            "weights": MULTI_VIEW_SCORE_WEIGHTS,
            "components": components,
        }

    total_weight = sum(MULTI_VIEW_SCORE_WEIGHTS[key] for key in available)
    normalized_weights = {
        key: round(float(MULTI_VIEW_SCORE_WEIGHTS[key] / total_weight), 6)
        for key in available
    }
    score = sum(float(available[key]) * normalized_weights[key] for key in available)
    if score <= 0.08:
        verdict = "excellent"
    elif score <= 0.14:
        verdict = "good"
    elif score <= 0.24:
        verdict = "usable_needs_review"
    else:
        verdict = "needs_review"
    return {
        "score": round(float(score), 6),
        "verdict": verdict,
        "score_scale": "lower_is_better",
        "weights": normalized_weights,
        "raw_weights": MULTI_VIEW_SCORE_WEIGHTS,
        "components": {
            key: round(float(value), 6) if value is not None else None
            for key, value in components.items()
        },
        "forward_weight": round(
            float(normalized_weights.get("camera_view_depth", 0.0) + normalized_weights.get("camera_preview", 0.0)),
            6,
        ),
        "side_and_top_weight": round(
            float(
                normalized_weights.get("axis_depth", 0.0)
                + normalized_weights.get("axis_normal", 0.0)
                + normalized_weights.get("ortho", 0.0)
                + normalized_weights.get("iso", 0.0)
            ),
            6,
        ),
    }


def row_summary_score(summary: dict | None) -> float | None:
    if not isinstance(summary, dict):
        return None
    value = summary.get("mean_mae")
    if value is None:
        return None
    return float(value)


def camera_view_depth_score(summary: dict | None) -> float | None:
    if not isinstance(summary, dict):
        return None
    mean_abs_error = summary.get("mean_abs_error")
    if mean_abs_error is None:
        return None
    return float(
        float(mean_abs_error)
        + float(summary.get("rmse", 0.0)) * 0.35
        + float(summary.get("p95_abs_error", 0.0)) * 0.15
        + float(summary.get("bad_pixel_ratio_010", 0.0)) * 0.10
    )


def compare_metric_images(original_root: Path, generated_root: Path, output_root: Path) -> list[dict]:
    rows: list[dict] = []
    for group in METRIC_GROUPS:
        original_dir = original_root / group
        generated_dir = generated_root / group
        if not original_dir.is_dir() or not generated_dir.is_dir():
            continue
        comparison_dir = output_root / group
        comparison_dir.mkdir(parents=True, exist_ok=True)
        for original_path in sorted(original_dir.glob("*.png")):
            generated_path = generated_dir / original_path.name
            if not generated_path.exists():
                continue
            original = Image.open(original_path).convert("RGB")
            generated = Image.open(generated_path).convert("RGB")
            if generated.size != original.size:
                generated = generated.resize(original.size)
            diff = ImageChops.difference(original, generated)
            stat = ImageStat.Stat(diff)
            row = {
                "group": group,
                "view": original_path.stem,
                "mae": round(float(sum(stat.mean) / (3 * 255.0)), 6),
                "rmse": round(float((sum(value**2 for value in stat.rms) / 3.0) ** 0.5 / 255.0), 6),
                "max_diff": round(float(diff.convert("L").getextrema()[1] / 255.0), 6),
            }
            rows.append(row)
            write_side_by_side(
                comparison_dir / f"{original_path.stem}_comparison.png",
                original,
                generated,
                diff,
                f"diff mae={row['mae']:.3f}",
            )
    rows.sort(key=lambda item: float(item["mae"]), reverse=True)
    return rows


def summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        return {"mean_mae": None, "worst_view": None, "worst_mae": None}
    mean_mae = sum(float(item["mae"]) for item in rows) / len(rows)
    worst = max(rows, key=lambda item: float(item["mae"]))
    return {
        "mean_mae": round(mean_mae, 6),
        "worst_view": worst["view"],
        "worst_mae": worst["mae"],
    }


def write_comparison_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("group", "view", "mae", "rmse", "max_diff"))
        writer.writeheader()
        writer.writerows(rows)


def write_side_by_side(path: Path, original: Image.Image, generated: Image.Image, diff: Image.Image, diff_label: str) -> None:
    output = Image.new("RGB", (original.width * 3, original.height), (20, 20, 20))
    output.paste(original, (0, 0))
    output.paste(generated, (original.width, 0))
    output.paste(diff, (original.width * 2, 0))
    draw = ImageDraw.Draw(output)
    for label, x in (
        ("original", 0),
        ("generated", original.width),
        (diff_label, original.width * 2),
    ):
        draw.rectangle((x + 8, 8, x + 260, 32), fill=(0, 0, 0))
        draw.text((x + 14, 13), label, fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    output.save(path)
