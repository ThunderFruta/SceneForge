from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from OutputWriter.metrics_summary import write_fit_metrics_summary, write_metrics_comparison_summary


def write_metric_image(root: Path, group: str, name: str, color: tuple[int, int, int]) -> None:
    path = root / group
    path.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path / f"{name}.png")


def test_write_fit_metrics_summary_keeps_camera_and_worst_objects(tmp_path: Path) -> None:
    output = tmp_path / "metrics_summary.json"

    summary = write_fit_metrics_summary(
        {
            "mean_abs_error": 0.12,
            "rmse": 0.2,
            "p95_abs_error": 0.4,
            "bad_pixel_ratio_010": 0.3,
            "objects": [
                {"id": 1, "depth_mae": 0.1, "bad_pixel_ratio_010": 0.2},
                {"id": 2, "depth_mae": 0.2, "bad_pixel_ratio_010": 0.8},
            ],
        },
        output,
    )

    assert summary["camera_view_depth"]["mean_abs_error"] == 0.12
    assert summary["object_worst"][0]["id"] == 2
    assert json.loads(output.read_text(encoding="utf-8")) == summary


def test_write_metrics_comparison_summary_writes_csv_and_summary(tmp_path: Path) -> None:
    original = tmp_path / "original"
    generated = tmp_path / "generated"
    output = tmp_path / "metrics"
    write_metric_image(original, "depth", "pos_z", (255, 255, 255))
    write_metric_image(generated, "depth", "pos_z", (128, 128, 128))
    write_metric_image(original, "normal", "pos_z", (0, 0, 0))
    write_metric_image(generated, "normal", "pos_z", (0, 64, 0))
    write_metric_image(original, "preview", "camera", (20, 20, 20))
    write_metric_image(generated, "preview", "camera", (40, 40, 40))

    summary = write_metrics_comparison_summary(original, generated, output)

    assert (output / "summary.json").is_file()
    assert (output / "comparison" / "metrics_comparison.csv").is_file()
    assert (output / "comparison" / "preview" / "camera_comparison.png").is_file()
    assert (output / "comparison" / "depth" / "pos_z_comparison.png").is_file()
    assert summary["camera_preview"]["worst_view"] == "camera"
    assert summary["all_axis_depth"]["worst_view"] == "pos_z"
    assert summary["top_failing_views"][0]["group"] == "depth"
