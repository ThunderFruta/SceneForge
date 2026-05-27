from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


THERMAL_STOPS = (
    (0.00, (0, 0, 0)),
    (0.14, (42, 0, 92)),
    (0.30, (0, 35, 180)),
    (0.48, (0, 190, 255)),
    (0.64, (0, 220, 110)),
    (0.78, (255, 220, 0)),
    (0.91, (255, 70, 0)),
    (1.00, (255, 255, 255)),
)


def thermal_colormap(depth_values: np.ndarray, auto_contrast: bool = True) -> np.ndarray:
    values = _normalize_depth(depth_values, auto_contrast=auto_contrast)
    output = np.zeros((*values.shape, 3), dtype=np.float32)

    for (start_value, start_color), (end_value, end_color) in zip(THERMAL_STOPS, THERMAL_STOPS[1:]):
        mask = (values >= start_value) & (values <= end_value)
        if not np.any(mask):
            continue
        span = max(end_value - start_value, 1e-6)
        t = ((values[mask] - start_value) / span)[:, None]
        start = np.asarray(start_color, dtype=np.float32)
        end = np.asarray(end_color, dtype=np.float32)
        output[mask] = start + (end - start) * t

    return np.rint(np.clip(output, 0, 255)).astype(np.uint8)


def write_thermal_depth_preview(
    input_path: str | Path,
    output_path: str | Path,
    auto_contrast: bool = True,
) -> None:
    with Image.open(input_path) as image:
        depth = np.asarray(image.convert("L"), dtype=np.float32)

    preview = Image.fromarray(thermal_colormap(depth, auto_contrast=auto_contrast), mode="RGB")
    resolved_output = Path(output_path)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    preview.save(resolved_output)


def _normalize_depth(depth_values: np.ndarray, auto_contrast: bool) -> np.ndarray:
    values = depth_values.astype(np.float32, copy=False)
    if values.size == 0:
        return values
    if float(np.nanmax(values)) > 1.0:
        values = values / 255.0
    values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)

    if auto_contrast:
        low, high = np.percentile(values, (1.0, 99.0))
        if high > low:
            values = (values - low) / (high - low)

    return np.clip(values, 0.0, 1.0)
