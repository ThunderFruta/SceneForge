from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


class DepthLoadError(ValueError):
    pass


def load_grayscale_depth(
    path: str | Path,
    expected_size: tuple[int, int] | None = None,
) -> np.ndarray:
    depth_path = Path(path)
    if not depth_path.is_file():
        raise DepthLoadError(f"Depth path does not exist or is not a file: {depth_path}")

    try:
        with Image.open(depth_path) as image:
            depth = image.convert("L")
    except UnidentifiedImageError as exc:
        raise DepthLoadError(f"Depth file could not be decoded: {depth_path}") from exc

    if expected_size is not None and depth.size != expected_size:
        raise DepthLoadError(
            f"Depth image size {depth.size[0]}x{depth.size[1]} does not match "
            f"RGB image size {expected_size[0]}x{expected_size[1]}."
        )

    return np.asarray(depth, dtype=np.float32) / 255.0
