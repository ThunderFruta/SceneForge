from __future__ import annotations

import numpy as np

from OutputWriter.depth_colormap import thermal_colormap


def test_thermal_colormap_returns_rgb_uint8() -> None:
    depth = np.array([[0, 128, 255]], dtype=np.uint8)

    colored = thermal_colormap(depth, auto_contrast=False)

    assert colored.shape == (1, 3, 3)
    assert colored.dtype == np.uint8
    assert tuple(colored[0, 0]) == (0, 0, 0)
    assert tuple(colored[0, 2]) == (255, 255, 255)
