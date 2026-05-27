from __future__ import annotations

import numpy as np

from PrimitiveFitting.depth_check import object_depth_metrics
from ShapeDetection.report import ObjectShapeDetection


def test_object_depth_metrics_measure_masked_difference() -> None:
    detection = ObjectShapeDetection(
        id=7,
        bbox_xyxy=(1.0, 1.0, 3.0, 3.0),
        mask_polygon=[(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)],
        detector_label="box",
        detector_confidence=0.9,
        primitive_label="box",
        primitive_confidence=0.8,
    )
    source = np.ones((5, 5), dtype=np.float32)
    fitted = np.ones((5, 5), dtype=np.float32) * 0.8
    difference = np.abs(source - fitted)

    metrics = object_depth_metrics([detection], difference, source, fitted)

    assert metrics[0]["id"] == 7
    assert metrics[0]["depth_mae"] == 0.2
    assert metrics[0]["bad_pixel_ratio_010"] == 1.0
    assert metrics[0]["mask_pixel_count"] > 0
