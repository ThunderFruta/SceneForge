from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from Segmentation.rgbd_yolo_segmenter import make_bgrd_array, suppress_unreliable_plane_detections
from Segmentation.types import SegmentDetection


def test_make_bgrd_array_fuses_rgb_and_depth(tmp_path: Path) -> None:
    image = Image.new("RGB", (2, 1))
    image.putpixel((0, 0), (10, 20, 30))
    image.putpixel((1, 0), (40, 50, 60))
    depth_path = tmp_path / "depth.png"
    depth = Image.new("L", (2, 1))
    depth.putpixel((0, 0), 255)
    depth.putpixel((1, 0), 0)
    depth.save(depth_path)

    fused = make_bgrd_array(image, depth_path, channel_weights=(0.25, 0.25, 0.25, 0.25))

    assert fused.shape == (1, 2, 4)
    assert fused[0, 0].tolist() == [30, 20, 10, 255]
    assert fused[0, 1].tolist() == [60, 50, 40, 0]


def test_make_bgrd_array_can_emphasize_depth(tmp_path: Path) -> None:
    image = Image.new("RGB", (1, 1), (100, 100, 100))
    depth_path = tmp_path / "depth.png"
    Image.new("L", (1, 1), 128).save(depth_path)

    fused = make_bgrd_array(image, depth_path, channel_weights=(0.20, 0.20, 0.20, 0.40))

    assert fused[0, 0].tolist() == [80, 80, 80, 204]


def test_suppress_unreliable_plane_detections_drops_low_confidence_fragment() -> None:
    floor = SegmentDetection(
        bbox_xyxy=(0.0, 50.0, 100.0, 100.0),
        mask_polygon=[(0.0, 50.0), (100.0, 50.0), (100.0, 100.0), (0.0, 100.0)],
        detector_label="plane",
        detector_confidence=0.87,
    )
    fragment = SegmentDetection(
        bbox_xyxy=(30.0, 55.0, 70.0, 75.0),
        mask_polygon=[(30.0, 55.0), (70.0, 55.0), (70.0, 75.0), (30.0, 75.0)],
        detector_label="plane",
        detector_confidence=0.29,
    )
    cone = SegmentDetection(
        bbox_xyxy=(40.0, 40.0, 60.0, 80.0),
        mask_polygon=[(50.0, 40.0), (60.0, 80.0), (40.0, 80.0)],
        detector_label="cone",
        detector_confidence=0.91,
    )

    kept = suppress_unreliable_plane_detections(
        [floor, fragment, cone],
        image_size=(100, 100),
        depth=np.full((100, 100), 0.5, dtype=np.float32),
    )

    assert kept == [floor, cone]
