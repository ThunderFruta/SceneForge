from __future__ import annotations

from pathlib import Path

from PIL import Image

from Segmentation.depth_edge_segmenter import (
    DepthEdgeSegmenter,
    candidate_components_from_boundaries,
    component_bbox,
    component_polygon,
    merge_object_fragments,
    should_keep_component,
)

import numpy as np


def test_depth_edge_segmenter_emits_geometry_first_masks(tmp_path: Path) -> None:
    image = Image.new("RGB", (64, 64), "black")
    depth = Image.new("L", (64, 64), 210)
    for y in range(24, 40):
        for x in range(24, 40):
            depth.putpixel((x, y), 70)
    depth_path = tmp_path / "depth.png"
    depth.save(depth_path)

    segmenter = DepthEdgeSegmenter(
        depth_path=depth_path,
        min_component_area_ratio=0.001,
        max_components=4,
    )
    detections = segmenter.detect(image)

    assert detections
    assert segmenter.backend == "depth-edge-instance-scaffold"
    assert segmenter.input_channels == ("depth", "edge")
    assert segmenter.backend_info.to_dict()["primitive_labels_are_authoritative"] is False
    assert segmenter.backend_info.to_dict()["legacy"] is False
    assert {item.detector_label for item in detections}.issubset({"plane", "unknown"})
    assert any(item.detector_label == "unknown" for item in detections)


def test_component_polygon_preserves_non_rectangular_component_shape() -> None:
    component = np.zeros((8, 8), dtype=bool)
    component[1:6, 1:3] = True
    component[1:3, 1:6] = True

    polygon = component_polygon(component, component_bbox(component))

    assert len(polygon) > 4
    assert polygon != [(1.0, 1.0), (6.0, 1.0), (6.0, 6.0), (1.0, 6.0)]


def test_depth_edge_segmenter_suppresses_full_frame_support_plane(tmp_path: Path) -> None:
    image = Image.new("RGB", (96, 64), "black")
    depth = Image.new("L", (96, 64), 210)
    for y in range(24, 40):
        for x in range(36, 52):
            depth.putpixel((x, y), 70)
    depth_path = tmp_path / "depth.png"
    depth.save(depth_path)

    segmenter = DepthEdgeSegmenter(
        depth_path=depth_path,
        min_component_area_ratio=0.001,
        max_components=8,
    )
    detections = segmenter.detect(image)

    assert detections
    assert all(item.bbox_xyxy != (1.0, 1.0, 95.0, 63.0) for item in detections)
    assert all(item.detector_label != "plane" for item in detections)


def test_merge_object_fragments_groups_same_depth_same_chroma_faces() -> None:
    depth = np.full((72, 112), 0.5, dtype=np.float32)
    rgb = Image.new("RGB", (112, 72), (30, 80, 180))
    pixels = rgb.load()
    for y in range(20, 48):
        for x in range(92, 108):
            pixels[x, y] = (180, 60, 60)

    left_face = np.zeros((72, 112), dtype=bool)
    left_face[8:32, 10:55] = True
    left_face[32:55, 10:25] = True
    right_face = np.zeros((72, 112), dtype=bool)
    right_face[34:62, 30:75] = True
    other_object = np.zeros((72, 112), dtype=bool)
    other_object[20:48, 92:108] = True

    merged = merge_object_fragments([left_face, right_face, other_object], depth, rgb)

    assert len(merged) == 2
    assert sorted(int(item.sum()) for item in merged) == sorted([
        int((left_face | right_face).sum()),
        int(other_object.sum()),
    ])


def test_should_keep_compact_tiny_visible_component() -> None:
    component = np.zeros((64, 64), dtype=bool)
    component[20:34, 30:44] = True

    assert should_keep_component(component, (64, 64), min_area=512)


def test_candidate_components_recovers_edge_enclosed_region() -> None:
    boundaries = np.zeros((32, 48), dtype=bool)
    boundaries[8, 12:36] = True
    boundaries[23, 12:36] = True
    boundaries[8:24, 12] = True
    boundaries[8:24, 35] = True
    boundaries[15, 12] = False

    components = candidate_components_from_boundaries(boundaries, tiny_area=16, max_components=8)

    assert any(component_bbox(component) == (14.0, 10.0, 34.0, 22.0) for component in components)
