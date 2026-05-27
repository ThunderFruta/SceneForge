from __future__ import annotations

import numpy as np

from PrimitiveFitting.camera import PinholeCamera
from PrimitiveFitting.fitter import fit_primitive, fit_primitive_candidates
from ShapeDetection.report import ObjectShapeDetection


def make_detection(label: str) -> ObjectShapeDetection:
    return ObjectShapeDetection(
        id=1,
        bbox_xyxy=(0.0, 0.0, 10.0, 10.0),
        mask_polygon=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
        detector_label=label,
        detector_confidence=0.9,
        primitive_label=label,
        primitive_confidence=0.8,
    )


def sample_points() -> np.ndarray:
    xs = np.linspace(-0.5, 0.5, 5)
    ys = np.linspace(2.0, 2.5, 5)
    zs = np.linspace(-0.25, 0.25, 5)
    return np.array([(x, y, z) for x in xs for y in ys for z in zs], dtype=np.float64)


def test_fitters_return_finite_transforms_for_supported_primitives() -> None:
    points = sample_points()

    for label in ("sphere", "cylinder", "cone", "box", "plane"):
        fit = fit_primitive(make_detection(label), points)
        values = [*fit.center_xyz, *fit.dimensions_xyz]
        assert all(np.isfinite(values))
        assert fit.primitive_label == label
        assert fit.fit_quality["sample_count"] == len(points)


def test_unknown_detection_fits_as_unknown_geometric_proxy() -> None:
    fit = fit_primitive(make_detection("unknown"), sample_points())

    assert fit.primitive_label == "unknown"


def test_candidate_fit_generation_returns_finite_transforms() -> None:
    camera = PinholeCamera(image_width=100, image_height=100)
    candidates = fit_primitive_candidates(make_detection("box"), sample_points(), camera=camera)

    assert {item.fit_quality["mode"] for item in candidates} == {"camera_silhouette", "depth_pca"}
    for fit in candidates:
        values = [*fit.center_xyz, *fit.dimensions_xyz]
        assert all(np.isfinite(values))


def test_camera_fit_keeps_silhouette_when_depth_candidate_is_not_projection_safe() -> None:
    camera = PinholeCamera(image_width=100, image_height=100)
    fit = fit_primitive(make_detection("box"), sample_points(), camera=camera)

    assert fit.primitive_label == "box"
    assert fit.fit_quality["selected_fit_mode"] == "camera_silhouette"
    assert fit.fit_quality["mode"] == "camera_silhouette"
    assert "depth_pca" in fit.fit_quality["candidate_scores"]


def test_box_fit_flags_cylinder_like_depth_profile() -> None:
    camera = PinholeCamera(image_width=100, image_height=100)
    fit = fit_primitive(make_detection("box"), sample_points(), camera=camera)

    assert fit.primitive_label_source == "detector"
    assert fit.to_dict()["primitive_label_source"] == "detector"
    assert fit.fit_quality["label_warning"] == "box_may_be_cylinder"


def test_plane_fit_prefers_depth_pca_for_tilted_plane() -> None:
    xs = np.linspace(-1.0, 1.0, 12)
    zs = np.linspace(-0.75, 0.75, 10)
    points = []
    for x in xs:
        for z in zs:
            y = 4.0 + 0.35 * x - 0.20 * z
            points.append((x, y, z))
    point_array = np.asarray(points, dtype=np.float64)
    camera = PinholeCamera(image_width=100, image_height=100)

    fit = fit_primitive(make_detection("plane"), point_array, camera=camera)
    rotation = np.asarray(fit.rotation_matrix, dtype=np.float64)
    expected_normal = np.array((-0.35, 1.0, 0.20), dtype=np.float64)
    expected_normal /= np.linalg.norm(expected_normal)

    assert fit.primitive_label == "plane"
    assert fit.fit_quality["selected_fit_mode"] == "depth_pca"
    assert fit.fit_quality["plane_extent_source"] == "visible_depth_pca_patch"
    assert fit.dimensions_xyz[2] <= 0.03
    assert abs(float(rotation[:, 2] @ expected_normal)) > 0.98
