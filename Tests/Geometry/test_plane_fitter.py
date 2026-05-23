from __future__ import annotations

import math

from Geometry.Planes.plane_fitter import fit_plane


def test_fit_plane_returns_none_for_too_few_points() -> None:
    assert fit_plane([]) is None
    assert fit_plane([(0.0, 0.0, 0.0)]) is None
    assert fit_plane([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]) is None


def test_fit_plane_axis_aligned_front_wall() -> None:
    # Points lying in y = -2 (front wall at depth 2 in our coord system).
    points = [
        (-1.0, -2.0, -1.0), (0.0, -2.0, -1.0), (1.0, -2.0, -1.0),
        (-1.0, -2.0,  0.0), (0.0, -2.0,  0.0), (1.0, -2.0,  0.0),
        (-1.0, -2.0,  1.0), (0.0, -2.0,  1.0), (1.0, -2.0,  1.0),
    ]
    result = fit_plane(points)
    assert result is not None
    centroid, normal = result
    # Normal must be (0, ±1, 0); orientation rule makes it (0, +1, 0).
    assert abs(normal[0]) < 1e-6
    assert abs(normal[2]) < 1e-6
    assert abs(normal[1] - 1.0) < 1e-6


def test_fit_plane_normal_faces_camera() -> None:
    # Camera is at origin; scene is in the -Y half-space.
    # Fitted normal should always point toward +Y (toward the camera).
    points = [
        (-1.0, -3.0, -1.0), (1.0, -3.0, -1.0),
        (-1.0, -3.0,  1.0), (1.0, -3.0,  1.0),
        ( 0.0, -3.0,  0.0),
    ]
    _, normal = fit_plane(points)
    # dot(normal, camera_direction_from_centroid) > 0
    assert normal[1] > 0


def test_fit_plane_tilted_surface() -> None:
    # Plane defined by z = -y - 1  (y + z = -1), a surface tilted in the Y-Z plane.
    # Tangent vectors: (1,0,0) and (0,-1,1).
    # Normal: (1,0,0) × (0,-1,1) = (0·1 - 0·(-1), 0·0 - 1·1, 1·(-1) - 0·0) = (0, -1, -1).
    # Normalized: (0, -1/√2, -1/√2).
    # Centroid is near (0, -1, 0) → camera_dir ~ (0,1,0) → dot with (0,-1/√2,-1/√2) < 0 → flip.
    # Oriented normal: (0, 1/√2, 1/√2).
    points = []
    for x in (-1.0, 0.0, 1.0):
        for z in (-1.0, 0.0, 1.0):
            y = -1.0 - z
            points.append((x, y, z))

    result = fit_plane(points)
    assert result is not None
    _, normal = result

    expected = 1.0 / math.sqrt(2)
    assert abs(normal[0]) < 1e-4
    assert abs(normal[1] - expected) < 1e-4
    assert abs(normal[2] - expected) < 1e-4


def test_fit_plane_centroid_is_mean_of_points() -> None:
    points = [
        (1.0, -1.0, 3.0),
        (3.0, -3.0, 1.0),
        (2.0, -2.0, 2.0),
    ]
    centroid, _ = fit_plane(points)
    assert abs(centroid[0] - 2.0) < 1e-10
    assert abs(centroid[1] - (-2.0)) < 1e-10
    assert abs(centroid[2] - 2.0) < 1e-10


def test_fit_plane_normal_is_unit_length() -> None:
    points = [
        (-1.0, -2.0, -1.0), (1.0, -2.0, -1.0),
        (-1.0, -2.0,  1.0), (1.0, -2.0,  1.0),
        ( 0.0, -2.0,  0.0),
    ]
    _, normal = fit_plane(points)
    length = math.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
    assert abs(length - 1.0) < 1e-6
