from __future__ import annotations

from Core.Types.scene_data import SceneMeshPart
from Geometry.Cleanup.spike_filter import filter_spike_faces


def test_filter_spike_faces_removes_long_skinny_horn_triangle() -> None:
    part = SceneMeshPart(
        name="detail_000",
        kind="detail",
        vertices=[
            (0.0, 1.0, 0.0),
            (0.01, 1.0, 0.0),
            (0.0, 2.0, 0.0),
        ],
        faces=[(0, 1, 2)],
        uvs=[(0.0, 0.0), (0.1, 0.0), (0.0, 0.1)],
    )

    filtered, rejected = filter_spike_faces(part, threshold="balanced")

    assert rejected == 1
    assert filtered.faces == []
    assert filtered.normals is None


def test_filter_spike_faces_keeps_normal_plane_triangle() -> None:
    part = SceneMeshPart(
        name="plane_000",
        kind="plane",
        vertices=[
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 1.0),
        ],
        faces=[(0, 1, 2)],
        uvs=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
    )

    filtered, rejected = filter_spike_faces(part, threshold="balanced")

    assert rejected == 0
    assert filtered is part
