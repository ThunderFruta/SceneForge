from __future__ import annotations

from Geometry.Regions.region_analyzer import analyze_depth_regions


def test_analyze_depth_regions_detects_one_flat_plane() -> None:
    depth = [[0.4 for _column in range(4)] for _row in range(4)]

    regions = analyze_depth_regions(
        depth,
        analysis_columns=4,
        analysis_rows=4,
        min_plane_cells=4,
    )

    assert [region.kind for region in regions] == ["plane"]
    assert regions[0].bounds == (0, 0, 4, 4)


def test_analyze_depth_regions_detects_two_flat_planes() -> None:
    depth = [[0.2, 0.2, 0.8, 0.8] for _row in range(4)]

    regions = analyze_depth_regions(
        depth,
        analysis_columns=4,
        analysis_rows=4,
        min_plane_cells=4,
    )

    plane_regions = [region for region in regions if region.kind == "plane"]
    assert len(plane_regions) == 2
    assert [region.bounds for region in plane_regions] == [(0, 0, 2, 4), (2, 0, 4, 4)]


def test_analyze_depth_regions_keeps_small_noisy_area_as_detail() -> None:
    depth = [[0.4 for _column in range(5)] for _row in range(5)]
    depth[2][2] = 0.9

    regions = analyze_depth_regions(
        depth,
        analysis_columns=5,
        analysis_rows=5,
        min_plane_cells=4,
    )

    assert any(region.kind == "plane" for region in regions)
    assert any(region.kind == "detail" and region.bounds == (2, 2, 3, 3) for region in regions)


def test_analyze_depth_regions_rejects_tiny_plane_region() -> None:
    depth = [[0.5, 0.5], [0.5, 0.5]]

    regions = analyze_depth_regions(
        depth,
        analysis_columns=2,
        analysis_rows=2,
        min_plane_cells=5,
    )

    assert [region.kind for region in regions] == ["detail"]


def test_analyze_depth_regions_rejects_thin_small_plane_region() -> None:
    depth = [[0.5 for _column in range(8)], [0.0 for _column in range(8)]]

    regions = analyze_depth_regions(
        depth,
        analysis_columns=8,
        analysis_rows=2,
        min_plane_cells=4,
        min_thin_plane_cells=16,
    )

    assert [region.kind for region in regions] == ["detail"]
