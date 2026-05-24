from __future__ import annotations

from Geometry.DepthValidity.depth_validity import (
    DepthValidityConfig,
    build_depth_validity_metrics,
    is_depth_valid,
)


def test_black_mode_preserves_near_black_far_depth() -> None:
    config = DepthValidityConfig(min_valid_depth=0.04, invalid_mode="black")

    assert not is_depth_valid(0.0, config)
    assert is_depth_valid(0.01, config)


def test_threshold_mode_discards_near_black_depth() -> None:
    config = DepthValidityConfig(min_valid_depth=0.04, invalid_mode="threshold")

    assert not is_depth_valid(0.0, config)
    assert not is_depth_valid(0.01, config)
    assert is_depth_valid(0.04, config)


def test_depth_validity_metrics_separate_recovered_far_depth() -> None:
    metrics = build_depth_validity_metrics(
        [[0.0, 0.01, 0.5]],
        DepthValidityConfig(min_valid_depth=0.04, invalid_mode="black"),
    )

    assert metrics["invalid_depth_cells"] == 1
    assert metrics["recovered_far_depth_cells"] == 1
    assert metrics["discarded_near_black_cells"] == 0
