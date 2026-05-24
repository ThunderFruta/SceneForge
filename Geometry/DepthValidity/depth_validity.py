from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


DepthInvalidMode = Literal["black", "threshold", "none"]


@dataclass(frozen=True)
class DepthValidityConfig:
    min_valid_depth: float = 0.04
    invalid_mode: DepthInvalidMode = "black"


def is_depth_valid(depth: float, config: DepthValidityConfig) -> bool:
    if config.invalid_mode == "none":
        return True
    if config.invalid_mode == "black":
        return depth > 0.0
    if config.invalid_mode == "threshold":
        return depth >= config.min_valid_depth
    raise ValueError(f"Unsupported depth invalid mode: {config.invalid_mode}")


def build_depth_validity_metrics(
    depth_map: list[list[float]],
    config: DepthValidityConfig,
) -> dict[str, int]:
    exact_black = 0
    near_black = 0
    invalid = 0
    for row in depth_map:
        for depth in row:
            if depth <= 0.0:
                exact_black += 1
            elif depth < config.min_valid_depth:
                near_black += 1
            if not is_depth_valid(depth, config):
                invalid += 1

    recovered_far = near_black if config.invalid_mode in {"black", "none"} else 0
    discarded_near_black = near_black if config.invalid_mode == "threshold" else 0
    return {
        "invalid_depth_cells": invalid,
        "recovered_far_depth_cells": recovered_far,
        "discarded_near_black_cells": discarded_near_black,
        "exact_black_depth_cells": exact_black,
        "near_black_depth_cells": near_black,
    }
