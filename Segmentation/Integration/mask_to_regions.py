from __future__ import annotations

from collections import deque
from math import floor

from Geometry.DepthValidity.depth_validity import DepthValidityConfig, is_depth_valid
from Geometry.Regions.region_analyzer import DepthRegion
from Segmentation.Core.segmentation_labels import PLANE_LABELS, SegmentationLabel
from Segmentation.Core.segmentation_mask import SegmentationMask


def segmentation_mask_to_regions(
    mask: SegmentationMask,
    depth_map: list[list[float]],
    *,
    analysis_columns: int,
    analysis_rows: int,
    min_valid_depth: float = 0.04,
    depth_invalid_mode: str = "black",
) -> list[DepthRegion]:
    _validate_depth_map(depth_map)
    mask.validate_size(width=len(depth_map[0]), height=len(depth_map))

    columns = max(2, min(analysis_columns, mask.width))
    rows = max(2, min(analysis_rows, mask.height))
    labels, averages, variances, valid = _downsample_mask_and_depth(
        mask,
        depth_map,
        columns,
        rows,
        DepthValidityConfig(min_valid_depth=min_valid_depth, invalid_mode=depth_invalid_mode),
    )

    regions: list[DepthRegion] = []
    visited: set[tuple[int, int]] = set()
    counters = {"plane": 0, "detail": 0}

    for row in range(rows):
        for column in range(columns):
            cell = (column, row)
            label = labels[row][column]
            if cell in visited or not valid[row][column]:
                continue
            component = _collect_component(
                start=cell,
                columns=columns,
                rows=rows,
                visited=visited,
                can_visit=lambda x, y: valid[y][x] and labels[y][x] == label,
            )
            kind = "plane" if label in PLANE_LABELS else "detail"
            name = f"{kind}_{counters[kind]:03d}"
            counters[kind] += 1
            regions.append(
                _make_region(
                    name=name,
                    kind=kind,
                    cells=component,
                    averages=averages,
                    variances=variances,
                )
            )

    return regions


def _downsample_mask_and_depth(
    mask: SegmentationMask,
    depth_map: list[list[float]],
    columns: int,
    rows: int,
    validity_config: DepthValidityConfig,
) -> tuple[
    list[list[SegmentationLabel]],
    list[list[float]],
    list[list[float]],
    list[list[bool]],
]:
    source_rows = len(depth_map)
    source_columns = len(depth_map[0])
    label_rows: list[list[SegmentationLabel]] = []
    averages: list[list[float]] = []
    variances: list[list[float]] = []
    valid: list[list[bool]] = []

    for row in range(rows):
        y0 = floor(row * source_rows / rows)
        y1 = max(y0 + 1, floor((row + 1) * source_rows / rows))
        label_row = []
        average_row = []
        variance_row = []
        valid_row = []
        for column in range(columns):
            x0 = floor(column * source_columns / columns)
            x1 = max(x0 + 1, floor((column + 1) * source_columns / columns))
            labels = [
                mask.label_at(source_x, source_y)
                for source_y in range(y0, min(y1, source_rows))
                for source_x in range(x0, min(x1, source_columns))
            ]
            depths = [
                depth_map[source_y][source_x]
                for source_y in range(y0, min(y1, source_rows))
                for source_x in range(x0, min(x1, source_columns))
            ]
            average = sum(depths) / len(depths)
            label = _majority_label(labels)
            label_row.append(label)
            average_row.append(average)
            variance_row.append(sum((depth - average) ** 2 for depth in depths) / len(depths))
            valid_row.append(is_depth_valid(average, validity_config))
        label_rows.append(label_row)
        averages.append(average_row)
        variances.append(variance_row)
        valid.append(valid_row)

    return label_rows, averages, variances, valid


def _majority_label(labels: list[SegmentationLabel]) -> SegmentationLabel:
    counts: dict[SegmentationLabel, int] = {}
    first_index: dict[SegmentationLabel, int] = {}
    for index, label in enumerate(labels):
        counts[label] = counts.get(label, 0) + 1
        first_index.setdefault(label, index)
    return max(counts, key=lambda label: (counts[label], -first_index[label]))


def _collect_component(
    *,
    start: tuple[int, int],
    columns: int,
    rows: int,
    visited: set[tuple[int, int]],
    can_visit,
) -> list[tuple[int, int]]:
    queue = deque([start])
    visited.add(start)
    cells = []

    while queue:
        column, row = queue.popleft()
        cells.append((column, row))
        for next_column, next_row in (
            (column - 1, row),
            (column + 1, row),
            (column, row - 1),
            (column, row + 1),
        ):
            next_cell = (next_column, next_row)
            if (
                next_column < 0
                or next_row < 0
                or next_column >= columns
                or next_row >= rows
                or next_cell in visited
                or not can_visit(next_column, next_row)
            ):
                continue
            visited.add(next_cell)
            queue.append(next_cell)

    return cells


def _make_region(
    *,
    name: str,
    kind: str,
    cells: list[tuple[int, int]],
    averages: list[list[float]],
    variances: list[list[float]],
) -> DepthRegion:
    min_column = min(column for column, _row in cells)
    max_column = max(column for column, _row in cells) + 1
    min_row = min(row for _column, row in cells)
    max_row = max(row for _column, row in cells) + 1
    depth_values = [averages[row][column] for column, row in cells]
    variance_values = [variances[row][column] for column, row in cells]
    return DepthRegion(
        name=name,
        kind=kind,
        cells=cells,
        bounds=(min_column, min_row, max_column, max_row),
        average_depth=sum(depth_values) / len(depth_values),
        variance=sum(variance_values) / len(variance_values),
    )


def _validate_depth_map(depth_map: list[list[float]]) -> None:
    if not depth_map or not depth_map[0]:
        raise ValueError("Depth map must contain at least one pixel.")
    width = len(depth_map[0])
    if any(len(row) != width for row in depth_map):
        raise ValueError("Depth map rows must all have the same width.")
