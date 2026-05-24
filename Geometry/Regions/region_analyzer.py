from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import floor

from Geometry.DepthValidity.depth_validity import DepthValidityConfig, is_depth_valid


@dataclass(frozen=True)
class DepthRegion:
    name: str
    kind: str
    cells: list[tuple[int, int]]
    bounds: tuple[int, int, int, int]
    average_depth: float
    variance: float


def analyze_depth_regions(
    depth_map: list[list[float]],
    *,
    analysis_columns: int = 32,
    analysis_rows: int | None = None,
    flat_variance_threshold: float = 0.0008,
    depth_bucket_size: float = 0.08,
    min_plane_cells: int = 12,
    min_thin_plane_cells: int = 64,
    min_valid_depth: float = 0.04,
    depth_invalid_mode: str = "black",
) -> list[DepthRegion]:
    _validate_depth_map(depth_map)
    source_rows = len(depth_map)
    source_columns = len(depth_map[0])
    columns = max(2, min(analysis_columns, source_columns))
    rows = analysis_rows
    if rows is None:
        rows = max(2, round(columns * source_rows / source_columns))
    rows = max(2, min(rows, source_rows))

    averages, variances = _downsample_depth(depth_map, columns, rows)
    validity_config = DepthValidityConfig(
        min_valid_depth=min_valid_depth,
        invalid_mode=depth_invalid_mode,
    )
    planar = [
        [
            is_depth_valid(averages[row][column], validity_config)
            and variances[row][column] <= flat_variance_threshold
            for column in range(columns)
        ]
        for row in range(rows)
    ]
    valid = [
        [is_depth_valid(averages[row][column], validity_config) for column in range(columns)]
        for row in range(rows)
    ]
    buckets = [
        [
            floor(averages[row][column] / depth_bucket_size)
            for column in range(columns)
        ]
        for row in range(rows)
    ]

    regions: list[DepthRegion] = []
    plane_cells: set[tuple[int, int]] = set()
    visited: set[tuple[int, int]] = set()

    for row in range(rows):
        for column in range(columns):
            cell = (column, row)
            if cell in visited or not planar[row][column]:
                continue
            component = _collect_component(
                start=cell,
                columns=columns,
                rows=rows,
                visited=visited,
                can_visit=lambda x, y: planar[y][x] and buckets[y][x] == buckets[row][column],
            )
            if _accept_plane_component(
                component,
                min_plane_cells=min_plane_cells,
                min_thin_plane_cells=min_thin_plane_cells,
            ):
                plane_cells.update(component)
                regions.append(
                    _make_region(
                        name=f"plane_{len(regions):03d}",
                        kind="plane",
                        cells=component,
                        averages=averages,
                        variances=variances,
                    )
                )

    detail_visited = set(plane_cells)
    detail_index = 0
    for row in range(rows):
        for column in range(columns):
            cell = (column, row)
            if cell in detail_visited or not valid[row][column]:
                continue
            component = _collect_component(
                start=cell,
                columns=columns,
                rows=rows,
                visited=detail_visited,
                can_visit=lambda x, y: valid[y][x],
            )
            regions.append(
                _make_region(
                    name=f"detail_{detail_index:03d}",
                    kind="detail",
                    cells=component,
                    averages=averages,
                    variances=variances,
                )
            )
            detail_index += 1

    return regions


def _accept_plane_component(
    component: list[tuple[int, int]],
    *,
    min_plane_cells: int,
    min_thin_plane_cells: int,
) -> bool:
    if len(component) < min_plane_cells:
        return False

    columns = [column for column, _row in component]
    rows = [row for _column, row in component]
    width = max(columns) - min(columns) + 1
    height = max(rows) - min(rows) + 1
    if (width == 1 or height == 1) and len(component) < min_thin_plane_cells:
        return False

    return True


def _validate_depth_map(depth_map: list[list[float]]) -> None:
    if not depth_map or not depth_map[0]:
        raise ValueError("Depth map must contain at least one pixel.")
    width = len(depth_map[0])
    if any(len(row) != width for row in depth_map):
        raise ValueError("Depth map rows must all have the same width.")


def _downsample_depth(
    depth_map: list[list[float]],
    columns: int,
    rows: int,
) -> tuple[list[list[float]], list[list[float]]]:
    source_rows = len(depth_map)
    source_columns = len(depth_map[0])
    averages: list[list[float]] = []
    variances: list[list[float]] = []

    for row in range(rows):
        y0 = floor(row * source_rows / rows)
        y1 = max(y0 + 1, floor((row + 1) * source_rows / rows))
        average_row = []
        variance_row = []
        for column in range(columns):
            x0 = floor(column * source_columns / columns)
            x1 = max(x0 + 1, floor((column + 1) * source_columns / columns))
            values = [
                depth_map[source_y][source_x]
                for source_y in range(y0, min(y1, source_rows))
                for source_x in range(x0, min(x1, source_columns))
            ]
            average = sum(values) / len(values)
            variance = sum((value - average) ** 2 for value in values) / len(values)
            average_row.append(average)
            variance_row.append(variance)
        averages.append(average_row)
        variances.append(variance_row)

    return averages, variances


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
