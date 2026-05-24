from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from Segmentation.Core.segmentation_labels import SegmentationLabel
from Segmentation.Core.segmentation_mask import SegmentationMask


@dataclass(frozen=True)
class MaskCleanupResult:
    mask: SegmentationMask
    filled_mask_holes: int
    removed_mask_islands: int


def cleanup_segmentation_mask(
    mask: SegmentationMask,
    *,
    max_hole_cells: int = 12,
    max_island_cells: int = 3,
) -> MaskCleanupResult:
    labels = [list(row) for row in mask.labels]
    removed = _remove_tiny_islands(labels, max_island_cells=max_island_cells)
    filled = _fill_small_unknown_holes(labels, max_hole_cells=max_hole_cells)
    return MaskCleanupResult(
        mask=SegmentationMask.from_labels(labels),
        filled_mask_holes=filled,
        removed_mask_islands=removed,
    )


def _remove_tiny_islands(
    labels: list[list[SegmentationLabel]],
    *,
    max_island_cells: int,
) -> int:
    if max_island_cells <= 0:
        return 0

    height = len(labels)
    width = len(labels[0])
    visited: set[tuple[int, int]] = set()
    removed = 0

    for row in range(height):
        for column in range(width):
            label = labels[row][column]
            if label == SegmentationLabel.UNKNOWN or (column, row) in visited:
                continue
            component = _collect_component(
                column,
                row,
                labels,
                visited,
                target_label=label,
            )
            if len(component) > max_island_cells or _touches_border(component, width, height):
                continue
            neighbor_label = _dominant_neighbor_label(labels, component, exclude=label)
            if neighbor_label is None:
                continue
            for x, y in component:
                labels[y][x] = neighbor_label
            removed += 1

    return removed


def _fill_small_unknown_holes(
    labels: list[list[SegmentationLabel]],
    *,
    max_hole_cells: int,
) -> int:
    if max_hole_cells <= 0:
        return 0

    height = len(labels)
    width = len(labels[0])
    visited: set[tuple[int, int]] = set()
    filled = 0

    for row in range(height):
        for column in range(width):
            if labels[row][column] != SegmentationLabel.UNKNOWN or (column, row) in visited:
                continue
            component = _collect_component(
                column,
                row,
                labels,
                visited,
                target_label=SegmentationLabel.UNKNOWN,
            )
            if len(component) > max_hole_cells or _touches_border(component, width, height):
                continue
            fill_label = _dominant_neighbor_label(labels, component, exclude=SegmentationLabel.UNKNOWN)
            if fill_label is None:
                continue
            for x, y in component:
                labels[y][x] = fill_label
            filled += 1

    return filled


def _collect_component(
    column: int,
    row: int,
    labels: list[list[SegmentationLabel]],
    visited: set[tuple[int, int]],
    *,
    target_label: SegmentationLabel,
) -> list[tuple[int, int]]:
    height = len(labels)
    width = len(labels[0])
    queue = deque([(column, row)])
    visited.add((column, row))
    component = []

    while queue:
        x, y = queue.popleft()
        component.append((x, y))
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if (
                nx < 0
                or ny < 0
                or nx >= width
                or ny >= height
                or (nx, ny) in visited
                or labels[ny][nx] != target_label
            ):
                continue
            visited.add((nx, ny))
            queue.append((nx, ny))

    return component


def _dominant_neighbor_label(
    labels: list[list[SegmentationLabel]],
    component: list[tuple[int, int]],
    *,
    exclude: SegmentationLabel,
) -> SegmentationLabel | None:
    height = len(labels)
    width = len(labels[0])
    cells = set(component)
    counts: dict[SegmentationLabel, int] = {}
    first_seen: dict[SegmentationLabel, int] = {}
    order = 0

    for x, y in component:
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if nx < 0 or ny < 0 or nx >= width or ny >= height or (nx, ny) in cells:
                continue
            label = labels[ny][nx]
            if label == exclude or label == SegmentationLabel.UNKNOWN:
                continue
            counts[label] = counts.get(label, 0) + 1
            first_seen.setdefault(label, order)
            order += 1

    if not counts:
        return None
    return max(counts, key=lambda label: (counts[label], -first_seen[label]))


def _touches_border(component: list[tuple[int, int]], width: int, height: int) -> bool:
    return any(x == 0 or y == 0 or x == width - 1 or y == height - 1 for x, y in component)
