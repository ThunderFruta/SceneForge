from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WireframeLine:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float

    def to_list(self) -> list[float]:
        return [
            round(float(self.x1), 3),
            round(float(self.y1), 3),
            round(float(self.x2), 3),
            round(float(self.y2), 3),
            round(float(self.score), 6),
        ]


@dataclass(frozen=True)
class WireframeResult:
    status: str
    lines: list[WireframeLine]
    junction_count: int
    json_path: Path | None = None
    overlay_path: Path | None = None
    reason: str | None = None

    @property
    def line_count(self) -> int:
        return len(self.lines)


class WireframeProvider:
    backend: str
    model_dir: Path | None

    def detect_wireframe(
        self,
        rgb_crop_path: Path,
        mask_path: Path,
        output_json_path: Path,
        output_overlay_path: Path,
    ) -> WireframeResult:
        raise NotImplementedError


class NoWireframeProvider(WireframeProvider):
    backend = "none"
    model_dir = None

    def detect_wireframe(
        self,
        rgb_crop_path: Path,
        mask_path: Path,
        output_json_path: Path,
        output_overlay_path: Path,
    ) -> WireframeResult:
        del rgb_crop_path, mask_path, output_json_path, output_overlay_path
        return WireframeResult(status="not_available", lines=[], junction_count=0, reason="backend_none")
