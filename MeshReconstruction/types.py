from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MeshResult:
    status: str
    path: Path | None
    reason: str | None = None


class MeshProvider:
    backend: str
    model_dir: Path | None

    def reconstruct(self, rgb_crop_path: Path, mask_path: Path, output_path: Path) -> MeshResult:
        raise NotImplementedError
