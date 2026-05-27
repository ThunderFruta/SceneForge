from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class EdgeResult:
    image: Image.Image
    backend: str
    model_dir: Path | None = None


class EdgeProvider:
    backend: str
    model_dir: Path | None

    def detect_edges(self, image: Image.Image) -> EdgeResult:
        raise NotImplementedError
