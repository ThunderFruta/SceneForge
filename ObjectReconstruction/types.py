from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MeshResult:
    status: str
    path: Path | None
    reason: str | None = None
