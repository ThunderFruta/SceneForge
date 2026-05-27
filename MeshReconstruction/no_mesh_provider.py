from __future__ import annotations

from pathlib import Path

from MeshReconstruction.types import MeshProvider, MeshResult


class NoMeshProvider(MeshProvider):
    backend = "none"
    model_dir = None

    def reconstruct(self, rgb_crop_path: Path, mask_path: Path, output_path: Path) -> MeshResult:
        del rgb_crop_path, mask_path, output_path
        return MeshResult(status="missing", path=None, reason="backend_none")

