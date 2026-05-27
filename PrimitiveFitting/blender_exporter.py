from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class BlenderExportError(RuntimeError):
    pass


def export_fit_report_to_blend(
    report_path: str | Path,
    output_path: str | Path,
    blender_executable: str = "blender",
    layout: str = "camera",
    reference_blend_path: str | Path | None = None,
) -> None:
    report = Path(report_path)
    output = Path(output_path)
    if shutil.which(blender_executable) is None:
        raise BlenderExportError(f"Blender executable was not found: {blender_executable}")
    if layout not in {"camera", "ground", "original-camera"}:
        raise BlenderExportError(f"Unsupported Blender export layout: {layout}")

    script_path = Path(__file__).resolve().parents[1] / "Tools" / "Scripts" / "export_fitted_primitives_blend.py"
    command = [
        blender_executable,
        "-b",
        "--python",
        str(script_path),
        "--",
        "--fits",
        str(report),
        "--output",
        str(output),
        "--layout",
        layout,
    ]
    if reference_blend_path is not None:
        command.extend(["--reference-blend", str(reference_blend_path)])
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise BlenderExportError(
            f"Blender export failed with exit code {result.returncode}: {result.stderr.strip()}"
        )
