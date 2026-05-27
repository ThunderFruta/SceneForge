from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def default_blend_path() -> Path:
    return ROOT / "Assets" / "Samples" / "SyntheticNoPlane" / "synthetic_no_plane_01.blend"


def default_options(width: int, height: int) -> dict[str, Any]:
    return {
        "width": int(width),
        "height": int(height),
        "detector_backend": "depth-edge-object",
        "detector_model": "",
        "detector_weights": "",
        "edge_backend": "simple",
        "edge_model_dir": "",
        "mesh_backend": "none",
        "mesh_model_dir": "",
        "wireframe_backend": "none",
        "wireframe_model_dir": "",
        "device": "auto",
        "blender": "blender",
        "final_layout": "camera",
    }


def build_reconstruct_command(*, reference_blend: Path, output_dir: Path, options: dict[str, Any]) -> list[str]:
    command = [
        sys.executable,
        "run.py",
        "reconstruct-scene",
        "--reference-blend",
        str(reference_blend),
        "--output",
        str(output_dir),
        "--detector-backend",
        str(options.get("detector_backend", "depth-edge-object")),
        "--edge-backend",
        str(options.get("edge_backend", "simple")),
        "--mesh-backend",
        str(options.get("mesh_backend", "none")),
        "--wireframe-backend",
        str(options.get("wireframe_backend", "none")),
        "--width",
        str(options.get("width", 640)),
        "--height",
        str(options.get("height", 640)),
        "--device",
        str(options.get("device", "auto")),
        "--blender",
        str(options.get("blender", "blender")),
        "--final-layout",
        str(options.get("final_layout", "camera")),
    ]
    optional_flags = (
        ("detector_model", "--detector-model"),
        ("detector_weights", "--detector-weights"),
        ("edge_model_dir", "--edge-model-dir"),
        ("mesh_model_dir", "--mesh-model-dir"),
        ("wireframe_model_dir", "--wireframe-model-dir"),
    )
    for key, flag in optional_flags:
        value = options.get(key)
        if value:
            command.extend([flag, str(value)])
    return command


def main() -> int:
    options = default_options(640, 640)
    command = build_reconstruct_command(
        reference_blend=default_blend_path(),
        output_dir=Path("Output/Latest/generated"),
        options=options,
    )
    print(" ".join(command))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
