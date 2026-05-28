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
        "detector_backend": "groundingdino-sam3",
        "open_vocab_root": "Models/OpenVocabulary",
        "text_prompt_preset": "scene-primitives-v1",
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
        str(options.get("detector_backend", "groundingdino-sam3")),
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
        "--force",
    ]
    if options.get("open_vocab_root"):
        command.extend(["--open-vocab-root", str(options["open_vocab_root"])])
    if options.get("text_prompt_preset"):
        command.extend(["--text-prompt-preset", str(options["text_prompt_preset"])])
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
    from Runtime.guided_cli import run_after_confirmation

    options = default_options(640, 640)
    command = build_reconstruct_command(
        reference_blend=default_blend_path(),
        output_dir=Path("Output/Latest/generated"),
        options=options,
    )
    print("SceneForge guided generator")
    return run_after_confirmation(command)


if __name__ == "__main__":
    raise SystemExit(main())
