from __future__ import annotations

from pathlib import Path

from generate import build_reconstruct_command, default_blend_path, default_options


def test_generate_defaults_to_open_vocab_proposals() -> None:
    options = default_options(1280, 720)

    assert options["detector_backend"] == "groundingdino-sam3"
    assert options["detector_model"] == ""


def test_generate_defaults_to_current_synthetic_no_plane_sample() -> None:
    path = default_blend_path()

    assert path.name == "synthetic_no_plane_01.blend"
    assert path.parent.name == "SyntheticNoPlane"


def test_generate_command_renders_preview_instead_of_reconstructing() -> None:
    options = default_options(1280, 720)
    command = build_reconstruct_command(
        reference_blend=Path("Assets/Samples/SyntheticNoPlane/synthetic_no_plane_01.blend"),
        output_dir=Path("Output/Latest/generated"),
        options=options,
    )

    assert command[1] == "run.py"
    assert command[2] == "render-blend-png"
    assert "--detector-backend" not in command


def test_generate_preview_command_ignores_detector_model_option() -> None:
    options = default_options(1280, 720)
    options["detector_model"] = "Models/Unused/demo/primitive_3d_segmenter.pt"
    command = build_reconstruct_command(
        reference_blend=Path("Assets/Samples/SyntheticNoPlane/synthetic_no_plane_01.blend"),
        output_dir=Path("Output/Latest/generated"),
        options=options,
    )

    assert "--detector-model" not in command
    assert command[2] == "render-blend-png"
