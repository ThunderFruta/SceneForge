from __future__ import annotations

from pathlib import Path

from generate import build_reconstruct_command, default_blend_path, default_options


def test_generate_defaults_to_depth_edge_object_backend() -> None:
    options = default_options(1280, 720)

    assert options["detector_backend"] == "depth-edge-object"
    assert options["detector_model"] == ""


def test_generate_defaults_to_current_synthetic_no_plane_sample() -> None:
    path = default_blend_path()

    assert path.name == "synthetic_no_plane_01.blend"
    assert path.parent.name == "SyntheticNoPlane"


def test_generate_reconstruct_command_includes_detector_backend() -> None:
    options = default_options(1280, 720)
    command = build_reconstruct_command(
        reference_blend=Path("Assets/Samples/SyntheticNoPlane/synthetic_no_plane_01.blend"),
        output_dir=Path("Output/Latest/generated"),
        options=options,
    )

    backend_index = command.index("--detector-backend") + 1
    assert command[backend_index] == "depth-edge-object"
    assert "--detector-model" not in command


def test_generate_reconstruct_command_includes_detector_model_when_set() -> None:
    options = default_options(1280, 720)
    options["detector_model"] = "Models/InstanceDetector/demo/primitive_3d_segmenter.pt"
    command = build_reconstruct_command(
        reference_blend=Path("Assets/Samples/SyntheticNoPlane/synthetic_no_plane_01.blend"),
        output_dir=Path("Output/Latest/generated"),
        options=options,
    )

    model_index = command.index("--detector-model") + 1
    assert command[model_index] == "Models/InstanceDetector/demo/primitive_3d_segmenter.pt"
