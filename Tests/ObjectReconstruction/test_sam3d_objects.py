from __future__ import annotations

from pathlib import Path

from PIL import Image

from ObjectReconstruction.sam3d_objects import INPUT_NAME, MASK_NAME, build_command, run_sam3d_objects_reconstruction


def test_sam3d_objects_backend_prepares_inputs_without_required_dependency(tmp_path: Path) -> None:
    objects_dir = tmp_path / "objects"
    object_dir = objects_dir / "01_chair"
    object_dir.mkdir(parents=True)
    Image.new("RGBA", (32, 32), (240, 240, 240, 255)).save(object_dir / "completed_crop.png")

    manifest = run_sam3d_objects_reconstruction(objects_dir)

    assert manifest["backend"] == "sam3d-objects"
    assert manifest["objects"][0]["status"] == "skipped"
    assert manifest["objects"][0]["reason"] == "sam3d_objects_command_required"
    assert (object_dir / INPUT_NAME).is_file()
    assert (object_dir / MASK_NAME).is_file()
    assert (object_dir / "sam3d_objects_metadata.json").is_file()
    assert (objects_dir / "sam3d_objects_manifest.json").is_file()


def test_sam3d_objects_command_template_expands_paths(tmp_path: Path) -> None:
    command = build_command(
        "python run.py --image {image} --mask {mask} --output {output} --device {device}",
        object_dir=tmp_path / "object",
        input_path=tmp_path / "input.png",
        mask_path=tmp_path / "mask.png",
        output_mesh_path=tmp_path / "mesh.glb",
        repo_dir=tmp_path / "repo",
        checkpoint=tmp_path / "checkpoint.pt",
        device="cpu",
    )

    assert command == [
        "python",
        "run.py",
        "--image",
        str(tmp_path / "input.png"),
        "--mask",
        str(tmp_path / "mask.png"),
        "--output",
        str(tmp_path / "mesh.glb"),
        "--device",
        "cpu",
    ]
