from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
import types

import numpy as np
from PIL import Image

from BackgroundReconstruction import empty_room
from BackgroundReconstruction.empty_room import generate_empty_room, read_openai_api_key_assignment


ROOT = Path(__file__).resolve().parents[2]


def run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "run.py"), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def write_fixture_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    image_path = tmp_path / "room.png"
    detections_path = tmp_path / "detections.json"
    objects_dir = tmp_path / "objects"
    objects_dir.mkdir()
    image = Image.new("RGB", (24, 16), (180, 170, 150))
    for x in range(8, 15):
        for y in range(5, 12):
            image.putpixel((x, y), (30, 40, 55))
    image.save(image_path)
    detections = {
        "image_path": str(image_path),
        "image_width": 24,
        "image_height": 16,
        "model_info": {},
        "objects": [
            {
                "id": 1,
                "bbox_xyxy": [8, 5, 15, 12],
                "mask_polygon": [[8, 5], [15, 5], [15, 12], [8, 12]],
                "detector_label": "chair",
                "detector_confidence": 0.9,
                "primitive_label": "unassigned",
                "primitive_confidence": 0.0,
                "primitive_label_source": "none",
            },
            {
                "id": 2,
                "bbox_xyxy": [0, 11, 24, 16],
                "mask_polygon": [[0, 11], [24, 11], [24, 16], [0, 16]],
                "detector_label": "floor",
                "detector_confidence": 0.8,
                "primitive_label": "unassigned",
                "primitive_confidence": 0.0,
                "primitive_label_source": "none",
            },
            {
                "id": 3,
                "bbox_xyxy": [1, 1, 5, 5],
                "mask_polygon": [[1, 1], [5, 1], [5, 5], [1, 5]],
                "detector_label": "plant",
                "detector_confidence": 0.7,
                "primitive_label": "unassigned",
                "primitive_confidence": 0.0,
                "primitive_label_source": "none",
                "mask_quality": "rectangular_fallback",
            },
        ],
    }
    detections_path.write_text(json.dumps(detections), encoding="utf-8")
    return image_path, detections_path, objects_dir


def test_generate_empty_room_fake_writes_mask_input_output_and_metadata(tmp_path: Path) -> None:
    image_path, detections_path, objects_dir = write_fixture_inputs(tmp_path)
    output_dir = tmp_path / "background"

    report = generate_empty_room(
        image_path=image_path,
        detections_path=detections_path,
        objects_dir=objects_dir,
        output_dir=output_dir,
        backend="fake",
        mask_dilation_px=0,
    )

    for name in (
        "empty_room.png",
        "empty_room_openai_input.png",
        "empty_room_edit_input.png",
        "empty_room_openai_mask.png",
        "empty_room_mask.png",
        "foreground_removal_mask.png",
        "empty_room_metadata.json",
    ):
        assert (output_dir / name).is_file()
    assert report["selected_removed_detection_ids"] == [1]
    assert {item["id"]: item["reason"] for item in report["excluded_detection_ids"]} == {
        2: "protected_structural_label",
        3: "rectangular_fallback_mask",
    }
    assert report["mask_quality_counts"] == {"rectangular_fallback": 1, "unspecified": 2}
    assert report["resolution_framing_preserved"] is True
    assert Image.open(output_dir / "empty_room.png").size == (24, 16)
    mask = np.asarray(Image.open(output_dir / "empty_room_mask.png"), dtype=np.uint8)
    design_mask = np.asarray(Image.open(output_dir / "foreground_removal_mask.png"), dtype=np.uint8)
    assert np.array_equal(mask, design_mask)
    assert report["foreground_removal_mask_path"].endswith("foreground_removal_mask.png")
    assert report["empty_room_edit_input_path"].endswith("empty_room_edit_input.png")
    assert mask[6, 9] == 255
    assert mask[13, 3] == 0
    openai_mask = Image.open(output_dir / "empty_room_openai_mask.png").convert("RGBA")
    assert openai_mask.getpixel((9, 6))[3] == 0
    assert openai_mask.getpixel((3, 13))[3] == 255


def test_generate_empty_room_cli_accepts_fake_backend(tmp_path: Path) -> None:
    image_path, detections_path, objects_dir = write_fixture_inputs(tmp_path)
    output_dir = tmp_path / "background"

    result = run_cli(
        [
            "generate-empty-room",
            "--empty-room-backend",
            "fake",
            "--image",
            str(image_path),
            "--detections",
            str(detections_path),
            "--objects",
            str(objects_dir),
            "--output",
            str(output_dir),
            "--mask-dilation-px",
            "0",
        ]
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "empty_room_metadata.json").is_file()
    assert "Wrote" in result.stdout


def test_openai_empty_room_call_sends_mask_file(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "input.png"
    mask_path = tmp_path / "mask.png"
    result_image = Image.new("RGB", (4, 4), (11, 22, 33))
    input_path.write_bytes(png_bytes(Image.new("RGBA", (4, 4), (1, 2, 3, 255))))
    mask_path.write_bytes(png_bytes(Image.new("RGBA", (4, 4), (255, 255, 255, 0))))
    calls = []

    class FakeImages:
        def edit(self, **kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(data=[types.SimpleNamespace(b64_json=empty_room.base64.b64encode(png_bytes(result_image)).decode("ascii"))])

    class FakeOpenAI:
        def __init__(self):
            self.images = FakeImages()

    monkeypatch.setenv("OPENAI_API_KEY", "present-but-not-real")
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))

    image = empty_room.call_openai_empty_room_edit(
        input_path=input_path,
        mask_path=mask_path,
        prompt="empty room",
        model="gpt-image-1.5",
    )

    assert image.size == (4, 4)
    assert len(calls) == 1
    assert calls[0]["model"] == "gpt-image-1.5"
    assert calls[0]["mask"].name == str(mask_path)
    assert calls[0]["image"].name == str(input_path)
    assert calls[0]["background"] == "opaque"


def test_read_openai_key_from_shell_config_without_executing_file(tmp_path: Path) -> None:
    shell_config = tmp_path / ".bashrc"
    shell_config.write_text(
        "return 0\n"
        "export OPENAI_API_KEY='test-secret-value'\n",
        encoding="utf-8",
    )

    assert read_openai_api_key_assignment(shell_config) == "test-secret-value"


def test_run_empty_room_vggt_cli_writes_glb_with_fake_backends(tmp_path: Path) -> None:
    image_path, detections_path, objects_dir = write_fixture_inputs(tmp_path)
    output_dir = tmp_path / "background"

    result = run_cli(
        [
            "run-empty-room-vggt",
            "--empty-room-backend",
            "fake",
            "--vggt-backend",
            "fake",
            "--image",
            str(image_path),
            "--detections",
            str(detections_path),
            "--objects",
            str(objects_dir),
            "--output",
            str(output_dir),
            "--mask-dilation-px",
            "0",
            "--obj-stride",
            "4",
        ]
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "empty_room.png").is_file()
    assert (output_dir / "empty_room_mesh.glb").is_file()
    assert (output_dir / "vggt_geometry.json").is_file()
    report = json.loads((output_dir / "vggt_geometry.json").read_text(encoding="utf-8"))
    assert report["artifacts"]["mesh_glb"] == "empty_room_mesh.glb"


def test_construct_empty_room_alias_runs_combined_pipeline(tmp_path: Path) -> None:
    image_path, detections_path, objects_dir = write_fixture_inputs(tmp_path)
    output_dir = tmp_path / "background"

    result = run_cli(
        [
            "construct-empty-room",
            "--empty-room-backend",
            "fake",
            "--vggt-backend",
            "fake",
            "--image",
            str(image_path),
            "--detections",
            str(detections_path),
            "--objects",
            str(objects_dir),
            "--output",
            str(output_dir),
            "--mask-dilation-px",
            "0",
            "--obj-stride",
            "4",
        ]
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "empty_room.png").is_file()
    assert (output_dir / "empty_room_mesh.glb").is_file()
    assert (output_dir / "empty_room_metadata.json").is_file()


def png_bytes(image: Image.Image) -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
