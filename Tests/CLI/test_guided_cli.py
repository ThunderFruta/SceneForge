from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run_cli(args: list[str], stdin: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        input=stdin,
        text=True,
        capture_output=True,
        check=False,
    )


def test_run_py_no_args_shows_concise_core_guided_menu() -> None:
    result = run_cli(["run.py"], "4\n\nn\n")

    assert result.returncode == 0
    assert "SceneForge guided mode" in result.stdout
    assert "Process image" in result.stdout
    assert "placement fitting" in result.stdout
    assert "Construct empty room" in result.stdout
    assert "Generate empty-room image and VGGT background mesh" in result.stdout
    assert "Render .blend to PNG" in result.stdout
    assert "Complete object crops" in result.stdout
    assert "Reconstruct object meshes" in result.stdout
    assert "Fit and compose scene" in result.stdout
    assert "run-open-vocab-smoke" not in result.stdout
    assert "Check DINO/SAM readiness" not in result.stdout
    assert "Show command recipes" not in result.stdout


def test_run_py_no_args_does_not_execute_on_eof() -> None:
    result = run_cli(["run.py"], "")

    assert result.returncode == 0
    assert "SceneForge guided mode" in result.stdout
    assert "Equivalent command" in result.stdout
    assert "Running detection in an isolated process" not in result.stdout


def test_run_py_guided_render_can_print_command_without_running() -> None:
    result = run_cli(["run.py"], "3\n\n\n\n\nn\n")

    assert result.returncode == 0
    assert "render-blend-png" in result.stdout
    assert "Equivalent command" in result.stdout


def test_run_py_guided_reconstruct_defaults_to_textures_without_running() -> None:
    result = run_cli(["run.py"], "5\n\nn\n")

    assert result.returncode == 0
    assert "reconstruct-objects" in result.stdout
    assert "--with-texture" in result.stdout
    assert "--texture-resolution 512" in result.stdout
    assert "--texture-views 6" in result.stdout
    assert "--texture-reference-mode original" in result.stdout
    assert "--texture-remesh" in result.stdout


def test_run_py_guided_empty_room_can_print_command_without_running(monkeypatch) -> None:
    monkeypatch.setenv("SCENEFORGE_EMPTY_ROOM_BACKEND", "fake")
    result = run_cli(["run.py"], "2\n\n\n\n\nn\n")

    assert result.returncode == 0
    assert "construct-empty-room" in result.stdout
    assert "--detections Output/Latest/detect/detections.json" in result.stdout
    assert "--objects Output/Latest/objects" in result.stdout
    assert "--output Output/Latest/background" in result.stdout
    assert "--empty-room-backend fake" in result.stdout


def test_guided_process_image_includes_empty_room_when_run(monkeypatch) -> None:
    monkeypatch.setenv("SCENEFORGE_EMPTY_ROOM_BACKEND", "fake")
    result = run_cli(["run.py"], "1\n\n\nn\n")

    assert "process-image" in result.stdout
    assert "--empty-room-backend fake" in result.stdout
    assert "--background-fit room-corner" in result.stdout
    assert "--render-source-camera" in result.stdout


def test_guided_fit_and_compose_prints_fitting_commands_without_running() -> None:
    result = run_cli(["run.py"], "6\n\n\n\nn\n")

    assert result.returncode == 0
    assert "Equivalent commands:" in result.stdout
    assert "fit-vggt-boxes" in result.stdout
    assert "fit-empty-room-planes" in result.stdout
    assert "choose-object-supports" in result.stdout
    assert "build-object-fit-targets" in result.stdout
    assert "fit-object-placements" in result.stdout
    assert "compose-scene" in result.stdout
    assert "render-scene-camera-view" in result.stdout


def test_guided_fit_and_compose_autogenerates_missing_vggt(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SCENEFORGE_EMPTY_ROOM_BACKEND", "fake")
    monkeypatch.setenv("SCENEFORGE_VGGT_BACKEND", "fake")
    output_root = tmp_path / "Latest"
    result = run_cli(["run.py"], f"6\n{output_root}\n\n\nn\n")

    assert result.returncode == 0
    assert "detect-shapes" in result.stdout
    assert "run-vggt" in result.stdout
    assert "construct-empty-room" in result.stdout
    assert "--vggt Output/Latest/objects_vggt" not in result.stdout
    assert f"--vggt {output_root}/objects_vggt" in result.stdout


def test_explicit_run_py_command_bypasses_guided_mode() -> None:
    result = run_cli(["run.py", "prepare-open-vocab-layout", "--no-create-dirs", "--no-script"])

    assert "SceneForge guided mode" not in result.stdout
    assert result.returncode == 0


def test_integration_tool_no_args_enters_guided_default() -> None:
    result = run_cli(["Tools/Integration/open_vocab_readiness.py"], "n\n")

    assert result.returncode == 0
    assert "Audit DINO/SAM readiness" in result.stdout
    assert "open_vocab_readiness.py" in result.stdout


def test_view_blend_no_args_prints_guided_default_without_running() -> None:
    result = run_cli(["Tools/Scripts/view_blend.py"], "n\n")

    assert result.returncode == 0
    assert "Inspect a SceneForge .blend output" in result.stdout
    assert "fitted_scene.blend" not in result.stdout
