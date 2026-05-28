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


def test_run_py_no_args_can_show_recipes_without_heavy_imports() -> None:
    result = run_cli(["run.py"], "6
")

    assert result.returncode == 0
    assert "SceneForge guided mode" in result.stdout
    assert "run-open-vocab-smoke" in result.stdout


def test_run_py_guided_readiness_can_print_command_without_running() -> None:
    result = run_cli(["run.py"], "3

n
")

    assert result.returncode == 0
    assert "audit-open-vocab-readiness" in result.stdout
    assert "Equivalent command" in result.stdout


def test_explicit_run_py_command_bypasses_guided_mode() -> None:
    result = run_cli(["run.py", "prepare-open-vocab-layout", "--no-create-dirs", "--no-script"])

    assert "SceneForge guided mode" not in result.stdout
    assert result.returncode == 0


def test_integration_tool_no_args_enters_guided_default() -> None:
    result = run_cli(["Tools/Integration/open_vocab_readiness.py"], "n
")

    assert result.returncode == 0
    assert "Audit DINO/SAM readiness" in result.stdout
    assert "open_vocab_readiness.py" in result.stdout


def test_view_blend_no_args_prints_guided_default_without_running() -> None:
    result = run_cli(["Tools/Scripts/view_blend.py"], "n
")

    assert result.returncode == 0
    assert "Inspect a SceneForge .blend output" in result.stdout
    assert "--blend Output/Latest/fitted_scene.blend" in result.stdout
