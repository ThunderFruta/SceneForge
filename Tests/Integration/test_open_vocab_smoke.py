from __future__ import annotations

import subprocess
from pathlib import Path

from Tools.Integration.open_vocab_setup import prepare_layout
from Tools.Integration.open_vocab_smoke import build_command, run_smoke_test


SAM3_BUILDER_SOURCE = """def build_sam3_image_model():
    return object()
"""

SAM3_PROCESSOR_SOURCE = """class Sam3Processor:
    def set_image(self, image, state=None):
        return state or {}
    def set_text_prompt(self, prompt, state):
        return state
    def add_geometric_prompt(self, box, label, state):
        return state
"""

GROUNDINGDINO_SOURCE = """def load_model(*args, **kwargs):
    return object()
def load_image(*args, **kwargs):
    return None, None
def predict(*args, **kwargs):
    return [], [], []
"""


def fill_fake_repos(root: Path) -> None:
    sam3_repo = root / "SAM3" / "repo"
    (sam3_repo / "sam3" / "model").mkdir(parents=True)
    (sam3_repo / "sam3" / "__init__.py").write_text("", encoding="utf-8")
    (sam3_repo / "sam3" / "model" / "__init__.py").write_text("", encoding="utf-8")
    (sam3_repo / "sam3" / "model_builder.py").write_text(SAM3_BUILDER_SOURCE, encoding="utf-8")
    (sam3_repo / "sam3" / "model" / "sam3_image_processor.py").write_text(SAM3_PROCESSOR_SOURCE, encoding="utf-8")

    gdino_repo = root / "GroundingDINO" / "repo"
    weights = root / "GroundingDINO" / "weights"
    (gdino_repo / "groundingdino" / "util").mkdir(parents=True)
    (gdino_repo / "groundingdino" / "config").mkdir(parents=True)
    weights.mkdir(parents=True, exist_ok=True)
    (gdino_repo / "groundingdino" / "__init__.py").write_text("", encoding="utf-8")
    (gdino_repo / "groundingdino" / "util" / "__init__.py").write_text("", encoding="utf-8")
    (gdino_repo / "groundingdino" / "util" / "inference.py").write_text(GROUNDINGDINO_SOURCE, encoding="utf-8")
    (gdino_repo / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py").write_text("# fake", encoding="utf-8")
    (weights / "groundingdino_swint_ogc.pth").write_bytes(b"fake")


def test_build_command_uses_smoke_fixture(tmp_path: Path) -> None:
    command = build_command(root_dir=tmp_path / "OpenVocabulary")

    assert command[0:3] == ["python3", "run.py", "detect-shapes"]
    assert "Assets/Fixtures/OpenVocabulary/open_vocab_smoke_objects.png" in command


def test_smoke_test_stops_before_runner_when_not_ready(tmp_path: Path) -> None:
    called = False

    def runner(command):
        nonlocal called
        called = True
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    report = run_smoke_test(root_dir=tmp_path / "missing", output=tmp_path / "smoke.json", runner=runner)

    assert report["status"] == "not_ready"
    assert called is False
    assert (tmp_path / "smoke.json").is_file()


def test_smoke_test_runs_command_when_ready(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "OpenVocabulary"
    prepare_layout(root)
    fill_fake_repos(root)
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    captured = {}

    def runner(command):
        captured["command"] = list(command)
        return subprocess.CompletedProcess(command, 0, stdout="Wrote detections", stderr="")

    report = run_smoke_test(root_dir=root, output=tmp_path / "smoke.json", runner=runner)

    assert report["status"] == "passed"
    assert report["returncode"] == 0
    assert captured["command"][1:3] == ["run.py", "detect-shapes"]
    assert captured["command"][0].endswith("python") or "python" in captured["command"][0]
    assert "Assets/Fixtures/OpenVocabulary/open_vocab_smoke_objects.png" in captured["command"]
