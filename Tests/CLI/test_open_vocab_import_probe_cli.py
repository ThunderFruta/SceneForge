from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


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


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "run.py"), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def make_sam3_repo(root: Path) -> Path:
    repo = root / "SAM3" / "repo"
    (repo / "sam3" / "model").mkdir(parents=True)
    (repo / "sam3" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "sam3" / "model" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "sam3" / "model_builder.py").write_text(SAM3_BUILDER_SOURCE, encoding="utf-8")
    (repo / "sam3" / "model" / "sam3_image_processor.py").write_text(SAM3_PROCESSOR_SOURCE, encoding="utf-8")
    return repo


def test_cli_probe_open_vocab_imports_writes_ready_report(tmp_path: Path) -> None:
    sam3_repo = make_sam3_repo(tmp_path)
    output = tmp_path / "probe.json"

    result = run_cli(
        "probe-open-vocab-imports",
        "--backend",
        "sam3",
        "--sam3-repo-dir",
        str(sam3_repo),
        "--output",
        str(output),
    )

    assert result.returncode == 0
    assert "open-vocab imports ready" in result.stdout
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["ready"] is True


def test_cli_probe_open_vocab_imports_missing_repo_fails_after_report(tmp_path: Path) -> None:
    output = tmp_path / "probe.json"

    result = run_cli(
        "probe-open-vocab-imports",
        "--backend",
        "sam3",
        "--sam3-repo-dir",
        str(tmp_path / "missing-sam3"),
        "--output",
        str(output),
    )

    assert result.returncode == 2
    assert output.is_file()
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["ready"] is False
    assert "Open-vocabulary imports are not ready" in result.stderr
