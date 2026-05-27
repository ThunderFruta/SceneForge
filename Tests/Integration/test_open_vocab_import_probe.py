from __future__ import annotations

import json
from pathlib import Path

from Tools.Integration.open_vocab_import_probe import build_report, main


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

BROKEN_SOURCE = """BROKEN = True
"""


def make_sam3_repo(root: Path, *, valid: bool = True) -> Path:
    repo = root / "SAM3" / "repo"
    (repo / "sam3" / "model").mkdir(parents=True)
    (repo / "sam3" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "sam3" / "model" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "sam3" / "model_builder.py").write_text(SAM3_BUILDER_SOURCE if valid else BROKEN_SOURCE, encoding="utf-8")
    (repo / "sam3" / "model" / "sam3_image_processor.py").write_text(SAM3_PROCESSOR_SOURCE if valid else BROKEN_SOURCE, encoding="utf-8")
    return repo


def make_groundingdino_repo(root: Path, *, valid: bool = True) -> Path:
    repo = root / "GroundingDINO" / "repo"
    (repo / "groundingdino" / "util").mkdir(parents=True)
    (repo / "groundingdino" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "groundingdino" / "util" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "groundingdino" / "util" / "inference.py").write_text(GROUNDINGDINO_SOURCE if valid else BROKEN_SOURCE, encoding="utf-8")
    return repo


def test_import_probe_reports_ready_for_fake_repo_apis(tmp_path: Path) -> None:
    sam3_repo = make_sam3_repo(tmp_path)
    gdino_repo = make_groundingdino_repo(tmp_path)

    report = build_report(
        backend="groundingdino-sam3",
        groundingdino_repo_dir=gdino_repo,
        sam3_repo_dir=sam3_repo,
    )

    assert report["ready"] is True
    assert all(check["ok"] for check in report["checks"])


def test_import_probe_reports_missing_symbols(tmp_path: Path) -> None:
    sam3_repo = make_sam3_repo(tmp_path, valid=False)
    gdino_repo = make_groundingdino_repo(tmp_path, valid=False)

    report = build_report(
        backend="groundingdino-sam3",
        groundingdino_repo_dir=gdino_repo,
        sam3_repo_dir=sam3_repo,
    )

    assert report["ready"] is False
    details = "\n".join(check["detail"] for check in report["checks"])
    assert "missing symbols" in details


def test_import_probe_cli_writes_report(tmp_path: Path) -> None:
    sam3_repo = make_sam3_repo(tmp_path)
    output = tmp_path / "probe.json"

    code = main(
        [
            "--backend",
            "sam3",
            "--sam3-repo-dir",
            str(sam3_repo),
            "--output",
            str(output),
        ]
    )

    assert code == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["ready"] is True
    assert data["backend"] == "sam3"
