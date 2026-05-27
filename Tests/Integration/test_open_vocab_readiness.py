from __future__ import annotations

from pathlib import Path

from Tools.Integration.open_vocab_readiness import build_report
from Tools.Integration.open_vocab_setup import prepare_layout


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


def test_readiness_reports_layout_not_prepared(tmp_path: Path) -> None:
    report = build_report(root_dir=tmp_path / "OpenVocabulary")

    assert report["status"] == "layout_not_prepared"
    assert report["ready_for_smoke_test"] is False
    assert report["next_steps"][0].startswith("Run prepare-open-vocab-layout")


def test_readiness_reports_ready_for_smoke_test_with_fake_repos(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "OpenVocabulary"
    prepare_layout(root)
    fill_fake_repos(root)
    monkeypatch.setenv("HF_TOKEN", "fake-token")

    report = build_report(root_dir=root)

    assert report["status"] == "ready_for_smoke_test"
    assert report["ready_for_smoke_test"] is True
    assert report["preflight"]["ready"] is True
    assert report["import_probe"]["ready"] is True
    assert report["sam3_access"]["ready"] is True
    assert report["first_smoke_test_command"][0:3] == ["python3", "run.py", "detect-shapes"]
    assert report["smoke_image_exists"] is True
    assert "Assets/Fixtures/OpenVocabulary/open_vocab_smoke_objects.png" in report["first_smoke_test_command"]


def test_readiness_reports_sam3_auth_required_when_sources_ready_without_token(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "OpenVocabulary"
    prepare_layout(root)
    fill_fake_repos(root)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)

    report = build_report(root_dir=root)

    assert report["status"] == "sam3_auth_required"
    assert report["ready_for_smoke_test"] is False
    assert report["sam3_access"]["status"] == "auth_or_cache_missing"
    assert report["next_steps"][0].startswith("Authenticate for gated SAM3 access")
