from __future__ import annotations

from pathlib import Path

from Tools.Integration.open_vocab_setup import (
    GROUNDINGDINO_CHECKPOINT_URL,
    GROUNDINGDINO_REPO_URL,
    SAM3_REPO_URL,
    prepare_layout,
)


def test_prepare_layout_creates_manifest_and_setup_script(tmp_path: Path) -> None:
    root = tmp_path / "OpenVocabulary"

    manifest = prepare_layout(root)

    assert (root / "GroundingDINO" / "weights").is_dir()
    assert (root / "SAM3" / "hf").is_dir()
    assert (root / "open_vocab_setup_manifest.json").is_file()
    script = root / "setup_open_vocab_sources.sh"
    assert script.is_file()
    text = script.read_text(encoding="utf-8")
    assert GROUNDINGDINO_REPO_URL in text
    assert GROUNDINGDINO_CHECKPOINT_URL in text
    assert SAM3_REPO_URL in text
    assert "hf auth login" in text
    assert manifest["paths"]["groundingdino_checkpoint"].endswith("groundingdino_swint_ogc.pth")
    assert manifest["workflow"][0].startswith("Run setup_open_vocab_sources")
    assert manifest["paths"]["smoke_image_path"].endswith("open_vocab_smoke_objects.png")
    assert "Assets/Fixtures/OpenVocabulary/open_vocab_smoke_objects.png" in manifest["smoke_test_command"]


def test_prepare_layout_no_create_dirs_returns_manifest_only(tmp_path: Path) -> None:
    root = tmp_path / "OpenVocabulary"

    manifest = prepare_layout(root, create_dirs=False, write_script=False)

    assert not root.exists()
    assert manifest["sources"]["sam3_repo_url"] == SAM3_REPO_URL
    assert manifest["preflight_command"][0:3] == ["python3", "run.py", "check-open-vocab-integration"]
