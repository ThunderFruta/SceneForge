from __future__ import annotations

from pathlib import Path

from Tools.Integration.open_vocab_runtime import resolve_open_vocab_options, resolve_text_prompt


def test_resolve_text_prompt_uses_scene_primitives_preset() -> None:
    prompt, source, preset = resolve_text_prompt(None, "scene-primitives-v1")

    assert source == "preset"
    assert preset == "scene-primitives-v1"
    assert "foreground object" in prompt
    assert "cylinder" in prompt


def test_resolve_text_prompt_override_keeps_preset_metadata() -> None:
    prompt, source, preset = resolve_text_prompt("lamp . cabinet .", "scene-primitives-v1")

    assert prompt == "lamp . cabinet ."
    assert source == "override"
    assert preset == "scene-primitives-v1"


def test_open_vocab_root_expands_groundingdino_and_sam3_paths(tmp_path: Path) -> None:
    root = tmp_path / "OpenVocabulary"

    options = resolve_open_vocab_options(
        backend="groundingdino-sam3",
        open_vocab_root=root,
        text_prompt=None,
        text_prompt_preset="scene-primitives-v1",
        groundingdino_repo_dir=None,
        groundingdino_config=None,
        groundingdino_checkpoint=None,
        sam3_repo_dir=None,
        sam3_model_dir=None,
    )

    assert options["enabled"] is True
    assert options["paths"]["groundingdino_repo_dir"].endswith("GroundingDINO/repo")
    assert options["paths"]["groundingdino_config"].endswith("GroundingDINO/repo/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    assert options["paths"]["groundingdino_checkpoint"].endswith("GroundingDINO/weights/groundingdino_swint_ogc.pth")
    assert options["paths"]["sam3_repo_dir"].endswith("SAM3/repo")
    assert options["paths"]["sam3_model_dir"].endswith("SAM3/hf")
    assert options["metadata"]["text_prompt_preset"] == "scene-primitives-v1"
