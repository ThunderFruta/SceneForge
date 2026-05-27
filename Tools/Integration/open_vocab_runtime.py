from __future__ import annotations

from pathlib import Path
from typing import Any

from Tools.Integration.open_vocab_setup import OpenVocabLayout

SCENE_PRIMITIVES_V1_PROMPT = "object . foreground object . plane . floor . wall . box . sphere . cylinder . cone . chair . table ."
PROMPT_PRESETS = {
    "scene-primitives-v1": SCENE_PRIMITIVES_V1_PROMPT,
}
OPEN_VOCAB_BACKENDS = {"sam3", "groundingdino-sam3"}


def prompt_preset_names() -> tuple[str, ...]:
    return tuple(sorted(PROMPT_PRESETS))


def resolve_text_prompt(text_prompt: str | None, text_prompt_preset: str | None) -> tuple[str, str, str | None]:
    preset_name = text_prompt_preset or "scene-primitives-v1"
    if preset_name not in PROMPT_PRESETS:
        raise ValueError(f"Unknown text prompt preset: {preset_name}")
    if text_prompt:
        return text_prompt, "override", preset_name
    return PROMPT_PRESETS[preset_name], "preset", preset_name


def resolve_open_vocab_options(
    *,
    backend: str,
    open_vocab_root: str | Path | None,
    text_prompt: str | None,
    text_prompt_preset: str | None,
    groundingdino_repo_dir: str | None,
    groundingdino_config: str | None,
    groundingdino_checkpoint: str | None,
    sam3_repo_dir: str | None,
    sam3_model_dir: str | None,
) -> dict[str, Any]:
    resolved_prompt, prompt_source, preset_name = resolve_text_prompt(text_prompt, text_prompt_preset)
    root = Path(open_vocab_root) if open_vocab_root else None
    layout = OpenVocabLayout(root) if root else None
    paths = {
        "groundingdino_repo_dir": groundingdino_repo_dir,
        "groundingdino_config": groundingdino_config,
        "groundingdino_checkpoint": groundingdino_checkpoint,
        "sam3_repo_dir": sam3_repo_dir,
        "sam3_model_dir": sam3_model_dir,
    }
    if layout is not None:
        paths = {
            "groundingdino_repo_dir": groundingdino_repo_dir or str(layout.groundingdino_repo_dir),
            "groundingdino_config": groundingdino_config or str(layout.groundingdino_config),
            "groundingdino_checkpoint": groundingdino_checkpoint or str(layout.groundingdino_checkpoint),
            "sam3_repo_dir": sam3_repo_dir or str(layout.sam3_repo_dir),
            "sam3_model_dir": sam3_model_dir or str(layout.sam3_model_dir),
        }
    enabled = backend in OPEN_VOCAB_BACKENDS
    metadata = {
        "enabled": enabled,
        "backend": backend,
        "root_dir": str(root) if root else None,
        "text_prompt_preset": preset_name,
        "text_prompt_source": prompt_source,
        "text_prompt": resolved_prompt,
        "paths": dict(paths),
        "readiness_status": "not_checked",
        "ready_for_smoke_test": None,
    }
    return {
        "enabled": enabled,
        "text_prompt": resolved_prompt,
        "text_prompt_source": prompt_source,
        "text_prompt_preset": preset_name,
        "open_vocab_root": str(root) if root else None,
        "paths": paths,
        "metadata": metadata,
    }
