from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from Tools.Integration.open_vocab_setup import OpenVocabLayout

SCENE_PRIMITIVES_V1_PROMPT = "object . foreground object . chair . table . sofa . bed . lamp . couch . cabinet . plant . flower . flowers . vase . flower pot . person . box . sphere . cylinder . cone ."
PROMPT_PRESETS = {
    "scene-primitives-v1": SCENE_PRIMITIVES_V1_PROMPT,
}
OPEN_VOCAB_BACKENDS = {"sam3", "groundingdino-sam3", "ram-groundingdino-sam3"}
DEFAULT_QWEN_VOCAB_PATH = Path("Output/Latest/qwen_object_vocab.json")
DEFAULT_QWEN_MAX_OBJECTS = 24
QWEN_MIN_VALID_TERMS = 4
QWEN_DEFAULT_PROMPT = (
    "List the 24 most likely object classes for an indoor 3D scene. "
    "Return ONLY a JSON array of short object names."
)


def prompt_preset_names() -> tuple[str, ...]:
    return tuple(sorted(PROMPT_PRESETS))


def resolve_text_prompt(
    text_prompt: str | None,
    text_prompt_preset: str | None,
    *,
    refresh_text_prompt: bool = False,
    text_prompt_refresh_path: str | Path | None = None,
) -> tuple[str, str, str | None]:
    preset_name = text_prompt_preset or "scene-primitives-v1"
    if preset_name not in PROMPT_PRESETS:
        raise ValueError(f"Unknown text prompt preset: {preset_name}")
    if text_prompt:
        return text_prompt, "override", preset_name

    if refresh_text_prompt:
        refresh_path = Path(text_prompt_refresh_path) if text_prompt_refresh_path else DEFAULT_QWEN_VOCAB_PATH
        refreshed = _resolve_or_generate_prompt(
            path=refresh_path,
            refresh_text_prompt=refresh_text_prompt,
        )
        if refreshed:
            return refreshed, "qwen_refresh", preset_name

    return PROMPT_PRESETS[preset_name], "preset", preset_name


def _resolve_or_generate_prompt(path: Path, refresh_text_prompt: bool) -> str | None:
    if refresh_text_prompt:
        generated_terms = _generate_terms_with_qwen()
        if generated_terms:
            _write_terms_cache(generated_terms, path)
            return _format_prompt(generated_terms)

    terms = _load_terms_from_cache(path)
    if terms:
        _write_terms_cache(terms, path)
        return _format_prompt(terms)

    generated_terms = _generate_terms_with_qwen()
    if not generated_terms:
        return None
    _write_terms_cache(generated_terms, path)
    return _format_prompt(generated_terms)


def _load_terms_from_cache(path: Path) -> list[str]:
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return []

    if isinstance(data, dict):
        values = data.get("objects")
        if isinstance(values, list):
            data = values
        elif isinstance(values, str):
            return [values]
        elif "labels" in data and isinstance(data["labels"], list):
            data = data["labels"]
        else:
            return []

    if not isinstance(data, list):
        return []

    normalized = _normalize_terms(
        term
        for term in data
        if isinstance(term, str) or isinstance(term, (int, float))
    )
    return normalized


def _normalize_terms(terms: Iterable[str | int | float]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for term in terms:
        text = str(term).strip()
        if not _is_valid_term(text):
            continue
        text = re.sub(r"\s+", " ", text.strip().lower())
        text = text.strip(" .,:;!")
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
        if len(normalized) >= DEFAULT_QWEN_MAX_OBJECTS:
            break
    return normalized


def _is_valid_term(term: str) -> bool:
    text = term.strip().lower()
    if not text:
        return False
    if any(ch in text for ch in "[]{}\"'"):
        return False
    if any(ch in text for ch in ("note:", "for example", "shortname", "name", "object", "objects", "foreground")):
        return False
    if re.search(r"\s+", text):
        return len(text.split()) <= 3
    if len(text) < 3 or len(text) > 28:
        return False
    if text.count(".") > 0:
        return False
    return bool(re.fullmatch(r"[a-z0-9][a-z0-9 _-]*", text))


def _format_prompt(terms: list[str]) -> str:
    return " . ".join(terms) + " ."


def _write_terms_cache(terms: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(terms, indent=2) + "\n", encoding="utf-8")


def _generate_terms_with_qwen() -> list[str]:
    prompt = os.getenv("SCENEFORGE_QWEN_VOCAB_PROMPT", QWEN_DEFAULT_PROMPT)
    max_items = int(os.getenv("SCENEFORGE_QWEN_MAX_OBJECTS", str(DEFAULT_QWEN_MAX_OBJECTS)))
    model_id = os.getenv("SCENEFORGE_QWEN_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    try:
        from transformers import pipeline
    except Exception:
        return []

    try:
        generator = pipeline("text-generation", model=model_id, tokenizer=model_id)
        output = generator(prompt, max_new_tokens=180, do_sample=False)
    except Exception:
        return []

    if not output:
        return []

    text = output[0]["generated_text"] if isinstance(output[0], dict) else str(output[0])
    text = text[len(prompt):] if text.startswith(prompt) else text
    return _extract_terms(text, max_items=max_items)


def _extract_terms(text: str, max_items: int) -> list[str]:
    candidates = _extract_json_list(text)
    if not candidates:
        candidates = _extract_csv_terms(text)
    if not candidates:
        candidates = _extract_terms_from_text_blobs(text)

    terms = _normalize_terms(candidates[:max_items])
    if len(terms) >= QWEN_MIN_VALID_TERMS:
        return terms

    return []


def _extract_json_list(text: str) -> list[str]:
    match = re.search(r"\[[^\]]*\]", text, re.S)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []

    terms: list[str] = []
    for item in parsed:
        if isinstance(item, dict):
            name = item.get("name") or item.get("label")
            if isinstance(name, str):
                terms.append(name)
            continue
        if isinstance(item, str):
            terms.append(item)
        elif isinstance(item, (int, float)):
            terms.append(str(item))
    return terms


def _extract_csv_terms(text: str) -> list[str]:
    if "," not in text and "\n" not in text:
        return []
    return [item.strip() for item in re.split(r"[,\n]", text)]


def _extract_terms_from_text_blobs(text: str) -> list[str]:
    cleaned = re.sub(r"[-*\u2022]", "", text)
    chunks = re.split(r"[\n;]", cleaned)
    terms: list[str] = []
    for chunk in chunks:
        pieces = re.split(r"[.,]", chunk)
        terms.extend(piece.strip() for piece in pieces if piece.strip())
    return terms


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
    ram_repo_dir: str | None = None,
    ram_checkpoint: str | None = None,
    refresh_text_prompt: bool = False,
    text_prompt_refresh_path: str | Path | None = None,
) -> dict[str, Any]:
    if backend == "ram-groundingdino-sam3":
        resolved_prompt = ""
        prompt_source = "ram_tags"
        preset_name = None
        refresh_text_prompt = False
    else:
        resolved_prompt, prompt_source, preset_name = resolve_text_prompt(
            text_prompt,
            text_prompt_preset,
            refresh_text_prompt=refresh_text_prompt,
            text_prompt_refresh_path=text_prompt_refresh_path,
        )
    root = Path(open_vocab_root) if open_vocab_root else None
    layout = OpenVocabLayout(root) if root else None
    paths = {
        "groundingdino_repo_dir": groundingdino_repo_dir,
        "groundingdino_config": groundingdino_config,
        "groundingdino_checkpoint": groundingdino_checkpoint,
        "sam3_repo_dir": sam3_repo_dir,
        "sam3_model_dir": sam3_model_dir,
        "ram_repo_dir": ram_repo_dir,
        "ram_checkpoint": ram_checkpoint,
    }
    if layout is not None:
        paths = {
            "groundingdino_repo_dir": groundingdino_repo_dir or str(layout.groundingdino_repo_dir),
            "groundingdino_config": groundingdino_config or str(layout.groundingdino_config),
            "groundingdino_checkpoint": groundingdino_checkpoint or str(layout.groundingdino_checkpoint),
            "sam3_repo_dir": sam3_repo_dir or str(layout.sam3_repo_dir),
            "sam3_model_dir": sam3_model_dir or str(layout.sam3_model_dir),
            "ram_repo_dir": ram_repo_dir,
            "ram_checkpoint": ram_checkpoint,
        }
    enabled = backend in OPEN_VOCAB_BACKENDS
    metadata = {
        "enabled": enabled,
        "backend": backend,
        "root_dir": str(root) if root else None,
        "text_prompt_preset": preset_name,
        "text_prompt_source": prompt_source,
        "text_prompt": resolved_prompt,
        "text_prompt_refresh_enabled": refresh_text_prompt,
        "text_prompt_refresh_path": str(Path(text_prompt_refresh_path)) if text_prompt_refresh_path else str(DEFAULT_QWEN_VOCAB_PATH),
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
