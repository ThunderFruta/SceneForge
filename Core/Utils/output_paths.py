from __future__ import annotations

from datetime import datetime
from pathlib import Path


def resolve_output_blend_path(
    requested_output: str | Path,
    *,
    mode: str,
    image_path: str | Path,
    timestamp: datetime | None = None,
) -> Path:
    requested = Path(requested_output)
    stamp = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    image_stem = _safe_name(Path(image_path).stem)

    if requested.suffix:
        output_root = requested.parent if requested.parent != Path("") else Path("Output")
        blend_name = requested.with_suffix(".blend").name
        run_stem = _safe_name(requested.stem)
    else:
        output_root = requested
        blend_name = f"{image_stem}.blend"
        run_stem = image_stem

    run_directory = output_root / f"{stamp}_{mode}_{run_stem}"
    return run_directory / blend_name


def _safe_name(value: str) -> str:
    safe = []
    for character in value.lower():
        if character.isalnum():
            safe.append(character)
        elif character in {"-", "_"}:
            safe.append(character)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "scene"

