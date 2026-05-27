from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from WireframeDetection.types import WireframeLine


def write_wireframe_json(
    path: Path,
    *,
    width: int,
    height: int,
    lines: list[WireframeLine],
    junction_count: int,
    backend: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend": backend,
                "width": int(width),
                "height": int(height),
                "line_count": len(lines),
                "junction_count": int(junction_count),
                "lines": [line.to_list() for line in lines],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_wireframe_overlay(image_path: Path, output_path: Path, lines: list[WireframeLine]) -> None:
    image = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    for line in lines:
        alpha = int(max(80, min(255, 80 + 175 * line.score)))
        draw.line((line.x1, line.y1, line.x2, line.y2), fill=(0, 255, 255, alpha), width=2)
        r = 2
        draw.ellipse((line.x1 - r, line.y1 - r, line.x1 + r, line.y1 + r), fill=(255, 255, 0, alpha))
        draw.ellipse((line.x2 - r, line.y2 - r, line.x2 + r, line.y2 + r), fill=(255, 255, 0, alpha))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path)
