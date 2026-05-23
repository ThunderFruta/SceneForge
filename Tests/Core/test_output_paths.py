from __future__ import annotations

from datetime import datetime
from pathlib import Path

from Core.Utils.output_paths import resolve_output_blend_path


def test_resolve_output_blend_path_from_directory() -> None:
    path = resolve_output_blend_path(
        "Output",
        mode="structured",
        image_path="Assets/Samples/Room/room_render.png",
        timestamp=datetime(2026, 5, 23, 14, 5, 6),
    )

    assert path == Path("Output/20260523_140506_structured_room_render/room_render.blend")


def test_resolve_output_blend_path_from_file_name() -> None:
    path = resolve_output_blend_path(
        "Output/my scene.obj",
        mode="relief",
        image_path="input.png",
        timestamp=datetime(2026, 5, 23, 14, 5, 6),
    )

    assert path == Path("Output/20260523_140506_relief_my_scene/my scene.blend")
