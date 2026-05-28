from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from typing import Any, Callable

_SAMPLE_PATTERN = re.compile(r"Rendering\s+(\d+)\s*/\s*(\d+)\s+samples", re.IGNORECASE)
_SAMPLE_LINE_PATTERN = re.compile(r"Sample\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)
_PERCENT_PATTERN = re.compile(r"(\d{1,3})%\s*$")


def _format_bar(progress: float, width: int = 28) -> str:
    clamped = max(0.0, min(1.0, progress))
    filled = int(round(clamped * width))
    filled = max(0, min(width, filled))
    empty = width - filled
    return f"[{'#' * filled}{'.' * empty}]"


@dataclass
class _ProgressState:
    label: str
    total: float | None
    width: int = 28
    done: bool = False
    updated: bool = False
    last_percent_bucket: int | None = None
    last_report: str | None = None
    phase_updates: int = 0

    def write(self, progress: float | None, message: str = "", tty: bool = True) -> None:
        if progress is None:
            if self.total:
                progress = 0.0
            else:
                return
        self.updated = True
        bar = _format_bar(progress, self.width)
        suffix = f" ({message})" if message else ""
        percent = int(progress * 100.0 + 1e-9)
        if not tty:
            if self.last_percent_bucket is not None and percent == self.last_percent_bucket:
                return
            self.last_percent_bucket = percent
            print(f"{self.label} {bar} {progress * 100.0:5.1f}%{suffix}", flush=True)
            return
        print(f"\r{self.label} {bar} {progress * 100.0:5.1f}%{suffix}", end="", flush=True)

    def complete(self, tty: bool = True) -> None:
        if self.done:
            return
        self.done = True
        self.write(1.0, message="done", tty=tty)
        if self.updated:
            if tty:
                print("", flush=True)

    def bump_phase(self, report: str, tty: bool = True) -> None:
        if "Mem:" not in report:
            return
        prefix = report[:70]
        if self.last_report == prefix:
            return
        self.last_report = prefix
        self.phase_updates += 1
        progress = min(0.90, 0.02 + (self.phase_updates * 0.02))
        self.write(progress, message=prefix, tty=tty)


class BlenderRenderProgressBar:
    """Context manager that renders a terminal progress bar from Blender render callbacks."""

    def __init__(self, label: str, total_samples: int | None = None, width: int = 28) -> None:
        self.state = _ProgressState(label=label, total=float(total_samples) if total_samples and total_samples > 0 else None, width=width)
        self._handlers: list[tuple[list[Callable[..., Any]], Callable[..., Any]]] = []

    def __enter__(self) -> "BlenderRenderProgressBar":
        self._stdout_is_tty = sys.stdout.isatty()
        import bpy

        self._handlers = []
        self._register(bpy.app.handlers.render_stats, self._on_render_stats)
        self._register(bpy.app.handlers.render_complete, self._on_render_end)
        self._register(bpy.app.handlers.render_cancel, self._on_render_end)
        self._register(bpy.app.handlers.render_write, self._on_render_end)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.state.complete(tty=self._stdout_is_tty)
        else:
            if self.state.updated:
                print()
        self._unregister()

    def _register(self, collection: list[Callable[..., Any]], callback: Callable[..., Any]) -> None:
        if callback in collection:
            return
        collection.append(callback)
        self._handlers.append((collection, callback))

    def _unregister(self) -> None:
        for collection, callback in self._handlers:
            while callback in collection:
                collection.remove(callback)
        self._handlers = []

    def _on_render_stats(self, *_args: Any) -> None:
        report = str(_args[0]) if _args else ""
        if _match := _SAMPLE_PATTERN.search(report):
            current = int(_match.group(1))
            total = max(1, int(_match.group(2)))
            if not self.state.total:
                self.state.total = float(total)
            progress = current / total
            message = f"{current} / {total} samples"
            self.state.write(progress, message=message, tty=self._stdout_is_tty)
            return
        if _match := _SAMPLE_LINE_PATTERN.search(report):
            current = int(_match.group(1))
            total = max(1, int(_match.group(2)))
            if not self.state.total:
                self.state.total = float(total)
            progress = current / total
            message = f"{current} / {total} samples"
            self.state.write(progress, message=message, tty=self._stdout_is_tty)
            return
        if _match := _PERCENT_PATTERN.search(report):
            percent = min(100, max(0, int(_match.group(1))))
            self.state.write(percent / 100.0, message=report[:70], tty=self._stdout_is_tty)
            return
        self.state.bump_phase(report, tty=self._stdout_is_tty)
        if report.startswith("Saved:"):
            self.state.complete(tty=self._stdout_is_tty)

    def _on_render_end(self, *_args: Any) -> None:
        self.state.complete(tty=self._stdout_is_tty)
