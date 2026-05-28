from __future__ import annotations

import re
import sys
import time
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
    latest_progress: float = 0.0
    last_percent_bucket: int | None = None
    last_report: str | None = None
    phase_updates: int = 0
    last_line_length: int = 0

    def write(self, progress: float | None, message: str = "", tty: bool = True) -> None:
        if progress is None:
            if self.total:
                progress = 0.0
            else:
                return
        if progress < self.latest_progress:
            progress = self.latest_progress
        self.updated = True
        self.latest_progress = progress
        bar = _format_bar(progress, self.width)
        sanitized_message = ""
        if message:
            sanitized_message = str(message).replace("\r", " ").replace("\n", " ").strip()
            if len(sanitized_message) > 56:
                sanitized_message = sanitized_message[:53].rstrip() + "..."
        suffix = f" ({sanitized_message})" if sanitized_message else ""
        percent = int(progress * 100.0 + 1e-9)
        line = f"{self.label} {bar} {progress * 100.0:5.1f}%{suffix}"
        if not tty:
            if self.last_percent_bucket is not None and percent == self.last_percent_bucket:
                return
            self.last_percent_bucket = percent
            print(line, flush=True)
            self.last_line_length = len(line)
            return
        clear_width = ""
        if self.last_line_length > len(line):
            clear_width = " " * (self.last_line_length - len(line))
        print(f"\r{line}{clear_width}", end="", flush=True)
        self.last_line_length = len(line)

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
        progress = min(0.20, 0.02 + (self.phase_updates * 0.01))
        self.write(progress, message=prefix, tty=tty)

    def heartbeat(self, ttl_progress: float, message: str, tty: bool = True) -> None:
        if self.done:
            return
        target = min(ttl_progress, 0.98)
        if target <= self.latest_progress:
            return
        self.write(target, message=message, tty=tty)


class BlenderRenderProgressBar:
    """Context manager that renders a terminal progress bar from Blender render callbacks."""

    def __init__(self, label: str, total_samples: int | None = None, width: int = 28) -> None:
        self.state = _ProgressState(label=label, total=float(total_samples) if total_samples and total_samples > 0 else None, width=width)
        self._active = True
        self._heartbeat_running = False
        self._last_report_time: float = 0.0
        self._heartbeat_interval_seconds: float = 0.25
        self._heartbeat_step: float = 0.002
        self._heartbeat_next_target: float = 0.0
        self._heartbeat_message = "rendering"
        self._handlers: list[tuple[list[Callable[..., Any]], Callable[..., Any]]] = []

    def __enter__(self) -> "BlenderRenderProgressBar":
        self._stdout_is_tty = sys.stdout.isatty()
        self._last_report_time = time.monotonic()
        import bpy

        self._handlers = []
        self._register(bpy.app.handlers.render_init, self._on_render_init)
        self._register(bpy.app.handlers.render_stats, self._on_render_stats)
        self._register(bpy.app.handlers.render_complete, self._on_render_end)
        self._register(bpy.app.handlers.render_cancel, self._on_render_end)
        self._register(bpy.app.handlers.render_write, self._on_render_end)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._active = False
        self._stop_heartbeat()
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

    def _start_heartbeat(self) -> None:
        if self._heartbeat_running:
            return
        self._heartbeat_running = True

        import bpy

        self._heartbeat_next_target = min(0.90, self.state.latest_progress + self._heartbeat_step)

        def _tick() -> float | None:
            if not self._heartbeat_running or not self._active:
                return None
            now = time.monotonic()
            if now - self._last_report_time >= self._heartbeat_interval_seconds:
                self._heartbeat_next_target = min(
                    0.98,
                    max(self.state.latest_progress + self._heartbeat_step, self._heartbeat_next_target),
                )
                self.state.heartbeat(self._heartbeat_next_target, self._heartbeat_message, tty=self._stdout_is_tty)
                self._last_report_time = now
            return self._heartbeat_interval_seconds

        bpy.app.timers.register(_tick)

    def _stop_heartbeat(self) -> None:
        if not self._heartbeat_running:
            return
        self._heartbeat_running = False

    def _unregister(self) -> None:
        for collection, callback in self._handlers:
            while callback in collection:
                collection.remove(callback)
        self._handlers = []

    def _on_render_init(self, *_args: Any) -> None:
        self._last_report_time = time.monotonic()
        self.state.last_report = None
        self.state.phase_updates = 0
        self._heartbeat_next_target = 0.01
        self._heartbeat_message = "Starting render"
        self._start_heartbeat()

    def _on_render_stats(self, *_args: Any) -> None:
        report = str(_args[0]) if _args else ""
        if _match := _SAMPLE_PATTERN.search(report):
            current = int(_match.group(1))
            total = max(1, int(_match.group(2)))
            if not self.state.total:
                self.state.total = float(total)
            self._last_report_time = time.monotonic()
            progress = current / total
            message = f"{current} / {total} samples"
            self.state.write(progress, message=message, tty=self._stdout_is_tty)
            return
        if _match := _SAMPLE_LINE_PATTERN.search(report):
            current = int(_match.group(1))
            total = max(1, int(_match.group(2)))
            if not self.state.total:
                self.state.total = float(total)
            self._last_report_time = time.monotonic()
            progress = current / total
            message = f"{current} / {total} samples"
            self.state.write(progress, message=message, tty=self._stdout_is_tty)
            return
        if _match := _PERCENT_PATTERN.search(report):
            self._last_report_time = time.monotonic()
            percent = min(100, max(0, int(_match.group(1))))
            self.state.write(percent / 100.0, message=report[:70], tty=self._stdout_is_tty)
            return
        self._heartbeat_message = report[:70] or "rendering"
        self.state.bump_phase(report, tty=self._stdout_is_tty)
        if report.startswith("Saved:"):
            self.state.complete(tty=self._stdout_is_tty)

    def _on_render_end(self, *_args: Any) -> None:
        self._stop_heartbeat()
        self.state.complete(tty=self._stdout_is_tty)
