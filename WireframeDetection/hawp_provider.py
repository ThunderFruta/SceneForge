from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from PIL import Image

from WireframeDetection.render import write_wireframe_json, write_wireframe_overlay
from WireframeDetection.types import WireframeLine, WireframeProvider, WireframeResult


class HawpWireframeProvider(WireframeProvider):
    backend = "hawp"

    def __init__(
        self,
        model_dir: str | Path,
        device: str | None = None,
        threshold: float = 0.05,
        timeout_seconds: int = 120,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = device
        self.threshold = threshold
        self.timeout_seconds = timeout_seconds
        if not self.model_dir.is_dir():
            raise ValueError(f"--wireframe-model-dir does not exist or is not a directory: {self.model_dir}")
        self.repo_dir = self._find_repo_dir()
        self.checkpoint_path = self._find_checkpoint_path()

    def _find_repo_dir(self) -> Path:
        candidates = [self.model_dir / "repo", self.model_dir]
        for candidate in candidates:
            if (candidate / "hawp" / "ssl" / "predict.py").is_file():
                return candidate
        raise ValueError(
            "HAWP model directory is missing the source repo. "
            "Expected --wireframe-model-dir/repo/hawp/ssl/predict.py."
        )

    def _find_checkpoint_path(self) -> Path:
        preferred = [
            self.model_dir / "checkpoints" / "hawpv3-imagenet-03a84.pth",
            self.model_dir / "checkpoints" / "hawpv3-fdc5487a.pth",
            self.model_dir / "checkpoints" / "hawpv2-edb9b23f.pth",
        ]
        for candidate in preferred:
            if candidate.is_file():
                return candidate
        matches = sorted((self.model_dir / "checkpoints").glob("hawpv*.pth"))
        if matches:
            return matches[0]
        matches = sorted(self.model_dir.rglob("hawpv*.pth"))
        if matches:
            return matches[0]
        raise ValueError(
            "HAWP model directory is missing a hawpv*.pth checkpoint. "
            "Expected one under --wireframe-model-dir/checkpoints/."
        )

    def detect_wireframe(
        self,
        rgb_crop_path: Path,
        mask_path: Path,
        output_json_path: Path,
        output_overlay_path: Path,
    ) -> WireframeResult:
        del mask_path
        try:
            with tempfile.TemporaryDirectory(prefix="sceneforge_hawp_") as temp_name:
                temp_dir = Path(temp_name)
                command = [
                    sys.executable,
                    "-m",
                    "hawp.ssl.predict",
                    "--ckpt",
                    str(self.checkpoint_path.resolve()),
                    "--threshold",
                    str(self.threshold),
                    "--img",
                    str(rgb_crop_path.resolve()),
                    "--saveto",
                    str(temp_dir.resolve()),
                    "--ext",
                    "json",
                    "--device",
                    self._hawp_device(),
                    "--disable-show",
                ]
                env = os.environ.copy()
                existing_pythonpath = env.get("PYTHONPATH")
                env["PYTHONPATH"] = (
                    str(self.repo_dir)
                    if not existing_pythonpath
                    else f"{self.repo_dir}{os.pathsep}{existing_pythonpath}"
                )
                result = subprocess.run(
                    command,
                    cwd=self.repo_dir,
                    env=env,
                    text=True,
                    capture_output=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
                if result.returncode != 0:
                    return WireframeResult(
                        status="failed",
                        lines=[],
                        junction_count=0,
                        reason=self._format_subprocess_error(result),
                    )
                raw_json_path = temp_dir / rgb_crop_path.with_suffix(".json").name
                if not raw_json_path.is_file():
                    return WireframeResult(
                        status="failed",
                        lines=[],
                        junction_count=0,
                        reason=f"HAWP did not write expected JSON output: {raw_json_path}",
                    )
                raw = json.loads(raw_json_path.read_text(encoding="utf-8"))
        except subprocess.TimeoutExpired:
            return WireframeResult(
                status="failed",
                lines=[],
                junction_count=0,
                reason=f"HAWP timed out after {self.timeout_seconds}s",
            )
        except Exception as exc:
            return WireframeResult(status="failed", lines=[], junction_count=0, reason=str(exc))

        lines = self._lines_from_hawp_json(raw)
        image = Image.open(rgb_crop_path).convert("RGB")
        junction_count = len(raw.get("vertices", []))
        write_wireframe_json(
            output_json_path,
            width=image.width,
            height=image.height,
            lines=lines,
            junction_count=junction_count,
            backend=self.backend,
        )
        write_wireframe_overlay(rgb_crop_path, output_overlay_path, lines)
        return WireframeResult(
            status="ok",
            lines=lines,
            junction_count=junction_count,
            json_path=output_json_path,
            overlay_path=output_overlay_path,
        )

    def _hawp_device(self) -> str:
        if self.device in {None, "", "cuda", "cpu", "mps"}:
            return self.device or "cuda"
        if str(self.device).isdigit() or str(self.device).startswith("cuda"):
            return "cuda"
        return "cpu"

    def _lines_from_hawp_json(self, data: dict) -> list[WireframeLine]:
        vertices = data.get("vertices", [])
        edges = data.get("edges", [])
        weights = data.get("edges-weights", [])
        lines: list[WireframeLine] = []
        for edge, score in zip(edges, weights):
            score_value = float(score)
            if score_value < self.threshold:
                continue
            if len(edge) != 2:
                continue
            start_index, end_index = int(edge[0]), int(edge[1])
            if start_index >= len(vertices) or end_index >= len(vertices):
                continue
            start = vertices[start_index]
            end = vertices[end_index]
            if len(start) < 2 or len(end) < 2:
                continue
            lines.append(
                WireframeLine(
                    float(start[0]),
                    float(start[1]),
                    float(end[0]),
                    float(end[1]),
                    score_value,
                )
            )
        return lines

    @staticmethod
    def _format_subprocess_error(result: subprocess.CompletedProcess[str]) -> str:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        detail = stderr or stdout or "no output"
        if len(detail) > 500:
            detail = detail[-500:]
        return f"HAWP exited with code {result.returncode}: {detail}"
