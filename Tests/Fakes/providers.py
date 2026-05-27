from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from MeshReconstruction.types import MeshProvider, MeshResult
from Segmentation.types import SegmentDetection
from ShapeDetection.types import PrimitivePrediction
from WireframeDetection.render import write_wireframe_json, write_wireframe_overlay
from WireframeDetection.types import WireframeLine, WireframeProvider, WireframeResult


class FakeSegmenter:
    def __init__(self, mode: str = "sample") -> None:
        if mode not in {"sample", "none"}:
            raise ValueError(f"Unsupported fake segmenter mode: {mode}")
        self.mode = mode

    def detect(self, image: Image.Image) -> list[SegmentDetection]:
        if self.mode == "none":
            return []

        width, height = image.size
        left = round(width * 0.2, 2)
        top = round(height * 0.2, 2)
        right = round(width * 0.8, 2)
        bottom = round(height * 0.8, 2)
        polygon = [(left, top), (right, top), (right, bottom), (left, bottom)]
        return [
            SegmentDetection(
                bbox_xyxy=(left, top, right, bottom),
                mask_polygon=polygon,
                detector_label="object",
                detector_confidence=0.9,
            )
        ]


class FakePrimitiveClassifier:
    def classify(self, image: Image.Image, detection: SegmentDetection) -> PrimitivePrediction:
        del image
        left, top, right, bottom = detection.bbox_xyxy
        width = max(1.0, right - left)
        height = max(1.0, bottom - top)
        ratio = width / height
        label = "box" if 0.75 <= ratio <= 1.35 else "cylinder"
        return PrimitivePrediction(label=label, confidence=0.75)


class FakeMeshProvider(MeshProvider):
    backend = "test_fake"
    model_dir = None

    def reconstruct(self, rgb_crop_path: Path, mask_path: Path, output_path: Path) -> MeshResult:
        del rgb_crop_path, mask_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            "\n".join(
                [
                    "# test fake mesh candidate",
                    "v -0.5 -0.5 -0.5",
                    "v 0.5 -0.5 -0.5",
                    "v 0.5 0.5 -0.5",
                    "v -0.5 0.5 -0.5",
                    "v -0.5 -0.5 0.5",
                    "v 0.5 -0.5 0.5",
                    "v 0.5 0.5 0.5",
                    "v -0.5 0.5 0.5",
                    "f 1 2 3 4",
                    "f 5 8 7 6",
                    "f 1 5 6 2",
                    "f 2 6 7 3",
                    "f 3 7 8 4",
                    "f 4 8 5 1",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return MeshResult(status="ok", path=output_path)


class FakeWireframeProvider(WireframeProvider):
    backend = "test_fake"
    model_dir = None

    def detect_wireframe(
        self,
        rgb_crop_path: Path,
        mask_path: Path,
        output_json_path: Path,
        output_overlay_path: Path,
    ) -> WireframeResult:
        rgb = Image.open(rgb_crop_path).convert("RGB")
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 127
        ys, xs = np.nonzero(mask)
        if xs.size == 0 or ys.size == 0:
            lines: list[WireframeLine] = []
            junction_count = 0
        else:
            x0 = float(xs.min())
            y0 = float(ys.min())
            x1 = float(xs.max())
            y1 = float(ys.max())
            lines = [
                WireframeLine(x0, y0, x1, y0, 1.0),
                WireframeLine(x1, y0, x1, y1, 1.0),
                WireframeLine(x1, y1, x0, y1, 1.0),
                WireframeLine(x0, y1, x0, y0, 1.0),
            ]
            junction_count = 4

        write_wireframe_json(
            output_json_path,
            width=rgb.width,
            height=rgb.height,
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

