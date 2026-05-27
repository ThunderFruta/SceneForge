from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from EdgeDetection.types import EdgeProvider, EdgeResult


class DexiNedEdgeProvider(EdgeProvider):
    backend = "dexined"

    def __init__(self, model_dir: str | Path, device: str | None = None) -> None:
        self.model_dir = Path(model_dir)
        self.device = device
        if not self.model_dir.is_dir():
            raise ValueError(f"--edge-model-dir does not exist or is not a directory: {self.model_dir}")
        self.model_path = self._find_model_path()
        self._net = None

    def _find_model_path(self) -> Path:
        candidates = [
            self.model_dir / "edge_detection_dexined_2024sep.onnx",
            self.model_dir / "opencv" / "edge_detection_dexined_2024sep.onnx",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        matches = sorted(self.model_dir.rglob("edge_detection_dexined*.onnx"))
        if matches:
            return matches[0]
        raise ValueError(
            "DexiNed model directory is missing edge_detection_dexined_2024sep.onnx. "
            "Expected it directly under --edge-model-dir or under opencv/."
        )

    def detect_edges(self, image: Image.Image) -> EdgeResult:
        import cv2 as cv

        if self._net is None:
            self._net = cv.dnn.readNetFromONNX(str(self.model_path))
            self._net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
            self._net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        bgr = rgb[..., ::-1]
        blob = cv.dnn.blobFromImage(
            bgr,
            scalefactor=1.0,
            size=(512, 512),
            mean=(103.5, 116.2, 123.6),
            swapRB=False,
            crop=False,
        )
        self._net.setInput(blob)
        outputs = self._net.forward()
        edge = self._postprocess(outputs, image.size)
        return EdgeResult(image=Image.fromarray(edge, mode="L"), backend=self.backend, model_dir=self.model_dir)

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-values))

    def _postprocess(self, outputs, size: tuple[int, int]) -> np.ndarray:
        import cv2 as cv

        width, height = size
        predictions: list[np.ndarray] = []
        for output in outputs:
            edge = self._sigmoid(np.asarray(output))
            edge = np.squeeze(edge)
            edge = cv.normalize(edge, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
            edge = cv.resize(edge, (width, height), interpolation=cv.INTER_LINEAR)
            predictions.append(edge)
        if not predictions:
            return np.zeros((height, width), dtype=np.uint8)
        return predictions[-1]
