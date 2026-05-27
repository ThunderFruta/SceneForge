from __future__ import annotations

import numpy as np
from PIL import Image

from EdgeDetection.types import EdgeProvider, EdgeResult


class NoEdgeProvider(EdgeProvider):
    backend = "none"
    model_dir = None

    def detect_edges(self, image: Image.Image) -> EdgeResult:
        empty = np.zeros((image.height, image.width), dtype=np.uint8)
        return EdgeResult(image=Image.fromarray(empty, mode="L"), backend=self.backend, model_dir=None)

