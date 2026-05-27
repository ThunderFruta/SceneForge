from __future__ import annotations

from PIL import Image, ImageFilter

from EdgeDetection.types import EdgeProvider, EdgeResult


class SimpleEdgeProvider(EdgeProvider):
    backend = "simple"
    model_dir = None

    def detect_edges(self, image: Image.Image) -> EdgeResult:
        edge_image = image.convert("L").filter(ImageFilter.FIND_EDGES)
        return EdgeResult(image=edge_image, backend=self.backend, model_dir=None)

