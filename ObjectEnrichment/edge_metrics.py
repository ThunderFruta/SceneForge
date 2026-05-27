from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from ObjectEnrichment.types import EdgeEvidence


def measure_edge_evidence(mask_path: str, edge_path: str) -> EdgeEvidence:
    mask = Image.open(mask_path).convert("L")
    edge = Image.open(edge_path).convert("L")
    if mask.size != edge.size:
        edge = edge.resize(mask.size)

    mask_array = np.asarray(mask, dtype=np.uint8) > 127
    edge_array = np.asarray(edge, dtype=np.uint8) > 32
    if mask_array.sum() == 0:
        return EdgeEvidence(status="not_available", boundary_agreement=0.0, edge_density=0.0)

    boundary = np.asarray(mask.filter(ImageFilter.FIND_EDGES), dtype=np.uint8) > 0
    boundary_count = int(boundary.sum())
    if boundary_count == 0:
        agreement = 0.0
    else:
        agreement = float((boundary & edge_array).sum() / boundary_count)
    density = float((edge_array & mask_array).sum() / max(1, int(mask_array.sum())))
    return EdgeEvidence(status="ok", boundary_agreement=agreement, edge_density=density)
