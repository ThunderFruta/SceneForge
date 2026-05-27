from __future__ import annotations

import numpy as np
from PIL import Image

from Segmentation.primitive_3d import (
    Primitive3DSegNet,
    build_point_cloud_arrays,
    segment_detections_from_arrays,
)


def test_primitive_3d_model_forward_shapes() -> None:
    import torch

    model = Primitive3DSegNet(input_dim=10, hidden_dim=12, embedding_dim=5)
    embeddings, objectness = model(torch.zeros((2, 7, 10), dtype=torch.float32))

    assert tuple(embeddings.shape) == (2, 7, 5)
    assert tuple(objectness.shape) == (2, 7)


def test_point_cloud_builder_is_deterministic() -> None:
    image = Image.new("RGB", (4, 4), (128, 64, 32))
    depth = np.full((4, 4), 0.5, dtype=np.float32)
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True

    first = build_point_cloud_arrays(image=image, depth=depth, masks=[mask], max_points=8)
    second = build_point_cloud_arrays(image=image, depth=depth, masks=[mask], max_points=8)

    assert first["features"].shape == (8, 10)
    assert np.array_equal(first["features"], second["features"])
    assert np.array_equal(first["labels"], second["labels"])
    assert set(first["labels"].tolist()) <= {0, 1}


def test_cluster_adapter_returns_segment_detection() -> None:
    embeddings = np.zeros((9, 2), dtype=np.float32)
    objectness = np.ones(9, dtype=np.float32)
    rows = np.asarray([1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=np.int32)
    cols = np.asarray([1, 2, 3, 1, 2, 3, 1, 2, 3], dtype=np.int32)

    detections = segment_detections_from_arrays(
        embeddings=embeddings,
        objectness=objectness,
        rows=rows,
        cols=cols,
        image_size=(4, 4),
        objectness_threshold=0.5,
        embedding_distance=0.1,
        min_cluster_points=9,
    )

    assert len(detections) == 1
    assert detections[0].detector_label == "unknown"
    assert detections[0].detector_confidence == 1.0
