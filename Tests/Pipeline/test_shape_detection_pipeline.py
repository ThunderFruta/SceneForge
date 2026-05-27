from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from Tests.Fakes.providers import FakePrimitiveClassifier, FakeSegmenter
from ShapeDetection.pipeline import run_shape_detection


def test_fake_pipeline_writes_json_and_overlay(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    Image.new("RGB", (100, 80), (40, 50, 60)).save(image_path)
    output_dir = tmp_path / "out"

    run_shape_detection(
        image_path=image_path,
        output_dir=output_dir,
        segmenter=FakeSegmenter(mode="sample"),
        classifier=FakePrimitiveClassifier(),
        model_info={"detector_backend": "fake", "classifier_backend": "fake"},
    )

    data = json.loads((output_dir / "detections.json").read_text(encoding="utf-8"))
    assert data["image_width"] == 100
    assert data["image_height"] == 80
    assert data["objects"][0]["id"] == 1
    assert data["objects"][0]["primitive_label"] == "box"
    assert (output_dir / "overlay.png").is_file()


def test_no_detections_still_writes_valid_outputs(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    Image.new("RGB", (16, 12), "white").save(image_path)
    output_dir = tmp_path / "out"

    run_shape_detection(
        image_path=image_path,
        output_dir=output_dir,
        segmenter=FakeSegmenter(mode="none"),
        classifier=FakePrimitiveClassifier(),
        model_info={"detector_backend": "fake", "classifier_backend": "fake"},
    )

    data = json.loads((output_dir / "detections.json").read_text(encoding="utf-8"))
    overlay = Image.open(output_dir / "overlay.png")
    assert data["objects"] == []
    assert overlay.size == (16, 12)
