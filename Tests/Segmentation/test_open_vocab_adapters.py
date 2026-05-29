from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

from Segmentation.groundingdino_sam3_segmenter import GroundingDinoSam3Segmenter
from Segmentation.sam3_segmenter import Sam3Segmenter


SAM3_MODEL_BUILDER = """class DummyModel:
    pass

def build_sam3_image_model():
    return DummyModel()
"""

SAM3_PROCESSOR = """class Sam3Processor:
    def __init__(self, model, resolution=1008, device='cpu', confidence_threshold=0.5):
        self.model = model
        self.device = device
        self.prompts = []

    def set_image(self, image, state=None):
        state = state or {}
        state['original_width'], state['original_height'] = image.size
        return state

    def set_text_prompt(self, state, prompt):
        state['text_prompt'] = prompt
        return {
            'masks': [[[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]],
            'boxes': [[1, 1, 3, 3]],
            'scores': [0.91],
            'labels': [prompt],
        }

    def add_geometric_prompt(self, box, label, state):
        self.prompts.append((box, label))
        state['last_box'] = box
        return {
            'masks': [[[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]],
            'boxes': [[1, 1, 3, 3]],
            'scores': [0.95],
            'labels': ['sam3-box'],
        }

    def reset_all_prompts(self, state):
        state.pop('last_box', None)
"""

GROUNDINGDINO_INFERENCE = """class DummyModel:
    pass

def load_model(model_config_path, model_checkpoint_path, device='cpu'):
    return DummyModel()

def load_image(image_path):
    return None, 'image-tensor'

def predict(model, image, caption, box_threshold, text_threshold, device='cpu', remove_combined=False):
    return [[0.5, 0.5, 0.5, 0.5]], [0.88], ['box']
"""


def clear_modules() -> None:
    for prefix in ("sam3", "groundingdino"):
        for name in list(sys.modules):
            if name == prefix or name.startswith(f"{prefix}."):
                del sys.modules[name]


def make_sam3_repo(root: Path) -> tuple[Path, Path]:
    repo = root / "SAM3" / "repo"
    model_dir = root / "SAM3" / "hf"
    (repo / "sam3" / "model").mkdir(parents=True)
    model_dir.mkdir(parents=True)
    (repo / "sam3" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "sam3" / "model" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "sam3" / "model_builder.py").write_text(SAM3_MODEL_BUILDER, encoding="utf-8")
    (repo / "sam3" / "model" / "sam3_image_processor.py").write_text(SAM3_PROCESSOR, encoding="utf-8")
    return repo, model_dir


def make_groundingdino_repo(root: Path) -> tuple[Path, Path, Path]:
    repo = root / "GroundingDINO" / "repo"
    weights = root / "GroundingDINO" / "weights"
    (repo / "groundingdino" / "util").mkdir(parents=True)
    (repo / "groundingdino" / "config").mkdir(parents=True)
    weights.mkdir(parents=True)
    (repo / "groundingdino" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "groundingdino" / "util" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "groundingdino" / "util" / "inference.py").write_text(GROUNDINGDINO_INFERENCE, encoding="utf-8")
    config = repo / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
    checkpoint = weights / "groundingdino_swint_ogc.pth"
    config.write_text("# fake config", encoding="utf-8")
    checkpoint.write_bytes(b"fake checkpoint")
    return repo, config, checkpoint


def test_sam3_segmenter_text_prompt_outputs_segment_detection(tmp_path: Path) -> None:
    clear_modules()
    repo, model_dir = make_sam3_repo(tmp_path)
    segmenter = Sam3Segmenter(repo_dir=repo, model_dir=model_dir, text_prompt="box .", device="cpu")

    detections = segmenter.detect(Image.new("RGB", (4, 4), "white"))

    assert len(detections) == 1
    assert detections[0].detector_label == "box ."
    assert detections[0].detector_confidence == 0.91
    assert detections[0].bbox_xyxy == (1.0, 1.0, 3.0, 3.0)


def test_groundingdino_sam3_segmenter_refines_boxes_to_masks(tmp_path: Path) -> None:
    clear_modules()
    sam3_repo, sam3_model = make_sam3_repo(tmp_path)
    gdino_repo, gdino_config, gdino_checkpoint = make_groundingdino_repo(tmp_path)
    segmenter = GroundingDinoSam3Segmenter(
        groundingdino_repo_dir=gdino_repo,
        groundingdino_config=gdino_config,
        groundingdino_checkpoint=gdino_checkpoint,
        sam3_repo_dir=sam3_repo,
        sam3_model_dir=sam3_model,
        text_prompt="box .",
        device="cpu",
    )

    detections = segmenter.detect(Image.new("RGB", (100, 80), "white"))

    assert len(detections) == 1
    assert detections[0].detector_label == "box"
    assert detections[0].detector_confidence == 0.95
    assert detections[0].mask_polygon == [(25.0, 20.0), (75.0, 20.0), (75.0, 60.0), (25.0, 60.0)]
    processor = segmenter.sam3._processor
    assert processor.prompts == [([0.5, 0.5, 0.5, 0.5], True)]
