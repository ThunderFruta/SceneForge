from __future__ import annotations

from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
from PIL import Image

from Segmentation.backend import SegmentationBackendInfo
from Segmentation.sam3_segmenter import Sam3Segmenter
from Segmentation.types import SegmentDetection


class GroundingDinoSam3Segmenter:
    """GroundingDINO box proposals refined by SAM3 masks.

    GroundingDINO owns open-vocabulary detection from text. SAM3 owns mask
    refinement when its local API exposes box prompts; otherwise SceneForge keeps
    the GroundingDINO rectangle proposals so the adapter remains inspectable.
    """

    backend = "groundingdino-sam3-open-vocabulary"

    def __init__(
        self,
        groundingdino_repo_dir: str | Path,
        groundingdino_config: str | Path,
        groundingdino_checkpoint: str | Path,
        sam3_repo_dir: str | Path,
        sam3_model_dir: str | Path,
        text_prompt: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        score_threshold: float = 0.25,
        device: str | None = None,
    ) -> None:
        self.groundingdino_repo_dir = Path(groundingdino_repo_dir)
        self.groundingdino_config = Path(groundingdino_config)
        self.groundingdino_checkpoint = Path(groundingdino_checkpoint)
        self.text_prompt = text_prompt
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.score_threshold = float(score_threshold)
        self.device = device
        if not self.groundingdino_repo_dir.is_dir():
            raise ValueError(f"--groundingdino-repo-dir does not exist or is not a directory: {self.groundingdino_repo_dir}")
        if not self.groundingdino_config.is_file():
            raise ValueError(f"--groundingdino-config does not exist or is not a file: {self.groundingdino_config}")
        if not self.groundingdino_checkpoint.is_file():
            raise ValueError(f"--groundingdino-checkpoint does not exist or is not a file: {self.groundingdino_checkpoint}")
        self.sam3 = Sam3Segmenter(
            repo_dir=sam3_repo_dir,
            model_dir=sam3_model_dir,
            text_prompt=text_prompt,
            score_threshold=score_threshold,
            device=device,
        )
        self.backend_info = SegmentationBackendInfo(
            name=self.backend,
            architecture="groundingdino_boxes_plus_sam3_masks",
            input_channels=("rgb", "text_prompt"),
            primitive_labels_are_authoritative=False,
            legacy=False,
            model_path=str(self.groundingdino_checkpoint),
            proposal_only=True,
            output_contract="open_vocab_box_guided_instance_masks",
            primitive_label_policy="geometry_fitting_downstream",
            notes="IDEA-Research/GroundingDINO proposes boxes; facebookresearch/sam3 refines masks when box prompts are available.",
        )
        self._grounding_model = None

    def detect(self, image: Image.Image) -> list[SegmentDetection]:
        boxes, labels, scores = self._groundingdino_boxes(image)
        if not boxes:
            return []
        return self.sam3.detect_boxes(image, boxes, labels, scores)

    def _groundingdino_boxes(
        self,
        image: Image.Image,
    ) -> tuple[list[tuple[float, float, float, float]], list[str], list[float]]:
        load_image, predict = self._load_groundingdino_functions()
        with tempfile.NamedTemporaryFile(prefix="sceneforge_groundingdino_", suffix=".png") as temp_file:
            image.convert("RGB").save(temp_file.name)
            _, image_tensor = load_image(temp_file.name)
        boxes, logits, phrases = predict(
            model=self._grounding_model,
            image=image_tensor,
            caption=self.text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device or "cpu",
        )
        boxes_xyxy = [cxcywh_to_xyxy(box, image.size) for box in _to_numpy(boxes)]
        labels = [str(item) for item in phrases]
        scores = [float(value) for value in _to_numpy(logits).reshape(-1)]
        filtered = [
            (box, label or "object", score)
            for box, label, score in zip(boxes_xyxy, labels, scores)
            if score >= self.score_threshold
        ]
        return (
            [item[0] for item in filtered],
            [item[1] for item in filtered],
            [item[2] for item in filtered],
        )

    def _load_groundingdino_functions(self) -> tuple[Any, Any]:
        if self._grounding_model is None:
            if str(self.groundingdino_repo_dir) not in sys.path:
                sys.path.insert(0, str(self.groundingdino_repo_dir))
            from groundingdino.util.inference import load_image, load_model, predict

            self._grounding_model = load_model(str(self.groundingdino_config), str(self.groundingdino_checkpoint))
            if self.device:
                try:
                    self._grounding_model.to(self.device)
                except AttributeError:
                    pass
            return load_image, predict
        from groundingdino.util.inference import load_image, predict

        return load_image, predict


def cxcywh_to_xyxy(box: Any, image_size: tuple[int, int]) -> tuple[float, float, float, float]:
    width, height = image_size
    cx, cy, w, h = [float(value) for value in _to_numpy(box).reshape(-1)[:4]]
    if max(abs(cx), abs(cy), abs(w), abs(h)) <= 1.5:
        cx, w = cx * width, w * width
        cy, h = cy * height, h * height
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)
