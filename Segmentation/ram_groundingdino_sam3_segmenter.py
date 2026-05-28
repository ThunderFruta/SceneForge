from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from Segmentation.backend import SegmentationBackendInfo
from Segmentation.groundingdino_sam3_segmenter import GroundingDinoSam3Segmenter
from Segmentation.sam3_segmenter import detection_from_box
from Segmentation.types import SegmentDetection


@dataclass(frozen=True)
class RamProposal:
    box_xyxy: tuple[float, float, float, float]
    label: str
    score: float


class RamGroundingDinoSam3Segmenter:
    """RAM-like proposals -> GroundingDINO label refinement -> SAM3 box segmentation."""

    backend = "ram-groundingdino-sam3-open-vocabulary"

    def __init__(
        self,
        ram_repo_dir: str | Path,
        ram_checkpoint: str | Path,
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
        self.ram_repo_dir = Path(ram_repo_dir)
        self.ram_checkpoint = Path(ram_checkpoint)
        self.text_prompt = text_prompt
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.score_threshold = float(score_threshold)
        self.device = device
        if not self.ram_repo_dir.is_dir():
            raise ValueError(f"--ram-repo-dir does not exist or is not a directory: {self.ram_repo_dir}")
        if not self.ram_checkpoint.is_file():
            raise ValueError(f"--ram-checkpoint does not exist or is not a file: {self.ram_checkpoint}")

        self.groundingdino = GroundingDinoSam3Segmenter(
            groundingdino_repo_dir=groundingdino_repo_dir,
            groundingdino_config=groundingdino_config,
            groundingdino_checkpoint=groundingdino_checkpoint,
            sam3_repo_dir=sam3_repo_dir,
            sam3_model_dir=sam3_model_dir,
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            score_threshold=score_threshold,
            device=device,
        )
        self.sam3 = self.groundingdino.sam3
        self.backend_info = SegmentationBackendInfo(
            name=self.backend,
            architecture="ram_boxes_plus_groundingdino_plus_sam3_masks",
            input_channels=("rgb", "ram_checkpoint", "text_prompt"),
            primitive_labels_are_authoritative=False,
            legacy=False,
            model_path=str(self.ram_checkpoint),
            proposal_only=True,
            output_contract="open_vocab_box_guided_instance_masks",
            primitive_label_policy="geometry_fitting_downstream",
            notes=(
                "RAM proposals guide GroundingDINO box generation, then SAM3 refines"
                " box proposals into masks."
            ),
        )
        self._ram_predictor = None
        self._ram_model_ready = False

    def detect(self, image: Image.Image) -> list[SegmentDetection]:
        proposals = self._ram_proposals(image)
        if not proposals:
            proposals = [
                RamProposal(box, label, score)
                for box, label, score in self._groundingdino_boxes(image)
                if score >= self.score_threshold
            ]
            if not proposals:
                return []
            fallback_source = "groundingdino_fallback"
        else:
            fallback_source = "ram"

        boxes, labels, scores = self._align_grounding_to_ram(image, proposals)
        refined = self.sam3.detect_boxes(image, boxes, labels, scores)
        if refined:
            return [self._with_proposal_source(item, self._proposed_source(item, fallback_source)) for item in refined]

        return [
            detection_from_box(
                box=proposal.box_xyxy,
                label=proposal.label,
                score=proposal.score,
                image_size=image.size,
                proposal_source=f"{fallback_source}_sam3_fallback",
            )
            for proposal in proposals
            if proposal.score >= self.score_threshold
        ]

    def _proposed_source(self, detection: SegmentDetection, fallback_source: str) -> str:
        if detection.proposal_source == "groundingdino_box_fallback":
            return "ram_groundingdino_box_fallback"
        if detection.proposal_source:
            return "ram_" + detection.proposal_source
        return f"{fallback_source}_ram_sam3_refined"

    @staticmethod
    def _with_proposal_source(detection: SegmentDetection, source: str) -> SegmentDetection:
        return SegmentDetection(
            bbox_xyxy=detection.bbox_xyxy,
            mask_polygon=detection.mask_polygon,
            detector_label=detection.detector_label,
            detector_confidence=detection.detector_confidence,
            proposal_source=source,
        )

    def _ram_proposals(self, image: Image.Image) -> list[RamProposal]:
        predictor = self._load_ram_predictor()
        if predictor is None:
            return []
        boxes, labels, scores = predictor(image, self.text_prompt)
        proposals: list[RamProposal] = []
        for box, label, score in zip(boxes, labels, scores):
            score_value = float(score)
            if score_value < self.box_threshold:
                continue
            box_xyxy = _to_xyxy_tuple(box, image.size)
            proposals.append(RamProposal(box_xyxy=box_xyxy, label=str(label), score=score_value))
        return proposals

    def _align_grounding_to_ram(
        self,
        image: Image.Image,
        ram_proposals: list[RamProposal],
    ) -> tuple[list[tuple[float, float, float, float]], list[str], list[float]]:
        gdino_boxes, gdino_labels, gdino_scores = self._groundingdino_boxes(image)
        if not gdino_boxes:
            return [item.box_xyxy for item in ram_proposals], [item.label for item in ram_proposals], [item.score for item in ram_proposals]

        boxes: list[tuple[float, float, float, float]] = []
        labels: list[str] = []
        scores: list[float] = []
        for proposal in ram_proposals:
            best_index = None
            best_iou = 0.0
            for index, box in enumerate(gdino_boxes):
                overlap = _iou(proposal.box_xyxy, box)
                if overlap > best_iou:
                    best_iou = overlap
                    best_index = index
            if best_index is not None and best_iou >= 0.2:
                boxes.append(gdino_boxes[best_index])
                labels.append(gdino_labels[best_index] or proposal.label)
                scores.append(max(float(gdino_scores[best_index]), proposal.score))
            else:
                boxes.append(proposal.box_xyxy)
                labels.append(proposal.label)
                scores.append(proposal.score)

        if not boxes:
            return [], [], []
        return _deduplicate_boxes(boxes, labels, scores, threshold=0.8)

    def _groundingdino_boxes(self, image: Image.Image) -> tuple[list[tuple[float, float, float, float]], list[str], list[float]]:
        return self.groundingdino._groundingdino_boxes(image)

    def _load_ram_predictor(self):
        if self._ram_model_ready:
            return self._ram_predictor

        if str(self.ram_repo_dir) not in sys.path:
            sys.path.insert(0, str(self.ram_repo_dir))

        for attempt in (
            self._build_ram_predictor_from_ram,
            self._build_ram_predictor_from_ram_inference,
        ):
            predictor = attempt()
            if predictor is not None:
                self._ram_model_ready = True
                self._ram_predictor = predictor
                return predictor

        self._ram_model_ready = True
        self._ram_predictor = None
        return None

    def _build_ram_predictor_from_ram(self):
        try:
            from ram import RAM

            return _RamModelAdapter(model=RAM(checkpoint=self.ram_checkpoint, device=self.device))
        except Exception:
            return None

    def _build_ram_predictor_from_ram_inference(self):
        try:
            from ram_inference import RAMModel

            return _RamModelAdapter(model=RAMModel(checkpoint=self.ram_checkpoint, device=self.device))
        except Exception:
            return None


@dataclass
class _RamModelAdapter:
    model: Any

    def __call__(self, image: Image.Image, prompt: str) -> tuple[list[tuple[float, float, float, float]], list[str], list[float]]:
        if not hasattr(self.model, "predict"):
            return [], [], []
        outputs = self.model.predict(image, prompt=prompt)
        if not outputs:
            return [], [], []

        triplet = _extract_triplet_outputs(outputs)
        if triplet is not None:
            boxes, labels, scores = triplet
            return list(_normalize_triplet_boxes(boxes, labels, scores))

        if not isinstance(outputs, (list, tuple)):
            return [], [], []
        boxes = []
        labels = []
        scores = []
        for output in outputs:
            box = _get_value(output, "box", None)
            if box is None:
                continue
            values = _flatten_box_values(box)
            if len(values) < 4:
                continue
            boxes.append((values[0], values[1], values[2], values[3]))
            labels.append(str(_get_value(output, "label", "object")))
            scores.append(float(_get_value(output, "score", 0.0)))
        return boxes, labels, scores


def _extract_triplet_outputs(outputs: Any) -> tuple[Any, Any, Any] | None:
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 3:
        return None
    if any(isinstance(item, (dict, list, tuple)) for item in outputs):
        return None
    return tuple(outputs)  # type: ignore[return-value]


def _normalize_triplet_boxes(
    boxes: Any,
    labels: Any,
    scores: Any,
) -> tuple[tuple[float, float, float, float], str, float]:
    box_rows = np.asarray(boxes).reshape(-1, 4)
    label_rows = np.asarray(labels)
    score_rows = np.asarray(scores).reshape(-1)
    limit = min(len(box_rows), len(label_rows), len(score_rows))
    for index in range(limit):
        box = _to_xyxy_tuple(box_rows[index], (1, 1))
        label = str(label_rows[index]) if hasattr(label_rows, "__len__") else str(label_rows)
        score = float(score_rows[index])
        yield (box, label, score)


def _to_xyxy_tuple(box: Any, image_size: tuple[int, int]) -> tuple[float, float, float, float]:
    if hasattr(box, "xyxy"):
        return tuple(float(v) for v in box.xyxy)
    values = [float(v) for v in np.asarray(box).reshape(-1)[:4]]
    if values:
        if all(v <= 1.5 for v in values):
            cx, cy, w, h = values[:4]
            width, height = image_size
            return (
                max(0.0, (cx - w / 2.0) * width),
                max(0.0, (cy - h / 2.0) * height),
                min(width, (cx + w / 2.0) * width),
                min(height, (cy + h / 2.0) * height),
            )
        return tuple(values[:4])
    return (0.0, 0.0, 0.0, 0.0)


def _get_value(container: Any, key: str, default: Any) -> Any:
    if isinstance(container, dict):
        return container.get(key, default)
    if hasattr(container, key):
        return getattr(container, key)
    if isinstance(container, (list, tuple)) and container:
        try:
            return container[0]
        except Exception:
            return default
    return default


def _flatten_box_values(value: Any) -> list[float]:
    if hasattr(value, "xyxy"):
        return [float(v) for v in np.asarray(value.xyxy).reshape(-1)]
    values = np.asarray(value).reshape(-1)
    return [float(v) for v in values]


def _iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _deduplicate_boxes(
    boxes: list[tuple[float, float, float, float]],
    labels: list[str],
    scores: list[float],
    *,
    threshold: float = 0.8,
) -> tuple[list[tuple[float, float, float, float]], list[str], list[float]]:
    selected_indices: set[int] = set()
    kept: list[int] = []
    for index, box in enumerate(boxes):
        if index in selected_indices:
            continue
        kept.append(index)
        for other, candidate in enumerate(boxes):
            if other <= index or other in selected_indices:
                continue
            if _iou(box, candidate) >= threshold:
                selected_indices.add(other)

    filtered_boxes: list[tuple[float, float, float, float]] = []
    filtered_labels: list[str] = []
    filtered_scores: list[float] = []
    for index in kept:
        if scores[index] < 0.0:
            continue
        if not all(math.isfinite(value) for value in boxes[index]):
            continue
        filtered_boxes.append(boxes[index])
        filtered_labels.append(labels[index])
        filtered_scores.append(scores[index])
    if not filtered_boxes:
        return [], [], []
    return filtered_boxes, filtered_labels, filtered_scores

