from __future__ import annotations

import os
from pathlib import Path
import sys
from contextlib import contextmanager
from typing import Any

import numpy as np
from PIL import Image

from Segmentation.backend import SegmentationBackendInfo
from Segmentation.types import SegmentDetection


class Sam3Segmenter:
    """Lazy adapter for facebookresearch/sam3 image promptable segmentation.

    The adapter emits SceneForge proposal masks only. It deliberately does not
    assign primitive labels; geometry/fusion remains the authority downstream.
    """

    backend = "sam3-open-vocabulary"

    def __init__(
        self,
        repo_dir: str | Path,
        model_dir: str | Path,
        text_prompt: str,
        score_threshold: float = 0.25,
        device: str | None = None,
    ) -> None:
        self.repo_dir = Path(repo_dir)
        self.model_dir = Path(model_dir)
        self.text_prompt = text_prompt
        self.score_threshold = float(score_threshold)
        self.device = device
        if not self.repo_dir.is_dir():
            raise ValueError(f"--sam3-repo-dir does not exist or is not a directory: {self.repo_dir}")
        if not self.model_dir.is_dir():
            raise ValueError(f"--sam3-model-dir does not exist or is not a directory: {self.model_dir}")
        self.backend_info = SegmentationBackendInfo(
            name=self.backend,
            architecture="sam3_promptable_concept_segmentation",
            input_channels=("rgb", "text_prompt"),
            primitive_labels_are_authoritative=False,
            legacy=False,
            model_path=str(self.model_dir),
            proposal_only=True,
            output_contract="open_vocab_instance_masks",
            primitive_label_policy="geometry_fitting_downstream",
            notes="facebookresearch/sam3 adapter; text prompt proposes masks, primitives are fitted downstream.",
        )
        self._processor = None

    def detect(self, image: Image.Image) -> list[SegmentDetection]:
        processor = self._load_processor()
        state = processor.set_image(image.convert("RGB"))
        output = processor.set_text_prompt(state=state, prompt=self.text_prompt)
        return detections_from_sam3_output(
            output,
            image.size,
            fallback_label=self.text_prompt,
            min_score=self.score_threshold,
            proposal_source="sam3_text_prompt",
        )

    def detect_boxes(
        self,
        image: Image.Image,
        boxes_xyxy: list[tuple[float, float, float, float]],
        labels: list[str],
        scores: list[float],
    ) -> list[SegmentDetection]:
        processor = self._load_processor()
        state = processor.set_image(image.convert("RGB"))
        detections: list[SegmentDetection] = []
        for box, label, score in zip(boxes_xyxy, labels, scores):
            self._reset_prompts(processor, state)
            output = self._try_box_prompt(processor, state, box)
            if output is None:
                detections.append(detection_from_box(box, label, score, image.size, proposal_source="groundingdino_box_fallback"))
                continue
            refined = detections_from_sam3_output(
                output,
                image.size,
                fallback_label=label,
                min_score=0.0,
                proposal_source="sam3_box_prompt",
            )
            if refined:
                best = max(refined, key=lambda item: item.detector_confidence)
                detections.append(
                    SegmentDetection(
                        bbox_xyxy=best.bbox_xyxy,
                        mask_polygon=best.mask_polygon,
                        detector_label=label or best.detector_label,
                        detector_confidence=max(float(score), best.detector_confidence),
                        proposal_source=best.proposal_source,
                    )
                )
            else:
                detections.append(detection_from_box(box, label, score, image.size, proposal_source="groundingdino_box_fallback"))
        return detections

    @staticmethod
    def _reset_prompts(processor: Any, state: Any) -> None:
        reset = getattr(processor, "reset_all_prompts", None)
        if reset is None:
            return
        try:
            reset(state)
        except TypeError:
            return

    def _try_box_prompt(self, processor: Any, state: Any, box: tuple[float, float, float, float]) -> Any | None:
        method = getattr(processor, "add_geometric_prompt", None)
        if method is not None:
            try:
                return method(box=xyxy_to_normalized_cxcywh(box, state), label=True, state=state)
            except RuntimeError:
                return None
            except TypeError:
                try:
                    return method(xyxy_to_normalized_cxcywh(box, state), True, state)
                except RuntimeError:
                    return None
                except TypeError:
                    pass

        for method_name in ("set_box_prompt", "set_visual_prompt", "set_boxes"):
            method = getattr(processor, method_name, None)
            if method is None:
                continue
            for kwargs in (
                {"state": state, "box": list(box)},
                {"state": state, "boxes": [list(box)]},
                {"inference_state": state, "box": list(box)},
                {"inference_state": state, "boxes": [list(box)]},
            ):
                try:
                    return method(**kwargs)
                except RuntimeError:
                    return None
                except TypeError:
                    continue
        return None

    def _load_processor(self):
        if self._processor is not None:
            return self._processor
        if str(self.repo_dir) not in sys.path:
            sys.path.insert(0, str(self.repo_dir))
        os.environ.setdefault("HF_HOME", str(self.model_dir))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(self.model_dir / "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(self.model_dir / "transformers"))
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        model_device = self.device or "cuda"
        with _sam3_cuda_literal_device_patch(model_device):
            model = build_sam3_image_model(device=model_device)
        if self.device:
            try:
                model.to(self.device)
            except AttributeError:
                pass
        try:
            self._processor = Sam3Processor(model, device=self.device or "cuda")
        except TypeError:
            self._processor = Sam3Processor(model)
        return self._processor


@contextmanager
def _sam3_cuda_literal_device_patch(device: str):
    if str(device).startswith("cuda"):
        yield
        return

    import torch

    original_zeros = torch.zeros
    original_arange = torch.arange

    def zeros_with_device_redirect(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = device
        return original_zeros(*args, **kwargs)

    def arange_with_device_redirect(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = device
        return original_arange(*args, **kwargs)

    torch.zeros = zeros_with_device_redirect
    torch.arange = arange_with_device_redirect
    try:
        yield
    finally:
        torch.zeros = original_zeros
        torch.arange = original_arange


def xyxy_to_normalized_cxcywh(box: tuple[float, float, float, float], state: Any) -> list[float]:
    width = float(state.get("original_width", 1) or 1) if isinstance(state, dict) else 1.0
    height = float(state.get("original_height", 1) or 1) if isinstance(state, dict) else 1.0
    left, top, right, bottom = box
    center_x = ((left + right) / 2.0) / width
    center_y = ((top + bottom) / 2.0) / height
    box_width = max(0.0, right - left) / width
    box_height = max(0.0, bottom - top) / height
    return [
        max(0.0, min(1.0, center_x)),
        max(0.0, min(1.0, center_y)),
        max(0.0, min(1.0, box_width)),
        max(0.0, min(1.0, box_height)),
    ]


def detections_from_sam3_output(
    output: dict[str, Any],
    image_size: tuple[int, int],
    *,
    fallback_label: str,
    min_score: float,
    proposal_source: str = "sam3_output",
) -> list[SegmentDetection]:
    masks = _output_value(output, "masks")
    boxes = _output_value(output, "boxes")
    scores = _output_value(output, "scores")
    labels = _output_value(output, "labels")
    if _length(labels) == 0:
        labels = _output_value(output, "phrases")
    count = max(_length(masks), _length(boxes), _length(scores), _length(labels))
    detections: list[SegmentDetection] = []
    for index in range(count):
        score = _item_float(scores, index, default=1.0)
        if score < min_score:
            continue
        label = str(_item(labels, index, fallback_label) or fallback_label)
        mask = _item(masks, index, None)
        box = _item(boxes, index, None)
        if mask is not None:
            mask_array = _mask_array(mask)
            mask_array = mask_for_image_size(mask_array, image_size)
            bbox = box_xyxy_from_mask(mask_array, image_size)
            if bbox is None and box is not None:
                bbox = normalize_box_xyxy(box, image_size)
            polygon = polygon_from_mask_bbox(mask_array, bbox, image_size) if bbox else []
        elif box is not None:
            bbox = normalize_box_xyxy(box, image_size)
            polygon = rectangle_polygon(bbox)
        else:
            continue
        if not polygon:
            continue
        detections.append(
            SegmentDetection(
                bbox_xyxy=bbox,
                mask_polygon=polygon,
                detector_label=label,
                detector_confidence=score,
                proposal_source=proposal_source,
            ).normalized(*image_size)
        )
    return detections


def detection_from_box(
    box: tuple[float, float, float, float],
    label: str,
    score: float,
    image_size: tuple[int, int],
    proposal_source: str = "box_fallback",
) -> SegmentDetection:
    bbox = normalize_box_xyxy(box, image_size)
    return SegmentDetection(
        bbox_xyxy=bbox,
        mask_polygon=rectangle_polygon(bbox),
        detector_label=label or "object",
        detector_confidence=float(score),
        proposal_source=proposal_source,
    ).normalized(*image_size)


def _mask_array(mask: Any) -> np.ndarray:
    values = _to_numpy(mask)
    values = np.squeeze(values)
    if values.ndim > 2:
        values = values[0]
    return values.astype(np.float32) > 0.5


def mask_for_image_size(mask: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    if mask.shape == (height, width):
        return mask
    resized = Image.fromarray(mask.astype(np.uint8) * 255).resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(resized) > 0


def box_xyxy_from_mask(mask: np.ndarray, image_size: tuple[int, int]) -> tuple[float, float, float, float] | None:
    if mask.size == 0 or not bool(mask.any()):
        return None
    rows, cols = np.where(mask)
    return normalize_box_xyxy((float(cols.min()), float(rows.min()), float(cols.max() + 1), float(rows.max() + 1)), image_size)


def polygon_from_mask_bbox(
    mask: np.ndarray,
    bbox: tuple[float, float, float, float] | None,
    image_size: tuple[int, int],
) -> list[tuple[float, float]]:
    if bbox is None or mask.size == 0 or not bool(mask.any()):
        return []
    try:
        from skimage import measure
    except ImportError:
        return rectangle_polygon(bbox)

    contours = measure.find_contours(mask.astype(np.uint8), 0.5)
    if not contours:
        return rectangle_polygon(bbox)
    contour = max(contours, key=len)
    if len(contour) < 3:
        return rectangle_polygon(bbox)

    points = [(float(col), float(row)) for row, col in contour]
    points = simplify_polygon(points, tolerance=max(image_size) * 0.0025)
    if len(points) < 3:
        return rectangle_polygon(bbox)
    return points[:128]


def simplify_polygon(points: list[tuple[float, float]], tolerance: float) -> list[tuple[float, float]]:
    try:
        from shapely.geometry import Polygon
    except ImportError:
        return points
    polygon = Polygon(points)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty:
        return points
    simplified = polygon.simplify(tolerance, preserve_topology=True)
    if simplified.geom_type == "MultiPolygon":
        simplified = max(simplified.geoms, key=lambda item: item.area)
    if simplified.geom_type != "Polygon":
        return points
    return [(float(x), float(y)) for x, y in list(simplified.exterior.coords)[:-1]]


def normalize_box_xyxy(box: Any, image_size: tuple[int, int]) -> tuple[float, float, float, float]:
    width, height = image_size
    values = [float(value) for value in np.asarray(_to_numpy(box)).reshape(-1)[:4]]
    if len(values) < 4:
        return (0.0, 0.0, 0.0, 0.0)
    left, top, right, bottom = values
    if max(abs(left), abs(top), abs(right), abs(bottom)) <= 1.5:
        left, right = left * width, right * width
        top, bottom = top * height, bottom * height
    if right < left:
        left, right = right, left
    if bottom < top:
        top, bottom = bottom, top
    return (left, top, right, bottom)


def rectangle_polygon(bbox: tuple[float, float, float, float]) -> list[tuple[float, float]]:
    left, top, right, bottom = bbox
    return [(left, top), (right, top), (right, bottom), (left, bottom)]


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _output_value(output: dict[str, Any], key: str) -> Any:
    value = output.get(key)
    return [] if value is None else value


def _length(value: Any) -> int:
    try:
        return len(value)
    except TypeError:
        return 0


def _item(value: Any, index: int, default: Any) -> Any:
    if _length(value) <= index:
        return default
    return value[index]


def _item_float(value: Any, index: int, default: float) -> float:
    item = _item(value, index, default)
    try:
        return float(_to_numpy(item).reshape(-1)[0])
    except Exception:
        return float(default)
