from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from Segmentation.types import SegmentDetection
from ShapeDetection.primitive_labels import PRIMITIVE_LABELS, PRIMITIVE_PROMPTS
from ShapeDetection.types import PrimitivePrediction


class ClipPrimitiveClassifier:
    def __init__(self, model_dir: str | Path, device: str | None = None) -> None:
        self.model_dir = Path(model_dir)
        if not self.model_dir.is_dir():
            raise ValueError(f"CLIP model directory does not exist: {self.model_dir}")

        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise ImportError(
                "torch and transformers are required for --backend real. Install requirements.txt first."
            ) from exc

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPProcessor.from_pretrained(
            str(self.model_dir),
            local_files_only=True,
        )
        self.model = CLIPModel.from_pretrained(
            str(self.model_dir),
            local_files_only=True,
        ).to(self.device)
        self.model.eval()
        self.labels = list(PRIMITIVE_LABELS)
        self.prompts = [PRIMITIVE_PROMPTS[label] for label in self.labels]

    def classify(self, image: Image.Image, detection: SegmentDetection) -> PrimitivePrediction:
        crop = crop_detection(image, detection)
        inputs = self.processor(
            text=self.prompts,
            images=crop,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = outputs.logits_per_image.softmax(dim=1)[0]
        best_index = int(probabilities.argmax().detach().cpu().item())
        confidence = float(probabilities[best_index].detach().cpu().item())
        return PrimitivePrediction(label=self.labels[best_index], confidence=confidence)


def crop_detection(image: Image.Image, detection: SegmentDetection) -> Image.Image:
    left, top, right, bottom = [int(round(value)) for value in detection.bbox_xyxy]
    left = max(0, min(image.width, left))
    top = max(0, min(image.height, top))
    right = max(left + 1, min(image.width, right))
    bottom = max(top + 1, min(image.height, bottom))

    if not detection.mask_polygon:
        return image.crop((left, top, right, bottom))

    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(detection.mask_polygon, fill=255)
    white = Image.new("RGB", image.size, "white")
    masked = Image.composite(image, white, mask)
    return masked.crop((left, top, right, bottom))
