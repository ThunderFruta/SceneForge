"""Run a YOLO segmentation model on one image and export mask images.

Usage:
    python3 Tests/Segmentation/run_yolo_mask_test.py \
        --image Input/example.png \
        --weights Models/YOLO/sceneforge-primitives-yolo11m-seg.pt \
        --output-dir Tests/Output/yolo_masks
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO segmentation on an image and export instance masks."
    )
    parser.add_argument("--image", required=True, help="Path to an input RGB image.")
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to local YOLO segmentation weights.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where mask PNGs will be written.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="YOLO confidence threshold for detections.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, for example cpu or cuda. Leave empty for default.",
    )
    return parser.parse_args()


def _to_mask_image(data: np.ndarray) -> Image.Image:
    # Keep the mask grayscale: 0/255.
    return Image.fromarray((data >= 0.5).astype("uint8") * 255, mode="L")


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    weights_path = Path(args.weights)
    output_dir = Path(args.output_dir)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights file does not exist: {weights_path}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - environment-specific dependency
        raise ImportError(
            "ultralytics is required to run this test script. Install requirements.txt."
        ) from exc

    model = YOLO(str(weights_path))
    results = model.predict(
        source=str(image_path),
        conf=args.confidence,
        device=args.device,
        verbose=False,
    )

    if not results:
        print("No prediction result returned by YOLO.")
        return

    result = results[0]
    names = getattr(result, "names", {}) or {}
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)

    if masks is None or masks.data is None:
        print("YOLO returned no masks for this image.")
        return

    mask_arrays = masks.data.detach().cpu().numpy()
    if boxes is None or len(boxes) == 0:
        print("YOLO returned masks but no box metadata for labels. Still exporting masks only.")

    output_dir.mkdir(parents=True, exist_ok=True)

    combined = np.zeros(mask_arrays.shape[1:], dtype=np.float32)
    for index in range(len(mask_arrays)):
        current = mask_arrays[index]
        combined = np.maximum(combined, current)

        label = "unknown"
        conf = 0.0
        if boxes is not None and len(boxes) > index:
            class_id = int(boxes[index].cls[0].detach().cpu().item())
            conf = float(boxes[index].conf[0].detach().cpu().item())
            label = str(names.get(class_id, class_id))
        _to_mask_image(current).save(
            output_dir / f"mask_{index + 1:02d}_{label}_{conf:.2f}.png",
        )

    _to_mask_image(combined).save(output_dir / "combined_mask.png")
    print(f"Wrote {len(mask_arrays)} instance masks to {output_dir}")
    print(f"Wrote combined mask to {output_dir / 'combined_mask.png'}")


if __name__ == "__main__":
    main()
