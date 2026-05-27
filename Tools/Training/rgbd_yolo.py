from __future__ import annotations

import json
from pathlib import Path

from Segmentation.rgbd_yolo_segmenter import apply_rgbd_channel_weights, parse_channel_weights


def patch_ultralytics_for_rgbd(channel_weights: str | tuple[float, float, float, float] | None = None) -> None:
    import cv2
    from ultralytics.data import base as ultralytics_base
    from ultralytics.data import loaders as ultralytics_loaders

    weights = parse_channel_weights(channel_weights)
    ultralytics_base.BaseDataset._sceneforge_rgbd_channel_weights = weights
    ultralytics_loaders.LoadImagesAndVideos._sceneforge_rgbd_channel_weights = weights

    original_base_init = getattr(
        ultralytics_base.BaseDataset,
        "_sceneforge_original_init",
        ultralytics_base.BaseDataset.__init__,
    )
    ultralytics_base.BaseDataset._sceneforge_original_init = original_base_init

    def base_init_rgbd(self, *args, channels: int = 3, **kwargs):
        original_base_init(self, *args, channels=channels, **kwargs)
        if channels == 4:
            self.cv2_flag = cv2.IMREAD_UNCHANGED

    ultralytics_base.BaseDataset.__init__ = base_init_rgbd
    original_load_image = getattr(
        ultralytics_base.BaseDataset,
        "_sceneforge_original_load_image",
        ultralytics_base.BaseDataset.load_image,
    )
    ultralytics_base.BaseDataset._sceneforge_original_load_image = original_load_image

    def load_image_rgbd(self, *args, **kwargs):
        image, original_shape, resized_shape = original_load_image(self, *args, **kwargs)
        if getattr(self, "cv2_flag", None) == cv2.IMREAD_UNCHANGED:
            image = apply_rgbd_channel_weights(
                image,
                getattr(self, "_sceneforge_rgbd_channel_weights", weights),
            )
        return image, original_shape, resized_shape

    ultralytics_base.BaseDataset.load_image = load_image_rgbd

    original_load_images_init = getattr(
        ultralytics_loaders.LoadImagesAndVideos,
        "_sceneforge_original_init",
        ultralytics_loaders.LoadImagesAndVideos.__init__,
    )
    ultralytics_loaders.LoadImagesAndVideos._sceneforge_original_init = original_load_images_init

    def load_images_init_rgbd(self, *args, channels: int = 3, **kwargs):
        original_load_images_init(self, *args, channels=channels, **kwargs)
        if channels == 4:
            self.cv2_flag = cv2.IMREAD_UNCHANGED

    ultralytics_loaders.LoadImagesAndVideos.__init__ = load_images_init_rgbd
    original_next = getattr(
        ultralytics_loaders.LoadImagesAndVideos,
        "_sceneforge_original_next",
        ultralytics_loaders.LoadImagesAndVideos.__next__,
    )
    ultralytics_loaders.LoadImagesAndVideos._sceneforge_original_next = original_next

    def next_rgbd(self):
        paths, images, info = original_next(self)
        if getattr(self, "cv2_flag", None) == cv2.IMREAD_UNCHANGED:
            images = [
                apply_rgbd_channel_weights(
                    image,
                    getattr(self, "_sceneforge_rgbd_channel_weights", weights),
                )
                for image in images
            ]
        return paths, images, info

    ultralytics_loaders.LoadImagesAndVideos.__next__ = next_rgbd


def train_rgbd_yolo(
    data_yaml: str | Path,
    model_yaml: str | Path,
    output_weights: str | Path,
    epochs: int,
    imgsz: int,
    batch: str | int,
    device: str | None,
    seed: int,
    patience: int,
    lr0: float | None = None,
    resume_from: str | Path | None = None,
    resume: bool = False,
    channel_weights: str | tuple[float, float, float, float] | None = None,
) -> Path:
    normalized_channel_weights = parse_channel_weights(channel_weights)
    patch_ultralytics_for_rgbd(normalized_channel_weights)

    from ultralytics import YOLO

    if resume and not resume_from:
        raise ValueError("--resume requires --resume-from so Ultralytics can restore the prior run state.")

    if resume_from:
        model = YOLO(str(resume_from))
    else:
        model = YOLO(str(model_yaml))

    project = Path(output_weights).parent.resolve()
    run_name = Path(output_weights).stem
    project.mkdir(parents=True, exist_ok=True)
    train_kwargs = {
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "project": str(project),
        "name": run_name,
        "seed": seed,
        "patience": patience,
        "pretrained": False,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
    }
    if resume:
        train_kwargs["resume"] = True
    if lr0 is not None:
        train_kwargs["lr0"] = lr0
    if device:
        train_kwargs["device"] = device

    result = model.train(**train_kwargs)
    save_dir = Path(getattr(result, "save_dir", project / run_name))
    best = save_dir / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"Training finished but best checkpoint was not found: {best}")

    output_path = Path(output_weights)
    if best.resolve() != output_path.resolve():
        output_path.write_bytes(best.read_bytes())
    return output_path


def evaluate_rgbd_yolo(
    data_yaml: str | Path,
    weights_path: str | Path,
    output_dir: str | Path,
    imgsz: int,
    batch: str | int,
    device: str | None,
    split: str = "test",
    channel_weights: str | tuple[float, float, float, float] | None = None,
) -> Path:
    normalized_channel_weights = parse_channel_weights(channel_weights)
    patch_ultralytics_for_rgbd(normalized_channel_weights)

    from ultralytics import YOLO

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights_path))
    kwargs = {
        "data": str(data_yaml),
        "split": split,
        "imgsz": imgsz,
        "batch": batch,
        "project": str(output_path.parent.resolve()),
        "name": output_path.name,
        "plots": True,
        "exist_ok": True,
    }
    if device:
        kwargs["device"] = device
    metrics = model.val(**kwargs)
    results = {
        "data": str(data_yaml),
        "weights": str(weights_path),
        "split": split,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "rgbd_channel_weights": [round(float(value), 6) for value in normalized_channel_weights],
        "results_dir": str(getattr(metrics, "save_dir", output_path)),
        "box_map50": float(getattr(metrics.box, "map50", 0.0)) if getattr(metrics, "box", None) else 0.0,
        "box_map50_95": float(getattr(metrics.box, "map", 0.0)) if getattr(metrics, "box", None) else 0.0,
        "mask_map50": float(getattr(metrics.seg, "map50", 0.0)) if getattr(metrics, "seg", None) else 0.0,
        "mask_map50_95": float(getattr(metrics.seg, "map", 0.0)) if getattr(metrics, "seg", None) else 0.0,
    }
    summary_path = output_path / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary_path
