from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from PrimitiveFitting.masks import polygon_to_mask
from Segmentation.primitive_3d import (
    FEATURE_NAMES,
    IMAGE_FEATURE_NAMES,
    Primitive3DConfig,
    Primitive3DSegNet,
    RGBDMaskSegNet,
    build_cache_from_paths,
    build_rgbd_mask_arrays_from_paths,
    default_checkpoint_metadata,
    load_checkpoint,
    load_point_cloud_cache,
    segment_detections_from_arrays,
    segment_detections_from_maps,
)


EXPECTED_INPUT_CONTRACT = "rgb_depth_camera_to_visible_point_cloud"
EXPECTED_OUTPUT_CONTRACT = "class_agnostic_instance_masks"
DEFAULT_ARCHITECTURE = "primitive_3d_point_embedding_v1"
DEFAULT_CONFIG_PATH = "Configs/InstanceDetector/primitive_3d_segmentation.json"


def load_instance_manifest(manifest_path: str | Path) -> dict[str, Any]:
    path = Path(manifest_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    contract = data.get("detector_training_contract", {})
    input_contract = contract.get("input_contract")
    input_channels = tuple(contract.get("input_channels", ()))
    output_contract = contract.get("output_contract")
    if input_contract != EXPECTED_INPUT_CONTRACT:
        raise ValueError(
            f"Expected detector manifest input_contract {EXPECTED_INPUT_CONTRACT!r}, got {input_contract!r}."
        )
    if input_channels != FEATURE_NAMES:
        raise ValueError(
            f"Expected detector manifest input_channels {FEATURE_NAMES}, got {input_channels}."
        )
    if output_contract != EXPECTED_OUTPUT_CONTRACT:
        raise ValueError(
            f"Expected detector manifest output_contract {EXPECTED_OUTPUT_CONTRACT!r}, got {output_contract!r}."
        )
    return data


def load_detector_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    input_contract = data.get("input_contract")
    input_channels = tuple(data.get("input_channels", ()))
    model_family = str(data.get("model_family", "pointnet_embedding"))
    expected_channels = IMAGE_FEATURE_NAMES if model_family == "rgbd_mask_embedding" else FEATURE_NAMES
    output_contract = data.get("output_contract")
    if input_contract != EXPECTED_INPUT_CONTRACT:
        raise ValueError(
            f"Expected detector config input_contract {EXPECTED_INPUT_CONTRACT!r}, got {input_contract!r}."
        )
    if input_channels != expected_channels:
        raise ValueError(
            f"Expected detector config input_channels {expected_channels}, got {input_channels}."
        )
    if output_contract != EXPECTED_OUTPUT_CONTRACT:
        raise ValueError(
            f"Expected detector config output_contract {EXPECTED_OUTPUT_CONTRACT!r}, got {output_contract!r}."
        )
    primitive_policy = data.get("primitive_label_policy")
    if primitive_policy != "geometry_fitting_downstream":
        raise ValueError(
            "Expected detector config primitive_label_policy 'geometry_fitting_downstream', "
            f"got {primitive_policy!r}."
        )
    return data


def write_training_scaffold(
    *,
    manifest_path: str | Path,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_dir: str | Path,
    architecture: str | None = None,
    epochs: int = 100,
    batch: int | str = 8,
    device: str | None = None,
    seed: int = 20260526,
    log_every: int = 100,
) -> Path:
    torch = import_torch()
    manifest_path = Path(manifest_path)
    config_path = Path(config_path)
    manifest = load_instance_manifest(manifest_path)
    config = load_detector_config(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    resolved_device = resolve_device(device)
    torch.manual_seed(int(seed))
    model_config = primitive_config_from_config(config)
    batch_size = max(1, int(batch))
    embedding_loss_weight = float(config.get("embedding_loss_weight", 1.0))
    if model_config.model_family == "rgbd_mask_embedding":
        model = RGBDMaskSegNet(
            input_dim=model_config.image_input_dim,
            base_channels=model_config.base_channels,
            embedding_dim=model_config.embedding_dim,
        ).to(resolved_device)
    else:
        model = Primitive3DSegNet(
            input_dim=model_config.input_dim,
            hidden_dim=model_config.hidden_dim,
            embedding_dim=model_config.embedding_dim,
            encoder_layers=model_config.encoder_layers,
            decoder_layers=model_config.decoder_layers,
            use_global_context=model_config.use_global_context,
        ).to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 0.001)))

    train_samples = (
        prepared_dense_split_samples(manifest, "train")
        if model_config.model_family == "rgbd_mask_embedding"
        else prepared_split_samples(manifest, "train", model_config)
    )
    dense_feature_names = tuple(config.get("input_channels", IMAGE_FEATURE_NAMES))
    print(
        "Training Primitive3D "
        f"samples={len(train_samples)} epochs={max(1, int(epochs))} "
        f"device={resolved_device} model_family={model_config.model_family} "
        f"batch={batch_size} embedding_loss_weight={embedding_loss_weight:g} "
        f"max_points={model_config.max_points} image_size={model_config.image_size}",
        flush=True,
    )
    metadata = default_checkpoint_metadata(model_config)
    metadata["architecture"] = architecture or config.get("architecture", DEFAULT_ARCHITECTURE)
    metadata["config_name"] = config.get("name")
    metadata["model"]["objectness_threshold"] = float(config.get("objectness_threshold", 0.5))
    metadata["training"] = {
        "manifest_path": str(manifest_path),
        "config_path": str(config_path),
        "epochs": int(epochs),
        "seed": int(seed),
        "device": resolved_device,
        "batch": batch_size,
        "embedding_loss_weight": embedding_loss_weight,
        "log_every": int(log_every),
    }
    checkpoint_path = output / "primitive_3d_segmenter.pt"
    latest_checkpoint_path = output / "primitive_3d_segmenter_latest.pt"
    losses: list[float] = []
    objectness_losses: list[float] = []
    embedding_losses: list[float] = []
    epoch_summaries: list[dict[str, Any]] = []
    model.train()
    total_epochs = max(1, int(epochs))
    progress_interval = max(0, int(log_every))
    for epoch_index in range(total_epochs):
        epoch_losses: list[float] = []
        epoch_objectness_losses: list[float] = []
        epoch_embedding_losses: list[float] = []
        epoch_foreground_accuracy: list[float] = []
        epoch_background_accuracy: list[float] = []
        epoch_objectness_iou: list[float] = []
        epoch_foreground_rate: list[float] = []
        for batch_start in range(0, len(train_samples), batch_size):
            batch_samples = train_samples[batch_start : batch_start + batch_size]
            sample_index = batch_start + len(batch_samples)
            if model_config.model_family == "rgbd_mask_embedding":
                features_batch = []
                labels_batch = []
                for sample in batch_samples:
                    arrays = build_rgbd_mask_arrays_from_paths(
                        rgb_path=sample["rgb_path"],
                        depth_path=sample["depth_path"],
                        mask_paths=sample["mask_paths"],
                        image_size=model_config.image_size,
                        feature_names=dense_feature_names,
                    )
                    features_batch.append(arrays["features"])
                    labels_batch.append(arrays["labels"])
                x = torch.from_numpy(np.stack(features_batch, axis=0)).to(device=resolved_device, dtype=torch.float32)
                y = torch.from_numpy(np.stack(labels_batch, axis=0)).to(device=resolved_device, dtype=torch.long)
                embeddings, logits = model(x)
                object_loss = dense_objectness_loss(logits, y, torch)
                embed_loss = dense_batch_embedding_loss(embeddings, y, torch)
            else:
                features_batch = []
                labels_batch = []
                for sample in batch_samples:
                    arrays = load_point_cloud_cache(sample["point_cloud_path"])
                    features = arrays["features"]
                    labels = arrays["labels"]
                    if features.size == 0:
                        continue
                    features_batch.append(features)
                    labels_batch.append(labels)
                if not features_batch:
                    continue
                x = torch.from_numpy(np.stack(features_batch, axis=0)).to(device=resolved_device, dtype=torch.float32)
                y = torch.from_numpy(np.stack(labels_batch, axis=0)).to(device=resolved_device, dtype=torch.long)
                embeddings, logits = model(x)
                object_loss = objectness_loss(logits, y, torch)
                embed_loss = point_batch_embedding_loss(embeddings, y, torch)
            loss = object_loss + (embedding_loss_weight * embed_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu().item())
            object_loss_value = float(object_loss.detach().cpu().item())
            embed_loss_value = float(embed_loss.detach().cpu().item())
            batch_metrics = objectness_metrics(logits.detach(), y, torch)
            losses.append(loss_value)
            objectness_losses.append(object_loss_value)
            embedding_losses.append(embed_loss_value)
            epoch_losses.append(loss_value)
            epoch_objectness_losses.append(object_loss_value)
            epoch_embedding_losses.append(embed_loss_value)
            epoch_foreground_accuracy.append(batch_metrics["foreground_accuracy"])
            epoch_background_accuracy.append(batch_metrics["background_accuracy"])
            epoch_objectness_iou.append(batch_metrics["objectness_iou"])
            epoch_foreground_rate.append(batch_metrics["foreground_rate"])
            if progress_interval and (sample_index % progress_interval == 0 or sample_index == len(train_samples)):
                print(
                    "train "
                    f"epoch={epoch_index + 1}/{total_epochs} "
                    f"sample={sample_index}/{len(train_samples)} "
                    f"loss={loss_value:.6f} "
                    f"objectness={object_loss_value:.6f} "
                    f"embedding={embed_loss_value:.6f} "
                    f"fg_acc={batch_metrics['foreground_accuracy']:.3f} "
                    f"bg_acc={batch_metrics['background_accuracy']:.3f} "
                    f"obj_iou={batch_metrics['objectness_iou']:.3f} "
                    f"fg_rate={batch_metrics['foreground_rate']:.3f}",
                    flush=True,
                )
        epoch_summary = {
            "epoch": epoch_index + 1,
            "loss": metric_summary(epoch_losses),
            "objectness_loss": metric_summary(epoch_objectness_losses),
            "embedding_loss": metric_summary(epoch_embedding_losses),
            "foreground_accuracy": mean_metric(epoch_foreground_accuracy),
            "background_accuracy": mean_metric(epoch_background_accuracy),
            "objectness_iou": mean_metric([value for value in epoch_objectness_iou if value is not None]),
            "foreground_rate": mean_metric(epoch_foreground_rate),
        }
        epoch_summaries.append(epoch_summary)
        print(
            "epoch "
            f"{epoch_index + 1}/{total_epochs} "
            f"loss_mean={epoch_summary['loss']['mean']} "
            f"loss_last={epoch_summary['loss']['last']} "
            f"objectness_mean={epoch_summary['objectness_loss']['mean']} "
            f"embedding_mean={epoch_summary['embedding_loss']['mean']} "
            f"fg_acc={epoch_summary['foreground_accuracy']} "
            f"bg_acc={epoch_summary['background_accuracy']} "
            f"obj_iou={epoch_summary['objectness_iou']} "
            f"fg_rate={epoch_summary['foreground_rate']}",
            flush=True,
        )
        metadata["training"]["completed_epochs"] = epoch_index + 1
        save_checkpoint(torch, latest_checkpoint_path, metadata, model)
        print(f"Wrote {latest_checkpoint_path}", flush=True)

    save_checkpoint(torch, checkpoint_path, metadata, model)

    train_summary = {
        "schema_version": 1,
        "status": "trained",
        "trained": True,
        "checkpoint": str(checkpoint_path),
        "manifest_path": str(manifest_path),
        "config_path": str(config_path),
        "architecture": metadata["architecture"],
        "input_channels": list(IMAGE_FEATURE_NAMES if model_config.model_family == "rgbd_mask_embedding" else FEATURE_NAMES),
        "output_contract": EXPECTED_OUTPUT_CONTRACT,
        "epochs": int(epochs),
        "batch": batch_size,
        "device": resolved_device,
        "embedding_loss_weight": embedding_loss_weight,
        "sample_count": len(train_samples),
        "loss": {
            "count": len(losses),
            "mean": round(float(np.mean(losses)), 6) if losses else None,
            "last": round(float(losses[-1]), 6) if losses else None,
        },
        "objectness_loss": metric_summary(objectness_losses),
        "embedding_loss": metric_summary(embedding_losses),
        "epochs_detail": epoch_summaries,
        "split_counts": split_counts(manifest),
    }
    (output / "training_summary.json").write_text(
        json.dumps(train_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    eval_split = "val" if "val" in manifest.get("splits", {}) else "train"
    print(f"Evaluating Primitive3D split={eval_split} device={resolved_device}", flush=True)
    eval_summary = evaluate_checkpoint(
        manifest=manifest,
        model_path=checkpoint_path,
        config=config,
        output_dir=output,
        split=eval_split,
        device=resolved_device,
        summary_name="eval_summary.json",
    )
    eval_data = json.loads(eval_summary.read_text(encoding="utf-8"))
    print(
        "eval "
        f"split={eval_data.get('split')} "
        f"samples={eval_data.get('sample_count')} "
        f"objects={eval_data.get('object_count')} "
        f"predicted={eval_data.get('predicted_object_count')} "
        f"matched={eval_data.get('matched_object_count')} "
        f"recall={eval_data.get('object_recall')} "
        f"mean_iou={eval_data.get('mean_iou')}",
        flush=True,
    )
    del eval_summary
    return checkpoint_path


def write_eval_scaffold(
    *,
    manifest_path: str | Path,
    model_path: str | Path,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_dir: str | Path,
    split: str = "test",
    device: str | None = None,
) -> Path:
    manifest = load_instance_manifest(manifest_path)
    config = load_detector_config(config_path)
    return evaluate_checkpoint(
        manifest=manifest,
        model_path=Path(model_path),
        config=config,
        output_dir=Path(output_dir),
        split=split,
        device=resolve_device(device),
        summary_name="eval_summary.json",
        manifest_path=Path(manifest_path),
        config_path=Path(config_path),
    )


def save_checkpoint(torch, path: Path, metadata: dict[str, Any], model) -> None:
    torch.save(
        {
            "schema_version": 1,
            "metadata": metadata,
            "model_state": model.state_dict(),
        },
        path,
    )


def evaluate_checkpoint(
    *,
    manifest: dict[str, Any],
    model_path: Path,
    config: dict[str, Any],
    output_dir: Path,
    split: str,
    device: str,
    summary_name: str,
    manifest_path: Path | None = None,
    config_path: Path | None = None,
) -> Path:
    torch = import_torch()
    if split not in manifest.get("splits", {}):
        raise ValueError(f"Split {split!r} does not exist in manifest.")
    output_dir.mkdir(parents=True, exist_ok=True)
    model, metadata = load_checkpoint(model_path, device=device)
    model_config = primitive_config_from_config(config)
    checkpoint_family = str(metadata.get("model", {}).get("model_family", model_config.model_family))
    dense_feature_names = tuple(metadata.get("input_channels", config.get("input_channels", IMAGE_FEATURE_NAMES)))
    samples = (
        prepared_dense_split_samples(manifest, split)
        if checkpoint_family == "rgbd_mask_embedding"
        else prepared_split_samples(manifest, split, model_config)
    )
    metrics: list[dict[str, float | int | str]] = []
    model.eval()
    preview_limit = int(config.get("eval_preview_count", 0)) if checkpoint_family == "rgbd_mask_embedding" else 0
    preview_dir = output_dir / "eval_previews"
    with torch.no_grad():
        for sample_index, sample in enumerate(samples):
            if checkpoint_family == "rgbd_mask_embedding":
                arrays = build_rgbd_mask_arrays_from_paths(
                    rgb_path=sample["rgb_path"],
                    depth_path=sample["depth_path"],
                    mask_paths=sample["mask_paths"],
                    image_size=int(metadata.get("model", {}).get("image_size", model_config.image_size)),
                    feature_names=dense_feature_names,
                )
                features = arrays["features"]
                labels = arrays["labels"]
                width, height = [int(value) for value in arrays["image_size"]]
                x = torch.from_numpy(features).to(device=device, dtype=torch.float32).unsqueeze(0)
                embeddings, logits = model(x)
                objectness_np = torch.sigmoid(logits.squeeze(0)).detach().cpu().numpy()
                detections = segment_detections_from_maps(
                    embeddings=embeddings.squeeze(0).detach().cpu().numpy(),
                    objectness=objectness_np,
                    objectness_threshold=float(metadata.get("model", {}).get("objectness_threshold", config.get("objectness_threshold", 0.5))),
                    embedding_distance=float(metadata.get("model", {}).get("embedding_distance", config.get("embedding_distance", 0.35))),
                    min_cluster_points=int(metadata.get("model", {}).get("min_cluster_points", config.get("min_cluster_points", 32))),
                    max_instances=32,
                )
                rows, cols = np.nonzero(np.ones(labels.shape, dtype=bool))
                metrics.append(sample_metrics(sample["id"], labels.reshape((-1,)), rows.astype(np.int32), cols.astype(np.int32), (width, height), detections))
                if sample_index < preview_limit:
                    write_dense_eval_preview(
                        output_dir=preview_dir / sample["id"],
                        rgb_path=sample["rgb_path"],
                        labels=labels,
                        objectness=objectness_np,
                        detections=detections,
                    )
            else:
                arrays = load_point_cloud_cache(sample["point_cloud_path"])
                features = arrays["features"]
                labels = arrays["labels"]
                width, height = [int(value) for value in arrays["image_size"]]
                if features.size == 0 or width <= 0 or height <= 0:
                    metrics.append(empty_sample_metrics(sample["id"]))
                    continue
                x = torch.from_numpy(features).to(device=device, dtype=torch.float32).unsqueeze(0)
                embeddings, logits = model(x)
                detections = segment_detections_from_arrays(
                    embeddings=embeddings.squeeze(0).detach().cpu().numpy(),
                    objectness=torch.sigmoid(logits.squeeze(0)).detach().cpu().numpy(),
                    rows=arrays["rows"],
                    cols=arrays["cols"],
                    image_size=(width, height),
                    objectness_threshold=float(metadata.get("model", {}).get("objectness_threshold", config.get("objectness_threshold", 0.5))),
                    embedding_distance=float(metadata.get("model", {}).get("embedding_distance", config.get("embedding_distance", 0.35))),
                    min_cluster_points=int(metadata.get("model", {}).get("min_cluster_points", config.get("min_cluster_points", 32))),
                    max_instances=32,
                )
                metrics.append(sample_metrics(sample["id"], labels, arrays["rows"], arrays["cols"], (width, height), detections))

    summary = summarize_metrics(metrics)
    summary.update(
        {
            "schema_version": 1,
            "status": "evaluated",
            "evaluated": True,
            "model_path": str(model_path),
            "manifest_path": str(manifest_path) if manifest_path is not None else str(metadata.get("training", {}).get("manifest_path")),
            "config_path": str(config_path) if config_path is not None else str(metadata.get("training", {}).get("config_path")),
            "config_name": config.get("name"),
            "architecture": metadata.get("architecture", config.get("architecture", DEFAULT_ARCHITECTURE)),
            "split": split,
            "device": device,
            "input_channels": list(
                IMAGE_FEATURE_NAMES
                if checkpoint_family == "rgbd_mask_embedding"
                else FEATURE_NAMES
            ),
            "output_contract": EXPECTED_OUTPUT_CONTRACT,
            "samples": metrics,
        }
    )
    summary_path = output_dir / summary_name
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary_path


def prepared_split_samples(manifest: dict[str, Any], split: str, config: Primitive3DConfig) -> list[dict[str, Any]]:
    dataset_root = Path(manifest["dataset_root"])
    samples = []
    for sample in manifest.get("splits", {}).get(split, {}).get("samples", []):
        if not sample.get("depth"):
            raise ValueError(f"Sample {sample.get('id', '<unknown>')} is missing required depth path.")
        rgb = dataset_root / sample["rgb"]
        depth = dataset_root / sample["depth"]
        masks = [dataset_root / obj["visible_mask"] for obj in sample.get("objects", [])]
        point_cloud_path = dataset_root / sample.get("point_cloud", f"{split}/point_cloud/{sample['id']}.npz")
        if not point_cloud_path.is_file():
            build_cache_from_paths(
                rgb_path=rgb,
                depth_path=depth,
                mask_paths=masks,
                cache_path=point_cloud_path,
                max_points=config.max_points,
                fov_degrees=config.fov_degrees,
                near_depth=config.near_depth,
                far_depth=config.far_depth,
            )
        samples.append({"id": sample["id"], "point_cloud_path": point_cloud_path})
    return samples


def write_dense_eval_preview(
    *,
    output_dir: Path,
    rgb_path: Path,
    labels: np.ndarray,
    objectness: np.ndarray,
    detections,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    height, width = labels.shape
    rgb = Image.open(rgb_path).convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
    rgb.save(output_dir / "rgb.png")
    objectness_image = Image.fromarray(np.clip(objectness * 255.0, 0, 255).astype(np.uint8), mode="L")
    objectness_image.save(output_dir / "objectness.png")
    colorize_labels(labels).save(output_dir / "gt_instances.png")
    pred = rgb.copy().convert("RGBA")
    overlay = Image.new("RGBA", pred.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    for index, detection in enumerate(detections, start=1):
        color = preview_color(index)
        if len(detection.mask_polygon) >= 3:
            draw.polygon(detection.mask_polygon, fill=(*color, 80), outline=(*color, 255))
            points = detection.mask_polygon + [detection.mask_polygon[0]]
            draw.line(points, fill=(*color, 255), width=2)
        draw.text((int(detection.bbox_xyxy[0]), int(detection.bbox_xyxy[1])), str(index), fill=(255, 255, 255, 255))
    Image.alpha_composite(pred, overlay).convert("RGB").save(output_dir / "pred_instances.png")


def colorize_labels(labels: np.ndarray) -> Image.Image:
    output = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for label in sorted(int(value) for value in np.unique(labels) if int(value) > 0):
        output[labels == label] = preview_color(label)
    return Image.fromarray(output, mode="RGB")


def preview_color(index: int) -> tuple[int, int, int]:
    palette = (
        (230, 57, 70),
        (42, 157, 143),
        (69, 123, 157),
        (244, 162, 97),
        (131, 56, 236),
        (255, 183, 3),
        (58, 134, 255),
        (6, 214, 160),
    )
    return palette[(int(index) - 1) % len(palette)]


def prepared_dense_split_samples(manifest: dict[str, Any], split: str) -> list[dict[str, Any]]:
    dataset_root = Path(manifest["dataset_root"])
    samples = []
    for sample in manifest.get("splits", {}).get(split, {}).get("samples", []):
        if not sample.get("depth"):
            raise ValueError(f"Sample {sample.get('id', '<unknown>')} is missing required depth path.")
        samples.append(
            {
                "id": sample["id"],
                "rgb_path": dataset_root / sample["rgb"],
                "depth_path": dataset_root / sample["depth"],
                "mask_paths": [dataset_root / obj["visible_mask"] for obj in sample.get("objects", [])],
            }
        )
    return samples


def primitive_config_from_config(config: dict[str, Any]) -> Primitive3DConfig:
    return Primitive3DConfig(
        input_dim=len(FEATURE_NAMES),
        image_input_dim=int(config.get("image_input_dim", 7)),
        hidden_dim=int(config.get("hidden_dim", 64)),
        embedding_dim=int(config.get("embedding_dim", 8)),
        encoder_layers=int(config.get("encoder_layers", 3)),
        decoder_layers=int(config.get("decoder_layers", 0)),
        use_global_context=bool(config.get("use_global_context", False)),
        model_family=str(config.get("model_family", "pointnet_embedding")),
        image_size=int(config.get("image_size", 256)),
        base_channels=int(config.get("base_channels", 32)),
        max_points=int(config.get("max_points", 4096)),
        min_cluster_points=int(config.get("min_cluster_points", 32)),
        embedding_distance=float(config.get("embedding_distance", 0.35)),
        fov_degrees=float(config.get("fov_degrees", 70.0)),
        near_depth=float(config.get("near_depth", 1.0)),
        far_depth=float(config.get("far_depth", 8.0)),
    )


def objectness_loss(logits, labels, torch):
    target = (labels > 0).float()
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, target)


def dense_objectness_loss(logits, labels, torch):
    target = (labels > 0).float()
    foreground_count = torch.clamp(target.sum(), min=1.0)
    background_count = torch.clamp(target.numel() - target.sum(), min=1.0)
    pos_weight = torch.clamp(background_count / foreground_count, min=1.0, max=20.0).to(device=logits.device)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    probs = torch.sigmoid(logits)
    intersection = (probs * target).sum()
    dice = 1.0 - ((2.0 * intersection + 1.0) / (probs.sum() + target.sum() + 1.0))
    return bce + dice


def dense_embedding_loss(embeddings, labels, torch):
    embedding_dim = int(embeddings.shape[0])
    normalized = torch.nn.functional.normalize(
        embeddings.permute(1, 2, 0).reshape((-1, embedding_dim)),
        p=2,
        dim=1,
    )
    flat_labels = labels.reshape((-1,))
    object_labels = torch.unique(flat_labels[flat_labels > 0])
    if int(object_labels.numel()) == 0:
        return normalized.sum() * 0.0
    centers = []
    pull_losses = []
    for label in object_labels:
        points = normalized[flat_labels == label]
        center = torch.nn.functional.normalize(points.mean(dim=0, keepdim=True), p=2, dim=1).squeeze(0)
        centers.append(center)
        distances = torch.norm(points - center, dim=1)
        pull_losses.append(torch.mean(torch.clamp(distances - 0.18, min=0.0) ** 2))
    pull = torch.stack(pull_losses).mean()
    if len(centers) < 2:
        return pull
    push_losses = []
    for left_index in range(len(centers)):
        for right_index in range(left_index + 1, len(centers)):
            distance = torch.norm(centers[left_index] - centers[right_index])
            push_losses.append(torch.clamp(1.2 - distance, min=0.0) ** 2)
    push = torch.stack(push_losses).mean() if push_losses else pull * 0.0
    return pull + push


def dense_batch_embedding_loss(embeddings, labels, torch):
    losses = [
        dense_embedding_loss(embeddings[index], labels[index], torch)
        for index in range(int(embeddings.shape[0]))
    ]
    return torch.stack(losses).mean() if losses else embeddings.sum() * 0.0


def embedding_loss(embeddings, labels, torch):
    object_labels = torch.unique(labels[labels > 0])
    if int(object_labels.numel()) == 0:
        return embeddings.sum() * 0.0
    centers = []
    pull_losses = []
    for label in object_labels:
        points = embeddings[labels == label]
        center = points.mean(dim=0)
        centers.append(center)
        distances = torch.norm(points - center, dim=1)
        pull_losses.append(torch.mean(torch.clamp(distances - 0.10, min=0.0) ** 2))
    pull = torch.stack(pull_losses).mean()
    if len(centers) < 2:
        return pull
    push_losses = []
    for left_index in range(len(centers)):
        for right_index in range(left_index + 1, len(centers)):
            distance = torch.norm(centers[left_index] - centers[right_index])
            push_losses.append(torch.clamp(1.0 - distance, min=0.0) ** 2)
    push = torch.stack(push_losses).mean() if push_losses else pull * 0.0
    return pull + push


def point_batch_embedding_loss(embeddings, labels, torch):
    losses = [
        embedding_loss(embeddings[index], labels[index], torch)
        for index in range(int(embeddings.shape[0]))
    ]
    return torch.stack(losses).mean() if losses else embeddings.sum() * 0.0


def objectness_metrics(logits, labels, torch) -> dict[str, float]:
    scores = torch.sigmoid(logits)
    foreground = labels > 0
    background = ~foreground
    predictions = scores >= 0.5
    foreground_count = int(foreground.sum().detach().cpu().item())
    background_count = int(background.sum().detach().cpu().item())
    foreground_accuracy = (
        float((predictions[foreground] == foreground[foreground]).float().mean().detach().cpu().item())
        if foreground_count
        else 0.0
    )
    background_accuracy = (
        float((predictions[background] == foreground[background]).float().mean().detach().cpu().item())
        if background_count
        else 0.0
    )
    intersection = int((predictions & foreground).sum().detach().cpu().item())
    union = int((predictions | foreground).sum().detach().cpu().item())
    objectness_iou = intersection / max(1, union)
    foreground_rate = foreground_count / max(1, foreground_count + background_count)
    return {
        "foreground_accuracy": foreground_accuracy,
        "background_accuracy": background_accuracy,
        "objectness_iou": objectness_iou,
        "foreground_rate": foreground_rate,
    }


def mean_metric(values: list[float]) -> float | None:
    return round(float(np.mean(values)), 6) if values else None


def metric_summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": mean_metric(values),
        "last": round(float(values[-1]), 6) if values else None,
    }


def sample_metrics(
    sample_id: str,
    labels: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    image_size: tuple[int, int],
    detections,
) -> dict[str, float | int | str]:
    width, height = image_size
    gt_masks = []
    for label in sorted(int(value) for value in np.unique(labels) if int(value) > 0):
        mask = np.zeros((height, width), dtype=bool)
        member = labels == label
        mask[rows[member], cols[member]] = True
        gt_masks.append(mask)
    pred_masks = [polygon_to_mask(det.mask_polygon, width, height) for det in detections]
    matches = greedy_mask_matches(gt_masks, pred_masks)
    gt_count = len(gt_masks)
    pred_count = len(pred_masks)
    recall = len(matches) / max(1, gt_count)
    false_positive_count = max(0, pred_count - len(matches))
    return {
        "sample_id": sample_id,
        "gt_objects": gt_count,
        "pred_objects": pred_count,
        "matches": len(matches),
        "object_recall": round(float(recall), 6),
        "duplicate_split_rate": round(float(max(0, pred_count - gt_count) / max(1, gt_count)), 6),
        "background_false_positive_rate": round(float(false_positive_count / max(1, pred_count)), 6),
        "mean_iou": round(float(np.mean([score for *_indices, score in matches])), 6) if matches else 0.0,
    }


def greedy_mask_matches(gt_masks: list[np.ndarray], pred_masks: list[np.ndarray]) -> list[tuple[int, int, float]]:
    candidates: list[tuple[int, int, float]] = []
    for gt_index, gt_mask in enumerate(gt_masks):
        for pred_index, pred_mask in enumerate(pred_masks):
            score = mask_iou(gt_mask, pred_mask)
            if score >= 0.50:
                candidates.append((gt_index, pred_index, score))
    matches = []
    used_gt = set()
    used_pred = set()
    for gt_index, pred_index, score in sorted(candidates, key=lambda item: item[2], reverse=True):
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        matches.append((gt_index, pred_index, score))
    return matches


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int((left & right).sum())
    union = int((left | right).sum())
    return intersection / max(1.0, float(union))


def summarize_metrics(metrics: list[dict[str, float | int | str]]) -> dict[str, Any]:
    total_gt = sum(int(item["gt_objects"]) for item in metrics)
    total_pred = sum(int(item["pred_objects"]) for item in metrics)
    total_matches = sum(int(item["matches"]) for item in metrics)
    return {
        "sample_count": len(metrics),
        "object_count": total_gt,
        "predicted_object_count": total_pred,
        "matched_object_count": total_matches,
        "object_recall": round(total_matches / max(1, total_gt), 6),
        "duplicate_split_rate": round(max(0, total_pred - total_gt) / max(1, total_gt), 6),
        "background_false_positive_rate": round(max(0, total_pred - total_matches) / max(1, total_pred), 6),
        "mean_iou": round(float(np.mean([float(item["mean_iou"]) for item in metrics])), 6) if metrics else 0.0,
    }


def empty_sample_metrics(sample_id: str) -> dict[str, float | int | str]:
    return {
        "sample_id": sample_id,
        "gt_objects": 0,
        "pred_objects": 0,
        "matches": 0,
        "object_recall": 0.0,
        "duplicate_split_rate": 0.0,
        "background_false_positive_rate": 0.0,
        "mean_iou": 0.0,
    }


def split_counts(manifest: dict[str, Any]) -> dict[str, dict[str, int]]:
    return {
        split: {
            "samples": int(split_data.get("sample_count", 0)),
            "objects": int(split_data.get("object_count", 0)),
        }
        for split, split_data in manifest.get("splits", {}).items()
    }


def resolve_device(device: str | None) -> str:
    torch = import_torch()
    value = device or "cpu"
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value.isdigit():
        return f"cuda:{value}"
    return value


def import_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for primitive 3D instance detector training/eval.") from exc
    return torch
