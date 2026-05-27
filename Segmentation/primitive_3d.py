from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from Input.Depth.depth_loader import load_grayscale_depth
from Segmentation.depth_edge_segmenter import component_bbox, component_polygon, connected_components, mask_iou
from Segmentation.types import SegmentDetection

try:
    import torch as _torch
    import torch.nn as _nn
except ModuleNotFoundError:
    _torch = None
    _nn = None


FEATURE_NAMES = (
    "x",
    "y",
    "z",
    "r",
    "g",
    "b",
    "depth",
    "u",
    "v",
    "edge_strength",
)
IMAGE_FEATURE_NAMES = (
    "r",
    "g",
    "b",
    "depth",
    "rgb_edge_strength",
    "depth_edge_strength",
    "depth_gradient_x",
    "depth_gradient_y",
    "u",
    "v",
)
DEFAULT_MAX_POINTS = 4096
DEFAULT_MIN_CLUSTER_POINTS = 32
DEFAULT_EMBEDDING_DISTANCE = 0.35


@dataclass(frozen=True)
class Primitive3DConfig:
    input_dim: int = len(FEATURE_NAMES)
    image_input_dim: int = len(IMAGE_FEATURE_NAMES)
    hidden_dim: int = 64
    embedding_dim: int = 8
    encoder_layers: int = 3
    decoder_layers: int = 0
    use_global_context: bool = False
    model_family: str = "pointnet_embedding"
    image_size: int = 256
    base_channels: int = 32
    max_points: int = DEFAULT_MAX_POINTS
    min_cluster_points: int = DEFAULT_MIN_CLUSTER_POINTS
    embedding_distance: float = DEFAULT_EMBEDDING_DISTANCE
    fov_degrees: float = 70.0
    near_depth: float = 1.0
    far_depth: float = 8.0


def torch_module():
    if _torch is None or _nn is None:
        raise RuntimeError("PyTorch is required for primitive-3d-segmenter.")
    return _torch, _nn


class ConvNormAct((_nn.Module if _nn is not None else object)):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        _, nn = torch_module()
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualConvBlock((_nn.Module if _nn is not None else object)):
    def __init__(self, channels: int) -> None:
        _, nn = torch_module()
        super().__init__()
        self.left = ConvNormAct(channels, channels)
        self.right = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.right(self.left(x))) + x)


class RGBDMaskSegNet((_nn.Module if _nn is not None else object)):
    """Small YOLO-style 2.5D mask embedding model.

    The network consumes dense RGB, depth, edge, and normalized coordinate
    channels, then predicts class-agnostic objectness plus per-pixel embeddings.
    """

    def __init__(self, input_dim: int = len(IMAGE_FEATURE_NAMES), base_channels: int = 32, embedding_dim: int = 16) -> None:
        _, nn = torch_module()
        super().__init__()
        base = int(base_channels)
        self.input_dim = int(input_dim)
        self.base_channels = base
        self.embedding_dim = int(embedding_dim)
        self.stem = ConvNormAct(self.input_dim, base)
        self.down1 = nn.Sequential(ConvNormAct(base, base * 2, stride=2), ResidualConvBlock(base * 2))
        self.down2 = nn.Sequential(ConvNormAct(base * 2, base * 4, stride=2), ResidualConvBlock(base * 4))
        self.down3 = nn.Sequential(ConvNormAct(base * 4, base * 8, stride=2), ResidualConvBlock(base * 8))
        self.fuse2 = ConvNormAct(base * 8 + base * 4, base * 4)
        self.fuse1 = ConvNormAct(base * 4 + base * 2, base * 2)
        self.fuse0 = ConvNormAct(base * 2 + base, base)
        self.head = nn.Sequential(ResidualConvBlock(base), ConvNormAct(base, base))
        self.embedding_head = nn.Conv2d(base, self.embedding_dim, kernel_size=1)
        self.objectness_head = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, image_features):
        torch, _ = torch_module()
        x0 = self.stem(image_features)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        up2 = torch.nn.functional.interpolate(x3, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        up2 = self.fuse2(torch.cat((up2, x2), dim=1))
        up1 = torch.nn.functional.interpolate(up2, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        up1 = self.fuse1(torch.cat((up1, x1), dim=1))
        up0 = torch.nn.functional.interpolate(up1, size=x0.shape[-2:], mode="bilinear", align_corners=False)
        features = self.fuse0(torch.cat((up0, x0), dim=1))
        features = self.head(features)
        embeddings = self.embedding_head(features)
        objectness_logits = self.objectness_head(features).squeeze(1)
        return embeddings, objectness_logits


class Primitive3DSegNet((_nn.Module if _nn is not None else object)):
    def __init__(
        self,
        input_dim: int = len(FEATURE_NAMES),
        hidden_dim: int = 64,
        embedding_dim: int = 8,
        encoder_layers: int = 3,
        decoder_layers: int = 0,
        use_global_context: bool = False,
    ) -> None:
        _, nn = torch_module()
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.embedding_dim = int(embedding_dim)
        self.encoder_layers = max(1, int(encoder_layers))
        self.decoder_layers = max(0, int(decoder_layers))
        self.use_global_context = bool(use_global_context)
        self.encoder = mlp(
            nn=nn,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            layer_count=self.encoder_layers,
        )
        decoder_input_dim = self.hidden_dim * 3 if self.use_global_context else self.hidden_dim
        self.decoder = (
            mlp(
                nn=nn,
                input_dim=decoder_input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.hidden_dim,
                layer_count=self.decoder_layers,
            )
            if self.decoder_layers
            else None
        )
        head_input_dim = self.hidden_dim if self.decoder_layers else decoder_input_dim
        self.embedding_head = nn.Linear(head_input_dim, self.embedding_dim)
        self.objectness_head = nn.Linear(head_input_dim, 1)

    def forward(self, points):
        features = self.encoder(points)
        if self.use_global_context:
            max_features = features.max(dim=1, keepdim=True).values.expand_as(features)
            mean_features = features.mean(dim=1, keepdim=True).expand_as(features)
            features = _torch.cat((features, max_features, mean_features), dim=-1)
        if self.decoder is not None:
            features = self.decoder(features)
        embeddings = self.embedding_head(features)
        objectness_logits = self.objectness_head(features).squeeze(-1)
        return embeddings, objectness_logits


def mlp(*, nn, input_dim: int, hidden_dim: int, output_dim: int, layer_count: int):
    layers = []
    current_dim = int(input_dim)
    for layer_index in range(max(1, int(layer_count))):
        next_dim = int(output_dim) if layer_index == int(layer_count) - 1 else int(hidden_dim)
        layers.append(nn.Linear(current_dim, next_dim))
        layers.append(nn.ReLU())
        current_dim = next_dim
    return nn.Sequential(*layers)


def default_checkpoint_metadata(config: Primitive3DConfig) -> dict:
    return {
        "schema_version": 1,
        "detector_backend": "primitive-3d-segmenter",
        "architecture": "primitive_3d_point_embedding_v1",
        "input_contract": "rgb_depth_camera_to_visible_point_cloud",
        "input_channels": list(IMAGE_FEATURE_NAMES if config.model_family == "rgbd_mask_embedding" else FEATURE_NAMES),
        "output_contract": "class_agnostic_instance_masks",
        "primitive_label_policy": "geometry_fitting_downstream",
        "model": {
            "input_dim": int(config.input_dim),
            "image_input_dim": int(config.image_input_dim),
            "hidden_dim": int(config.hidden_dim),
            "embedding_dim": int(config.embedding_dim),
            "encoder_layers": int(config.encoder_layers),
            "decoder_layers": int(config.decoder_layers),
            "use_global_context": bool(config.use_global_context),
            "model_family": str(config.model_family),
            "image_size": int(config.image_size),
            "base_channels": int(config.base_channels),
            "max_points": int(config.max_points),
            "min_cluster_points": int(config.min_cluster_points),
            "embedding_distance": float(config.embedding_distance),
            "fov_degrees": float(config.fov_degrees),
            "near_depth": float(config.near_depth),
            "far_depth": float(config.far_depth),
        },
    }


def load_checkpoint(path: str | Path, device: str | None = None) -> tuple[Primitive3DSegNet, dict]:
    torch, _ = torch_module()
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Primitive3D checkpoint does not exist: {checkpoint_path}")
    resolved_device = device or "cpu"
    try:
        data = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    except TypeError:
        data = torch.load(checkpoint_path, map_location=resolved_device)
    metadata = data.get("metadata", {})
    model_config = metadata.get("model", {})
    model_family = str(model_config.get("model_family", "pointnet_embedding"))
    if model_family == "rgbd_mask_embedding":
        model = RGBDMaskSegNet(
            input_dim=int(model_config.get("image_input_dim", len(IMAGE_FEATURE_NAMES))),
            base_channels=int(model_config.get("base_channels", 32)),
            embedding_dim=int(model_config.get("embedding_dim", 16)),
        )
    else:
        model = Primitive3DSegNet(
            input_dim=int(model_config.get("input_dim", len(FEATURE_NAMES))),
            hidden_dim=int(model_config.get("hidden_dim", 64)),
            embedding_dim=int(model_config.get("embedding_dim", 8)),
            encoder_layers=int(model_config.get("encoder_layers", 3)),
            decoder_layers=int(model_config.get("decoder_layers", 0)),
            use_global_context=bool(model_config.get("use_global_context", False)),
        )
    model.load_state_dict(data["model_state"])
    model.to(resolved_device)
    model.eval()
    return model, metadata


def build_point_cloud_arrays(
    *,
    image: Image.Image,
    depth: np.ndarray,
    masks: list[np.ndarray] | None = None,
    max_points: int = DEFAULT_MAX_POINTS,
    fov_degrees: float = 70.0,
    near_depth: float = 1.0,
    far_depth: float = 8.0,
) -> dict[str, np.ndarray]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    edge = np.asarray(image.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
    height, width = depth.shape
    valid = depth > 0.0
    rows, cols = np.nonzero(valid)
    if rows.size == 0:
        return empty_point_cloud()

    order = deterministic_point_indices(rows.size, max_points)
    rows = rows[order]
    cols = cols[order]
    normalized_depth = depth[rows, cols].astype(np.float32)
    scene_depth = near_depth + (1.0 - normalized_depth) * (far_depth - near_depth)
    fov = math.radians(float(fov_degrees))
    focal = (float(width) * 0.5) / max(1e-6, math.tan(fov * 0.5))
    x = ((cols.astype(np.float32) + 0.5) - width * 0.5) * scene_depth / focal
    z = (height * 0.5 - (rows.astype(np.float32) + 0.5)) * scene_depth / focal
    u = cols.astype(np.float32) / max(1.0, float(width - 1))
    v = rows.astype(np.float32) / max(1.0, float(height - 1))

    features = np.column_stack(
        (
            x,
            scene_depth,
            z,
            rgb[rows, cols, 0],
            rgb[rows, cols, 1],
            rgb[rows, cols, 2],
            normalized_depth,
            u,
            v,
            edge[rows, cols],
        )
    ).astype(np.float32)
    labels = np.zeros(rows.shape[0], dtype=np.int64)
    if masks:
        for object_index, mask in enumerate(masks, start=1):
            if mask.shape == depth.shape:
                labels[mask[rows, cols]] = object_index
    return {
        "features": features,
        "labels": labels,
        "rows": rows.astype(np.int32),
        "cols": cols.astype(np.int32),
        "image_size": np.asarray([width, height], dtype=np.int32),
    }


def build_rgbd_mask_arrays(
    *,
    image: Image.Image,
    depth: np.ndarray,
    masks: list[np.ndarray] | None = None,
    image_size: int = 256,
    feature_names: tuple[str, ...] = IMAGE_FEATURE_NAMES,
) -> dict[str, np.ndarray]:
    target_size = (int(image_size), int(image_size))
    rgb_image = image.convert("RGB").resize(target_size, Image.Resampling.BILINEAR)
    depth_image = Image.fromarray(np.clip(depth * 255.0, 0, 255).astype(np.uint8), mode="L").resize(
        target_size,
        Image.Resampling.BILINEAR,
    )
    edge_image = rgb_image.convert("L").filter(ImageFilter.FIND_EDGES)
    rgb = np.asarray(rgb_image, dtype=np.float32) / 255.0
    depth_values = np.asarray(depth_image, dtype=np.float32) / 255.0
    rgb_edge = np.asarray(edge_image, dtype=np.float32) / 255.0
    depth_gradient_y, depth_gradient_x = np.gradient(depth_values)
    depth_edge = np.sqrt(depth_gradient_x**2 + depth_gradient_y**2)
    depth_edge = np.clip(depth_edge * 8.0, 0.0, 1.0).astype(np.float32)
    depth_gradient_x = np.clip(depth_gradient_x * 8.0, -1.0, 1.0).astype(np.float32)
    depth_gradient_y = np.clip(depth_gradient_y * 8.0, -1.0, 1.0).astype(np.float32)
    height, width = depth_values.shape
    u = np.tile(np.linspace(0.0, 1.0, width, dtype=np.float32), (height, 1))
    v = np.tile(np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None], (1, width))
    channel_map = {
        "r": rgb[:, :, 0],
        "g": rgb[:, :, 1],
        "b": rgb[:, :, 2],
        "depth": depth_values,
        "edge_strength": rgb_edge,
        "rgb_edge_strength": rgb_edge,
        "depth_edge_strength": depth_edge,
        "depth_gradient_x": depth_gradient_x,
        "depth_gradient_y": depth_gradient_y,
        "u": u,
        "v": v,
    }
    features = np.stack(tuple(channel_map[name] for name in feature_names), axis=0).astype(np.float32)
    labels = np.zeros((height, width), dtype=np.int64)
    if masks:
        for object_index, mask in enumerate(masks, start=1):
            if mask.shape != depth.shape:
                continue
            mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L").resize(target_size, Image.Resampling.NEAREST)
            labels[np.asarray(mask_image, dtype=np.uint8) > 127] = object_index
    return {
        "features": features,
        "labels": labels,
        "image_size": np.asarray([width, height], dtype=np.int32),
        "source_image_size": np.asarray([image.width, image.height], dtype=np.int32),
    }


def empty_point_cloud() -> dict[str, np.ndarray]:
    return {
        "features": np.empty((0, len(FEATURE_NAMES)), dtype=np.float32),
        "labels": np.empty((0,), dtype=np.int64),
        "rows": np.empty((0,), dtype=np.int32),
        "cols": np.empty((0,), dtype=np.int32),
        "image_size": np.asarray([0, 0], dtype=np.int32),
    }


def deterministic_point_indices(count: int, max_points: int) -> np.ndarray:
    if count <= max_points:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, int(max_points), dtype=np.int64)


def save_point_cloud_cache(path: str | Path, arrays: dict[str, np.ndarray]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    return output_path


def load_point_cloud_cache(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path)) as data:
        return {key: data[key] for key in data.files}


def masks_from_paths(mask_paths: list[str | Path], image_size: tuple[int, int]) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for path in mask_paths:
        mask_image = Image.open(path).convert("L").resize(image_size)
        masks.append(np.asarray(mask_image, dtype=np.uint8) > 127)
    return masks


def build_cache_from_paths(
    *,
    rgb_path: str | Path,
    depth_path: str | Path,
    mask_paths: list[str | Path],
    cache_path: str | Path,
    max_points: int = DEFAULT_MAX_POINTS,
    fov_degrees: float = 70.0,
    near_depth: float = 1.0,
    far_depth: float = 8.0,
) -> Path:
    image = Image.open(rgb_path).convert("RGB")
    depth = load_grayscale_depth(depth_path, expected_size=image.size)
    masks = masks_from_paths(mask_paths, image.size)
    arrays = build_point_cloud_arrays(
        image=image,
        depth=depth,
        masks=masks,
        max_points=max_points,
        fov_degrees=fov_degrees,
        near_depth=near_depth,
        far_depth=far_depth,
    )
    return save_point_cloud_cache(cache_path, arrays)


def build_rgbd_mask_arrays_from_paths(
    *,
    rgb_path: str | Path,
    depth_path: str | Path,
    mask_paths: list[str | Path],
    image_size: int = 256,
    feature_names: tuple[str, ...] = IMAGE_FEATURE_NAMES,
) -> dict[str, np.ndarray]:
    image = Image.open(rgb_path).convert("RGB")
    depth = load_grayscale_depth(depth_path, expected_size=image.size)
    masks = masks_from_paths(mask_paths, image.size)
    return build_rgbd_mask_arrays(image=image, depth=depth, masks=masks, image_size=image_size, feature_names=feature_names)


def segment_detections_from_arrays(
    *,
    embeddings: np.ndarray,
    objectness: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    image_size: tuple[int, int],
    objectness_threshold: float = 0.5,
    embedding_distance: float = DEFAULT_EMBEDDING_DISTANCE,
    min_cluster_points: int = DEFAULT_MIN_CLUSTER_POINTS,
    max_instances: int = 32,
) -> list[SegmentDetection]:
    width, height = image_size
    if embeddings.size == 0 or objectness.size == 0:
        return []
    foreground = objectness >= float(objectness_threshold)
    if int(foreground.sum()) < min_cluster_points:
        return []
    fg_embeddings = embeddings[foreground]
    fg_rows = rows[foreground]
    fg_cols = cols[foreground]
    fg_scores = objectness[foreground]
    clusters = greedy_embedding_clusters(
        fg_embeddings,
        distance=float(embedding_distance),
        min_cluster_points=int(min_cluster_points),
        max_instances=int(max_instances),
    )
    detections: list[SegmentDetection] = []
    for cluster in clusters:
        mask = np.zeros((height, width), dtype=bool)
        mask[fg_rows[cluster], fg_cols[cluster]] = True
        if int(mask.sum()) < min_cluster_points:
            continue
        bbox = component_bbox(mask)
        polygon = component_polygon(mask, bbox, max_points=96)
        if len(polygon) < 3:
            continue
        confidence = float(np.mean(fg_scores[cluster]))
        detections.append(
            SegmentDetection(
                bbox_xyxy=bbox,
                mask_polygon=polygon,
                detector_label="unknown",
                detector_confidence=max(0.0, min(1.0, confidence)),
            )
        )
    detections.sort(key=lambda item: (item.bbox_xyxy[1], item.bbox_xyxy[0]))
    return detections


def segment_detections_from_maps(
    *,
    embeddings: np.ndarray,
    objectness: np.ndarray,
    objectness_threshold: float = 0.5,
    embedding_distance: float = DEFAULT_EMBEDDING_DISTANCE,
    min_cluster_points: int = DEFAULT_MIN_CLUSTER_POINTS,
    max_instances: int = 32,
    output_size: tuple[int, int] | None = None,
) -> list[SegmentDetection]:
    height, width = objectness.shape
    foreground = objectness >= float(objectness_threshold)
    if int(foreground.sum()) < min_cluster_points:
        return []
    embedding_map = np.moveaxis(embeddings, 0, -1)
    norms = np.linalg.norm(embedding_map, axis=2, keepdims=True)
    embedding_map = embedding_map / np.maximum(norms, 1e-6)
    components = connected_components(
        foreground,
        min_area=int(min_cluster_points),
        max_components=max(1, int(max_instances) * 4),
    )
    detections_with_masks: list[tuple[SegmentDetection, np.ndarray]] = []
    for component in components:
        component_rows, component_cols = np.nonzero(component)
        component_embeddings = embedding_map[component_rows, component_cols]
        component_scores = objectness[component_rows, component_cols]
        clusters = greedy_embedding_clusters(
            component_embeddings,
            distance=float(embedding_distance),
            min_cluster_points=int(min_cluster_points),
            max_instances=max(1, int(max_instances) - len(detections_with_masks)),
        )
        for cluster in clusters:
            mask = np.zeros((height, width), dtype=bool)
            mask[component_rows[cluster], component_cols[cluster]] = True
            detection = detection_from_mask(
                mask=mask,
                confidence=float(np.mean(component_scores[cluster])),
                min_cluster_points=int(min_cluster_points),
            )
            if detection is not None:
                detections_with_masks.append((detection, mask))
        if len(detections_with_masks) >= max_instances:
            break
    component_detections = suppress_overlapping_mask_detections(detections_with_masks, max_instances=max_instances)
    global_detections = segment_detections_from_maps_global(
        embedding_map=embedding_map,
        objectness=objectness,
        foreground=foreground,
        objectness_threshold=objectness_threshold,
        embedding_distance=embedding_distance,
        min_cluster_points=min_cluster_points,
        max_instances=max_instances,
        image_size=(width, height),
    )
    detections = choose_dense_detections(component_detections, global_detections)
    if output_size is None or output_size == (width, height):
        return detections
    scale_x = float(output_size[0]) / max(1.0, float(width))
    scale_y = float(output_size[1]) / max(1.0, float(height))
    return [scale_detection(detection, scale_x=scale_x, scale_y=scale_y) for detection in detections]


def segment_detections_from_maps_global(
    *,
    embedding_map: np.ndarray,
    objectness: np.ndarray,
    foreground: np.ndarray,
    objectness_threshold: float,
    embedding_distance: float,
    min_cluster_points: int,
    max_instances: int,
    image_size: tuple[int, int],
) -> list[SegmentDetection]:
    height, width = foreground.shape
    rows, cols = np.nonzero(np.ones((height, width), dtype=bool))
    detections = segment_detections_from_arrays(
        embeddings=embedding_map.reshape((-1, embedding_map.shape[2])),
        objectness=objectness.reshape((-1,)),
        rows=rows.astype(np.int32),
        cols=cols.astype(np.int32),
        image_size=image_size,
        objectness_threshold=float(objectness_threshold),
        embedding_distance=float(embedding_distance),
        min_cluster_points=int(min_cluster_points),
        max_instances=int(max_instances),
    )
    return detections


def choose_dense_detections(
    component_detections: list[SegmentDetection],
    global_detections: list[SegmentDetection],
) -> list[SegmentDetection]:
    if len(component_detections) >= max(1, int(round(len(global_detections) * 0.80))):
        return component_detections
    return global_detections


def detection_from_mask(mask: np.ndarray, confidence: float, min_cluster_points: int) -> SegmentDetection | None:
    if int(mask.sum()) < int(min_cluster_points):
        return None
    bbox = component_bbox(mask)
    polygon = component_polygon(mask, bbox, max_points=96)
    if len(polygon) < 3:
        return None
    return SegmentDetection(
        bbox_xyxy=bbox,
        mask_polygon=polygon,
        detector_label="unknown",
        detector_confidence=max(0.0, min(1.0, float(confidence))),
    )


def suppress_overlapping_mask_detections(
    detections_with_masks: list[tuple[SegmentDetection, np.ndarray]],
    *,
    max_instances: int,
    overlap_threshold: float = 0.82,
) -> list[SegmentDetection]:
    kept: list[tuple[SegmentDetection, np.ndarray]] = []
    ordered = sorted(
        detections_with_masks,
        key=lambda item: (item[0].detector_confidence, int(item[1].sum())),
        reverse=True,
    )
    for detection, mask in ordered:
        if any(mask_iou(mask, kept_mask) >= overlap_threshold for _kept_detection, kept_mask in kept):
            continue
        kept.append((detection, mask))
        if len(kept) >= max_instances:
            break
    detections = [detection for detection, _mask in kept]
    detections.sort(key=lambda item: (item.bbox_xyxy[1], item.bbox_xyxy[0]))
    return detections


def scale_detection(detection: SegmentDetection, *, scale_x: float, scale_y: float) -> SegmentDetection:
    left, top, right, bottom = detection.bbox_xyxy
    polygon = [(float(x) * scale_x, float(y) * scale_y) for x, y in detection.mask_polygon]
    return SegmentDetection(
        bbox_xyxy=(left * scale_x, top * scale_y, right * scale_x, bottom * scale_y),
        mask_polygon=polygon,
        detector_label=detection.detector_label,
        detector_confidence=detection.detector_confidence,
    )


def greedy_embedding_clusters(
    embeddings: np.ndarray,
    *,
    distance: float,
    min_cluster_points: int,
    max_instances: int,
) -> list[np.ndarray]:
    remaining = np.ones(embeddings.shape[0], dtype=bool)
    clusters: list[np.ndarray] = []
    while bool(remaining.any()) and len(clusters) < max_instances:
        remaining_indices = np.flatnonzero(remaining)
        seed_index = int(remaining_indices[0])
        seed = embeddings[seed_index]
        distances = np.linalg.norm(embeddings - seed, axis=1)
        cluster = np.flatnonzero(remaining & (distances <= distance))
        if cluster.size < min_cluster_points:
            remaining[seed_index] = False
            continue
        center = embeddings[cluster].mean(axis=0)
        distances = np.linalg.norm(embeddings - center, axis=1)
        cluster = np.flatnonzero(remaining & (distances <= distance))
        remaining[cluster] = False
        if cluster.size >= min_cluster_points:
            clusters.append(cluster)
    return clusters
