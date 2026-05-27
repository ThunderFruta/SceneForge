from __future__ import annotations

from pathlib import Path

from PIL import Image

from EdgeDetection.types import EdgeProvider
from Input.Depth.depth_loader import load_grayscale_depth
from Segmentation.backend import LearnedSegmentationModelSpec
from Segmentation.primitive_3d import (
    FEATURE_NAMES,
    IMAGE_FEATURE_NAMES,
    build_point_cloud_arrays,
    build_rgbd_mask_arrays,
    load_checkpoint,
    segment_detections_from_arrays,
    segment_detections_from_maps,
)
from Segmentation.types import SegmentDetection


class LearnedDepthEdgeSegmenter:
    """Primitive3D class-agnostic instance-mask detector.

    The adapter consumes aligned RGB/depth as a visible camera-space point cloud,
    emits instance masks, and leaves primitive classification to geometry/fusion.
    """

    # TODO(segmentation): Add a semantic background/object/plane head and cluster
    # only object pixels with embeddings; broad plane masks should be handled by
    # semantic plane output plus depth-geometry cleanup instead of one shared
    # class-agnostic embedding space.
    backend = "primitive-3d-segmenter"
    input_channels = FEATURE_NAMES

    def __init__(
        self,
        model_path: str | Path,
        depth_path: str | Path,
        edge_path: str | Path | None = None,
        edge_provider: EdgeProvider | None = None,
        device: str | None = None,
        max_components: int | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.depth_path = Path(depth_path)
        self.edge_path = Path(edge_path) if edge_path else None
        self.edge_provider = edge_provider
        self.device = device
        self.max_components = max_components
        self.model, self.checkpoint_metadata = load_checkpoint(self.model_path, device=self.device or "cpu")
        model_config = self.checkpoint_metadata.get("model", {})
        self.max_points = int(model_config.get("max_points", 4096))
        self.min_cluster_points = int(model_config.get("min_cluster_points", 32))
        self.embedding_distance = float(model_config.get("embedding_distance", 0.35))
        self.objectness_threshold = float(model_config.get("objectness_threshold", 0.5))
        self.fov_degrees = float(model_config.get("fov_degrees", 70.0))
        self.near_depth = float(model_config.get("near_depth", 1.0))
        self.far_depth = float(model_config.get("far_depth", 8.0))
        self.model_family = str(model_config.get("model_family", "pointnet_embedding"))
        self.image_size = int(model_config.get("image_size", 256))
        self.image_feature_names = tuple(self.checkpoint_metadata.get("input_channels", IMAGE_FEATURE_NAMES))
        self.backend_info = LearnedSegmentationModelSpec(
            architecture=str(self.checkpoint_metadata.get("architecture", "primitive_3d_point_embedding_v1")),
            input_channels=self.image_feature_names if self.model_family == "rgbd_mask_embedding" else FEATURE_NAMES,
            output_contract=str(self.checkpoint_metadata.get("output_contract", "class_agnostic_instance_masks")),
        ).to_backend_info(
            name=self.backend,
            model_path=str(self.model_path),
        )

    def detect(self, image: Image.Image) -> list[SegmentDetection]:
        torch = __import__("torch")
        depth = load_grayscale_depth(self.depth_path, expected_size=image.size)
        if self.model_family == "rgbd_mask_embedding":
            arrays = build_rgbd_mask_arrays(
                image=image,
                depth=depth,
                masks=None,
                image_size=self.image_size,
                feature_names=self.image_feature_names,
            )
            features = arrays["features"]
            with torch.no_grad():
                tensor = torch.from_numpy(features).to(device=self.device or "cpu", dtype=torch.float32).unsqueeze(0)
                embeddings, objectness_logits = self.model(tensor)
                embeddings_np = embeddings.squeeze(0).detach().cpu().numpy()
                objectness_np = torch.sigmoid(objectness_logits.squeeze(0)).detach().cpu().numpy()
            return segment_detections_from_maps(
                embeddings=embeddings_np,
                objectness=objectness_np,
                objectness_threshold=self.objectness_threshold,
                embedding_distance=self.embedding_distance,
                min_cluster_points=self.min_cluster_points,
                max_instances=int(self.max_components or 32),
                output_size=image.size,
            )

        arrays = build_point_cloud_arrays(
            image=image,
            depth=depth,
            masks=None,
            max_points=self.max_points,
            fov_degrees=self.fov_degrees,
            near_depth=self.near_depth,
            far_depth=self.far_depth,
        )
        features = arrays["features"]
        if features.size == 0:
            return []
        device = self.device or "cpu"
        with torch.no_grad():
            tensor = torch.from_numpy(features).to(device=device, dtype=torch.float32).unsqueeze(0)
            embeddings, objectness_logits = self.model(tensor)
            embeddings_np = embeddings.squeeze(0).detach().cpu().numpy()
            objectness_np = torch.sigmoid(objectness_logits.squeeze(0)).detach().cpu().numpy()
        return segment_detections_from_arrays(
            embeddings=embeddings_np,
            objectness=objectness_np,
            rows=arrays["rows"],
            cols=arrays["cols"],
            image_size=image.size,
            objectness_threshold=self.objectness_threshold,
            embedding_distance=self.embedding_distance,
            min_cluster_points=self.min_cluster_points,
            max_instances=int(self.max_components or 32),
        )
