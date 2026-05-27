from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from Input.Depth.depth_loader import load_grayscale_depth
from Input.Image.image_loader import load_rgb_image
from MeshReconstruction.types import MeshResult
from ObjectEnrichment.crops import write_object_crops
from ObjectEnrichment.edge_metrics import measure_edge_evidence
from ObjectEnrichment.geometry_classifier import classify_geometry
from ObjectEnrichment.fusion import fuse_object_state
from ObjectEnrichment.report_writer import write_enrichment_report
from ObjectEnrichment.types import (
    MeshEvidence,
    FUSED_LABELS,
    FusedState,
    ObjectEnrichment,
    ObjectEnrichmentReport,
    WireframeEvidence,
)
from PrimitiveFitting.report_loader import load_detection_report
from SceneGeometry.coordinate_contract import (
    camera_fusion_contract,
    load_fusion_contract_from_camera_metadata,
)
from WireframeDetection.types import NoWireframeProvider


def run_object_enrichment(
    image_path: str | Path,
    depth_path: str | Path,
    detections_path: str | Path,
    output_dir: str | Path,
    edge_provider,
    mesh_provider,
    wireframe_provider=None,
    device: str | None = None,
    seed: int = 20260525,
    max_objects: int = 32,
    max_mesh_objects: int = 16,
    min_edge_mask_pixels: int = 64,
    min_mesh_mask_pixels: int = 256,
    min_wireframe_mask_pixels: int = 64,
    edge_timeout_seconds: int = 120,
    mesh_timeout_seconds: int = 180,
    wireframe_timeout_seconds: int = 120,
) -> ObjectEnrichmentReport:
    resolved_image_path = Path(image_path)
    resolved_depth_path = Path(depth_path)
    resolved_detections_path = Path(detections_path)
    output_path = Path(output_dir)
    wireframe_provider = wireframe_provider or NoWireframeProvider()

    image = load_rgb_image(resolved_image_path)
    depth = load_grayscale_depth(resolved_depth_path, expected_size=image.size)
    detections = load_detection_report(resolved_detections_path)
    if detections.image_width != image.width or detections.image_height != image.height:
        raise ValueError("Detection report image dimensions do not match enrichment image dimensions.")

    output_path.mkdir(parents=True, exist_ok=True)
    edge_result = edge_provider.detect_edges(image)
    edge_map_path = output_path / "edge_map.png"
    edge_result.image.convert("L").save(edge_map_path)
    write_edge_overlay(image, edge_result.image.convert("L"), output_path / "edge_overlay.png")

    objects: list[ObjectEnrichment] = []
    sorted_detections = sorted(detections.objects, key=lambda item: item.id)
    for order_index, detection in enumerate(sorted_detections, start=1):
        object_dir = output_path / "objects" / f"{order_index:02d}"
        if order_index > max_objects:
            objects.append(skipped_object(detection, "max_objects"))
            continue

        paths = write_object_crops(image, depth, edge_result.image, detection, object_dir)
        relative_paths: dict[str, str | None] = {
            "rgb_crop": relative_to_output(paths["rgb_crop"], output_path),
            "mask": relative_to_output(paths["mask"], output_path),
            "depth_crop": relative_to_output(paths["depth_crop"], output_path),
            "edge_crop": relative_to_output(paths["edge_crop"], output_path),
            "crop_metadata": relative_to_output(paths["crop_metadata"], output_path),
            "wireframe_crop": None,
            "wireframe_json": None,
            "evidence_stack": relative_to_output(paths["evidence_stack"], output_path),
            "mesh_candidate": None,
        }
        mask_pixels = count_mask_pixels(paths["mask"])
        if edge_provider.backend == "none":
            edge_evidence = measure_unavailable_edge("backend_none")
        elif mask_pixels >= min_edge_mask_pixels:
            edge_evidence = measure_edge_evidence(str(paths["mask"]), str(paths["edge_crop"]))
        else:
            edge_evidence = measure_unavailable_edge("below_min_mask_pixels")
        geometry = classify_geometry(
            paths["mask"],
            paths["depth_crop"],
            paths["edge_crop"],
            object_dir / "shape_scores.json",
        )
        wireframe_evidence = WireframeEvidence()
        if mask_pixels >= min_wireframe_mask_pixels:
            wireframe_json_path = object_dir / "wireframe.json"
            wireframe_overlay_path = object_dir / "wireframe_crop.png"
            try:
                wireframe_result = wireframe_provider.detect_wireframe(
                    paths["rgb_crop"],
                    paths["mask"],
                    wireframe_json_path,
                    wireframe_overlay_path,
                )
            except Exception as exc:
                wireframe_result = None
                wireframe_evidence = WireframeEvidence(status="failed", reason=str(exc))
            if wireframe_result is not None:
                if wireframe_result.json_path is not None:
                    relative_paths["wireframe_json"] = relative_to_output(wireframe_result.json_path, output_path)
                if wireframe_result.overlay_path is not None:
                    relative_paths["wireframe_crop"] = relative_to_output(wireframe_result.overlay_path, output_path)
                wireframe_evidence = WireframeEvidence(
                    status=wireframe_result.status,
                    line_count=wireframe_result.line_count,
                    junction_count=wireframe_result.junction_count,
                    reason=wireframe_result.reason,
                )
        else:
            wireframe_evidence = WireframeEvidence(status="skipped", reason="below_min_mask_pixels")

        mesh_result = MeshResult(status="skipped", path=None, reason="below_min_mask_pixels")
        if order_index > max_mesh_objects:
            mesh_result = MeshResult(status="skipped", path=None, reason="max_mesh_objects")
        elif mask_pixels >= min_mesh_mask_pixels:
            mesh_output_path = object_dir / "mesh_candidate.obj"
            try:
                mesh_result = mesh_provider.reconstruct(paths["rgb_crop"], paths["mask"], mesh_output_path)
            except Exception as exc:  # provider errors are reported per object after preflight succeeds
                mesh_result = MeshResult(status="failed", path=None, reason=str(exc))
            if mesh_result.path is not None:
                relative_paths["mesh_candidate"] = relative_to_output(mesh_result.path, output_path)

        fused = fuse_object_state(
            detection=detection,
            geometry=geometry,
            edge=edge_evidence,
            wireframe=wireframe_evidence,
            mesh=MeshResult(
                status=mesh_result.status,
                path=relative_paths["mesh_candidate"],
                reason=mesh_result.reason,
            ),
        )
        mesh_evidence = MeshEvidence(
            status=mesh_result.status,
            path=relative_paths["mesh_candidate"],
            reason=mesh_result.reason,
        )
        objects.append(
            ObjectEnrichment(
                id=detection.id,
                status="ok",
                error=None,
                original_detector_label=detection.detector_label,
                detector_confidence=detection.detector_confidence,
                paths=relative_paths,
                edge=edge_evidence,
                wireframe=wireframe_evidence,
                mesh=mesh_evidence,
                geometry=geometry,
                fused_state=fused,
                fused_label=fused.fused_label,
                fused_confidence=fused.fused_confidence,
                fused_contributions=fused.fused_contributions,
                needs_review=fused.needs_review,
                needs_review_reason=fused.needs_review_reason,
            )
        )

    report = ObjectEnrichmentReport(
        image_path=str(resolved_image_path),
        depth_path=str(resolved_depth_path),
        detections_path=str(resolved_detections_path),
        model_info={
            "edge_backend": edge_provider.backend,
            "edge_model_dir": str(edge_provider.model_dir) if edge_provider.model_dir else None,
            "mesh_backend": mesh_provider.backend,
            "mesh_model_dir": str(mesh_provider.model_dir) if mesh_provider.model_dir else None,
            "wireframe_backend": wireframe_provider.backend,
            "wireframe_model_dir": str(wireframe_provider.model_dir) if wireframe_provider.model_dir else None,
            "geometry_backend": "deterministic_geometry_v1",
            "device": device,
            "seed": seed,
            "edge_timeout_seconds": edge_timeout_seconds,
            "mesh_timeout_seconds": mesh_timeout_seconds,
            "wireframe_timeout_seconds": wireframe_timeout_seconds,
            "fusion_contract": load_source_fusion_contract(resolved_image_path, image.width, image.height),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        objects=objects,
    )
    write_enrichment_report(report, output_path / "object_enrichment.json")
    return report


def relative_to_output(path: Path, output_path: Path) -> str:
    return str(path.relative_to(output_path))


def count_mask_pixels(mask_path: Path) -> int:
    with Image.open(mask_path) as image:
        return int((np.asarray(image.convert("L"), dtype=np.uint8) > 127).sum())


def measure_unavailable_edge(reason: str):
    from ObjectEnrichment.types import EdgeEvidence

    return EdgeEvidence(status="not_available", boundary_agreement=0.0, edge_density=0.0)


def load_source_fusion_contract(image_path: Path, width: int, height: int) -> dict:
    camera_path = image_path.parent / "camera.json"
    if camera_path.is_file():
        try:
            import json

            metadata = json.loads(camera_path.read_text(encoding="utf-8"))
            return load_fusion_contract_from_camera_metadata(metadata)
        except (OSError, ValueError, TypeError):
            pass
    return camera_fusion_contract(image_width=width, image_height=height)


def skipped_object(detection, reason: str) -> ObjectEnrichment:
    from ObjectEnrichment.types import EdgeEvidence, GeometryEvidence

    return ObjectEnrichment(
        id=detection.id,
        status="skipped",
        error=reason,
        original_detector_label=detection.detector_label,
        detector_confidence=detection.detector_confidence,
        paths={
            "rgb_crop": None,
            "mask": None,
            "depth_crop": None,
            "edge_crop": None,
            "crop_metadata": None,
            "wireframe_crop": None,
            "wireframe_json": None,
            "evidence_stack": None,
            "mesh_candidate": None,
        },
        edge=EdgeEvidence(status="not_available", boundary_agreement=0.0, edge_density=0.0),
        wireframe=WireframeEvidence(),
        mesh=MeshEvidence(status="skipped", path=None, reason=reason),
        geometry=GeometryEvidence(
            selected_label="unknown",
            confidence=0.0,
            candidate_scores={"unknown": 1.0},
        ),
        fused_state=FusedState(
            fused_label="unknown",
            fused_confidence=0.0,
            fused_contributions=_empty_fused_contributions(reason=reason),
            needs_review=True,
            needs_review_reason=[reason],
        ),
        fused_label="unknown",
        fused_confidence=0.0,
        fused_contributions=_empty_fused_contributions(reason=reason),
        needs_review=True,
        needs_review_reason=[reason],
    )


def _empty_fused_contributions(reason: str) -> dict[str, dict[str, Any]]:
    empty_scores = {label: 0.0 for label in FUSED_LABELS}
    return {
        "detector": {
            "status": "skipped",
            "selected_label": "unknown",
            "selected_score": 0.0,
            "label_scores": dict(empty_scores),
            "evidence": {"reason": reason},
        },
        "depth": {
            "status": "skipped",
            "selected_label": "unknown",
            "selected_score": 0.0,
            "label_scores": dict(empty_scores),
            "evidence": {"reason": reason},
        },
        "edge": {
            "status": "skipped",
            "selected_label": "unknown",
            "selected_score": 0.0,
            "label_scores": dict(empty_scores),
            "evidence": {"reason": reason},
        },
        "wireframe": {
            "status": "skipped",
            "selected_label": "unknown",
            "selected_score": 0.0,
            "label_scores": dict(empty_scores),
            "evidence": {"reason": reason},
        },
        "mesh": {
            "status": "skipped",
            "selected_label": "unknown",
            "selected_score": 0.0,
            "label_scores": dict(empty_scores),
            "evidence": {
                "mesh_status": "skipped",
                "has_candidate_path": False,
                "mesh_path": None,
                "reason": reason,
            },
        },
        "fusion": {
            "status": "missing",
            "label_scores": dict(empty_scores),
            "weights": {"detector": 0.0, "depth": 0.0, "edge": 0.0, "wireframe": 0.0, "mesh": 0.0},
            "active_modalities": [],
            "reason": reason,
        },
    }


def write_edge_overlay(image: Image.Image, edge: Image.Image, output_path: Path) -> None:
    output = image.convert("RGBA")
    red = Image.new("RGBA", image.size, (255, 32, 32, 160))
    mask = edge.point(lambda value: 255 if value > 40 else 0)
    output.alpha_composite(Image.composite(red, Image.new("RGBA", image.size, (0, 0, 0, 0)), mask))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.convert("RGB").save(output_path)
