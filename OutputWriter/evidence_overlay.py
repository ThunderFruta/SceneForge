from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ObjectEnrichment.report_loader import load_enrichment_report
from PrimitiveFitting.report_loader import load_detection_report
from ShapeDetection.report import ObjectShapeDetection


YOLO_COLOR = (255, 92, 92, 255)
EDGE_COLOR = (0, 210, 255, 150)
WIREFRAME_COLOR = (255, 232, 0, 255)
MESH_OK_COLOR = (190, 80, 255, 255)
MESH_OTHER_COLOR = (160, 160, 160, 255)
LABEL_BG = (18, 22, 26, 220)


def write_evidence_overlay(
    *,
    image_path: str | Path,
    detections_path: str | Path,
    enrichment_path: str | Path,
    output_path: str | Path,
    edge_map_path: str | Path | None = None,
) -> None:
    image_path = Path(image_path)
    detections_path = Path(detections_path)
    enrichment_path = Path(enrichment_path)
    output_path = Path(output_path)
    enrichment_root = enrichment_path.parent

    image = Image.open(image_path).convert("RGBA")
    detections = load_detection_report(detections_path)
    enrichment = load_enrichment_report(enrichment_path)
    if detections.image_width != image.width or detections.image_height != image.height:
        raise ValueError("Detection report dimensions do not match overlay image dimensions.")

    enrichment_by_id = {item.id: item for item in enrichment.objects}
    if {item.id for item in detections.objects} != set(enrichment_by_id):
        raise ValueError("Detection/enrichment ids do not match for evidence overlay.")

    output = image.copy()
    draw = ImageDraw.Draw(output, "RGBA")
    font = ImageFont.load_default()

    for detection in sorted(detections.objects, key=lambda item: item.id):
        draw_yolo_detection(draw, detection)

    draw_edges(output, edge_map_path or enrichment_root / "edge_map.png")
    draw = ImageDraw.Draw(output, "RGBA")
    for detection in sorted(detections.objects, key=lambda item: item.id):
        enrichment_object = enrichment_by_id[detection.id]
        draw_wireframe(draw, enrichment_root, enrichment_object.paths, detection)
        draw_mesh_marker(draw, font, enrichment_root, detection, enrichment_object.mesh.status, enrichment_object.mesh.path)
        label, confidence = _selected_overlay_label(enrichment_object)
        draw_object_label(draw, font, detection, label, confidence)

    draw_legend(draw, font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.convert("RGB").save(output_path)


def draw_edges(output: Image.Image, edge_path: Path) -> None:
    if not edge_path.is_file():
        return
    edge = Image.open(edge_path).convert("L").resize(output.size)
    edge_mask = edge.point(lambda value: 255 if value > 32 else 0)
    layer = Image.new("RGBA", output.size, EDGE_COLOR)
    output.alpha_composite(Image.composite(layer, Image.new("RGBA", output.size, (0, 0, 0, 0)), edge_mask))


def draw_yolo_detection(draw: ImageDraw.ImageDraw, detection: ObjectShapeDetection) -> None:
    if len(detection.mask_polygon) >= 3:
        draw.polygon(detection.mask_polygon, outline=YOLO_COLOR)
        draw.line(detection.mask_polygon + [detection.mask_polygon[0]], fill=YOLO_COLOR, width=2)
    draw.rectangle(detection.bbox_xyxy, outline=YOLO_COLOR, width=2)


def draw_wireframe(
    draw: ImageDraw.ImageDraw,
    enrichment_root: Path,
    paths: dict[str, str | None],
    detection: ObjectShapeDetection,
) -> None:
    wireframe_path = resolve_relative(enrichment_root, paths.get("wireframe_json"))
    crop_metadata_path = resolve_relative(enrichment_root, paths.get("crop_metadata"))
    if wireframe_path is None:
        return
    if not wireframe_path.is_file():
        return
    try:
        wireframe = json.loads(wireframe_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if crop_metadata_path is not None and crop_metadata_path.is_file():
        try:
            crop_metadata = json.loads(crop_metadata_path.read_text(encoding="utf-8"))
            x0, y0, _x1, _y1 = [float(value) for value in crop_metadata.get("crop_box_xyxy", [0, 0, 0, 0])]
        except (OSError, ValueError):
            x0, y0 = float(detection.bbox_xyxy[0]), float(detection.bbox_xyxy[1])
    else:
        x0, y0 = float(detection.bbox_xyxy[0]), float(detection.bbox_xyxy[1])
    for raw_line in wireframe.get("lines", []):
        if len(raw_line) < 4:
            continue
        x1, y1, x2, y2 = [float(value) for value in raw_line[:4]]
        score = float(raw_line[4]) if len(raw_line) >= 5 else 1.0
        alpha = int(max(90, min(255, 90 + 165 * score)))
        color = (*WIREFRAME_COLOR[:3], alpha)
        draw.line((x0 + x1, y0 + y1, x0 + x2, y0 + y2), fill=color, width=3)
        for x, y in ((x0 + x1, y0 + y1), (x0 + x2, y0 + y2)):
            r = 2
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color)


def draw_mesh_marker(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    enrichment_root: Path,
    detection: ObjectShapeDetection,
    mesh_status: str,
    mesh_path: str | None,
) -> None:
    left, top, right, bottom = detection.bbox_xyxy
    cx = int(round((left + right) * 0.5))
    cy = int(round((top + bottom) * 0.5))
    vertex_count = mesh_vertex_count(resolve_relative(enrichment_root, mesh_path))
    suffix = f" {vertex_count}v" if vertex_count is not None else ""
    if mesh_status == "ok":
        size = 7
        draw.polygon(
            [(cx, cy - size), (cx + size, cy), (cx, cy + size), (cx - size, cy)],
            fill=MESH_OK_COLOR,
            outline=(255, 255, 255, 230),
        )
        draw_small_text(draw, font, f"mesh:{mesh_status}{suffix}", (cx + size + 3, cy - 8), MESH_OK_COLOR)
        return
    draw_small_text(draw, font, f"mesh:{mesh_status}{suffix}", (cx + 4, cy - 8), MESH_OTHER_COLOR)


def draw_object_label(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    detection: ObjectShapeDetection,
    geometry_label: str,
    geometry_confidence: float,
) -> None:
    left, top, _right, _bottom = detection.bbox_xyxy
    label = (
        f"{detection.id:02d} yolo:{detection.detector_label} {detection.detector_confidence:.2f} "
        f"geo:{geometry_label} {geometry_confidence:.2f}"
    )
    draw_small_text(draw, font, label, (int(round(left)), max(0, int(round(top)) - 14)), YOLO_COLOR)


def _selected_overlay_label(enrichment_object) -> tuple[str, float]:
    fused_state = getattr(enrichment_object, "fused_state", None)
    if fused_state is not None:
        return fused_state.fused_label, float(fused_state.fused_confidence)
    return enrichment_object.geometry.selected_label, float(enrichment_object.geometry.confidence)


def draw_small_text(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    text: str,
    xy: tuple[int, int],
    color: tuple[int, int, int, int],
) -> None:
    x, y = xy
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    draw.rectangle((x, y, x + width + 6, y + height + 5), fill=LABEL_BG)
    draw.text((x + 3, y + 2), text, fill=color, font=font)


def draw_legend(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont) -> None:
    entries = [
        ("YOLO mask/box", YOLO_COLOR),
        ("edge", EDGE_COLOR),
        ("wireframe", WIREFRAME_COLOR),
        ("mesh marker", MESH_OK_COLOR),
    ]
    x = 8
    y = 8
    for label, color in entries:
        draw.rectangle((x, y + 2, x + 10, y + 12), fill=color)
        draw.text((x + 14, y), label, fill=(255, 255, 255, 240), font=font)
        y += 15


def resolve_relative(root: Path, path: str | Path | None) -> Path | None:
    if path is None:
        return None
    value = Path(path)
    return value if value.is_absolute() else root / value


def mesh_vertex_count(path: Path | None) -> int | None:
    if path is None or not path.is_file():
        return None
    count = 0
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.startswith("v "):
                    count += 1
    except OSError:
        return None
    return count
