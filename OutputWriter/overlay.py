from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ShapeDetection.report import ObjectShapeDetection


PRIMITIVE_COLORS = {
    "sphere": (42, 157, 143),
    "cylinder": (38, 70, 83),
    "cone": (230, 57, 70),
    "box": (244, 162, 97),
    "plane": (69, 123, 157),
    "unknown": (108, 117, 125),
}


def write_overlay(
    image: Image.Image,
    objects: list[ObjectShapeDetection],
    path: str | Path,
) -> None:
    output = image.copy().convert("RGB")
    if objects:
        draw = ImageDraw.Draw(output, "RGBA")
        font = ImageFont.load_default()
        for item in objects:
            display_label, display_confidence, color_label = _display_values(item)
            color = PRIMITIVE_COLORS.get(color_label, PRIMITIVE_COLORS["unknown"])
            outline = (*color, 255)
            fill = (*color, 64)
            if len(item.mask_polygon) >= 3:
                draw.polygon(item.mask_polygon, fill=fill, outline=outline)
                points = item.mask_polygon + [item.mask_polygon[0]]
                draw.line(points, fill=outline, width=3)
            else:
                draw.rectangle(item.bbox_xyxy, outline=outline, width=3)

            label = f"{item.id:02d} {display_label} {display_confidence:.2f}"
            _draw_label(draw, item.bbox_xyxy, label, color, font)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_path)


def _display_values(item: ObjectShapeDetection) -> tuple[str, float, str]:
    if item.primitive_label_source == "unassigned" and item.primitive_label == "unknown":
        return item.detector_label, item.detector_confidence, item.detector_label
    return item.primitive_label, item.primitive_confidence, item.primitive_label


def _draw_label(
    draw: ImageDraw.ImageDraw,
    bbox_xyxy: tuple[float, float, float, float],
    label: str,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    left, top, _, _ = bbox_xyxy
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    x = max(0, int(round(left)))
    y = max(0, int(round(top)) - text_height - 6)
    if y == 0:
        y = max(0, int(round(top)) + 4)
    draw.rectangle(
        (x, y, x + text_width + 8, y + text_height + 6),
        fill=(*color, 230),
    )
    draw.text((x + 4, y + 3), label, fill=(255, 255, 255, 255), font=font)
