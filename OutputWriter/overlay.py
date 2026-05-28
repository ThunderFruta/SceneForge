from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageStat

from ShapeDetection.report import ObjectShapeDetection


INSTANCE_COLORS = [
    (0, 95, 204),
    (176, 0, 32),
    (0, 107, 63),
    (138, 28, 124),
    (0, 109, 119),
    (112, 66, 20),
    (75, 46, 131),
    (0, 48, 73),
    (255, 190, 11),
    (0, 245, 212),
    (251, 86, 7),
    (131, 255, 0),
    (255, 0, 110),
    (58, 134, 255),
    (255, 214, 10),
    (6, 214, 160),
]


DEFAULT_MASK_ALPHA = 8
DEFAULT_OUTLINE_ALPHA = 235
DEFAULT_OUTLINE_WIDTH = 2
DEFAULT_LABEL_BG_ALPHA = 220
DEFAULT_BOX_ALPHA = 255


def _clamp_alpha(value: int) -> int:
    return max(0, min(255, int(value)))


def write_overlay(
    image: Image.Image,
    objects: list[ObjectShapeDetection],
    path: str | Path,
    *,
    mask_alpha: int = DEFAULT_MASK_ALPHA,
    outline_alpha: int = DEFAULT_OUTLINE_ALPHA,
    outline_width: int = DEFAULT_OUTLINE_WIDTH,
    label_background_alpha: int = DEFAULT_LABEL_BG_ALPHA,
) -> None:
    output = image.copy().convert("RGB")
    if objects:
        draw = ImageDraw.Draw(output, "RGBA")
        font = ImageFont.load_default()
        fill_alpha = _clamp_alpha(mask_alpha)
        edge_alpha = _clamp_alpha(outline_alpha)
        label_alpha = _clamp_alpha(label_background_alpha)
        used_colors: set[tuple[int, int, int]] = set()
        for item in objects:
            display_label, display_confidence, _color_label = _display_values(item)
            color = _contrast_color(output, item, used_colors)
            used_colors.add(color)
            outline = (*color, edge_alpha)
            fill = (*color, fill_alpha)
            if len(item.mask_polygon) >= 3:
                draw.polygon(item.mask_polygon, fill=fill, outline=outline)
                points = item.mask_polygon + [item.mask_polygon[0]]
                draw.line(points, fill=outline, width=max(1, int(outline_width)))
            _draw_detection_box(draw, item.bbox_xyxy, color, width=max(2, int(outline_width) + 1))

            label = f"{item.id:02d} {display_label} {display_confidence:.2f}"
            _draw_label(draw, item.bbox_xyxy, label, color, font, label_alpha)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_path)


def _display_values(item: ObjectShapeDetection) -> tuple[str, float, str]:
    if item.primitive_label_source == "unassigned" and item.primitive_label == "unknown":
        return item.detector_label, item.detector_confidence, item.detector_label
    return item.primitive_label, item.primitive_confidence, item.primitive_label


def _contrast_color(
    image: Image.Image,
    item: ObjectShapeDetection,
    used_colors: set[tuple[int, int, int]],
) -> tuple[int, int, int]:
    background_luminance = _bbox_luminance(image, item.bbox_xyxy)
    ranked = sorted(
        INSTANCE_COLORS,
        key=lambda color: (
            color in used_colors,
            -_contrast_ratio(_relative_luminance(color), background_luminance),
            INSTANCE_COLORS.index(color),
        ),
    )
    return ranked[0]


def _bbox_luminance(image: Image.Image, bbox_xyxy: tuple[float, float, float, float]) -> float:
    width, height = image.size
    left, top, right, bottom = bbox_xyxy
    box = (
        max(0, min(width, int(round(left)))),
        max(0, min(height, int(round(top)))),
        max(0, min(width, int(round(right)))),
        max(0, min(height, int(round(bottom)))),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        return 1.0
    mean = ImageStat.Stat(image.crop(box).convert("L")).mean[0] / 255.0
    return max(0.0, min(1.0, mean))


def _relative_luminance(color: tuple[int, int, int]) -> float:
    red, green, blue = (channel / 255.0 for channel in color)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _contrast_ratio(first: float, second: float) -> float:
    lighter = max(first, second)
    darker = min(first, second)
    return (lighter + 0.05) / (darker + 0.05)


def _draw_detection_box(
    draw: ImageDraw.ImageDraw,
    bbox_xyxy: tuple[float, float, float, float],
    color: tuple[int, int, int],
    width: int,
) -> None:
    halo = (255, 255, 255, 210) if _relative_luminance(color) < 0.45 else (0, 0, 0, 210)
    draw.rectangle(bbox_xyxy, outline=halo, width=max(1, width + 2))
    draw.rectangle(bbox_xyxy, outline=(*color, DEFAULT_BOX_ALPHA), width=max(1, width))


def _draw_label(
    draw: ImageDraw.ImageDraw,
    bbox_xyxy: tuple[float, float, float, float],
    label: str,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
    background_alpha: int,
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
        fill=(*color, background_alpha),
    )
    text_fill = (0, 0, 0, 255) if _relative_luminance(color) > 0.58 else (255, 255, 255, 255)
    draw.text((x + 4, y + 3), label, fill=text_fill, font=font)
