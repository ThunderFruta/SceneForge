from __future__ import annotations

import base64
import io

from PIL import Image

from ObjectCompletion.openai_image import (
    build_openai_prompt,
    decode_image_response,
    ensure_transparent_completed_image,
    render_target_square,
)


class FakeImageGenerationCall:
    type = "image_generation_call"

    def __init__(self, result: str) -> None:
        self.result = result


class FakeResponse:
    def __init__(self, image: Image.Image) -> None:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        self.output = [FakeImageGenerationCall(base64.b64encode(buffer.getvalue()).decode("ascii"))]


def test_openai_prompt_requests_transparent_background() -> None:
    prompt = build_openai_prompt("chair")

    assert "transparent background" in prompt
    assert "fully transparent" in prompt
    assert "no floor" in prompt.lower()
    assert "ground plane" in prompt
    assert "plain neutral background" not in prompt


def test_render_target_square_preserves_transparent_background() -> None:
    crop = Image.new("RGBA", (20, 10), (0, 0, 0, 0))
    for x in range(5, 15):
        for y in range(2, 8):
            crop.putpixel((x, y), (120, 80, 40, 255))

    rendered = render_target_square(crop, canvas_size=100)

    assert rendered.mode == "RGBA"
    assert rendered.getpixel((0, 0))[3] == 0
    assert rendered.getchannel("A").getbbox() is not None


def test_decode_image_response_preserves_alpha() -> None:
    image = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    image.putpixel((3, 3), (255, 0, 0, 255))

    decoded = decode_image_response(FakeResponse(image))

    assert decoded.mode == "RGBA"
    assert decoded.getpixel((0, 0))[3] == 0
    assert decoded.getpixel((3, 3))[3] == 255


def test_opaque_neutral_background_is_converted_to_transparency() -> None:
    image = Image.new("RGB", (32, 32), (245, 245, 240))
    for x in range(10, 22):
        for y in range(8, 24):
            image.putpixel((x, y), (60, 40, 30))

    transparent = ensure_transparent_completed_image(image)

    assert transparent.mode == "RGBA"
    assert transparent.getpixel((0, 0))[3] == 0
    assert transparent.getpixel((16, 16))[3] == 255
