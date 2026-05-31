from __future__ import annotations

import base64
import io

from PIL import Image

from ObjectCompletion.openai_image import (
    build_application_query_prompt,
    build_openai_prompt,
    call_image_edit_api,
    decode_image_response,
    ensure_transparent_completed_image,
    flatten_transparency_on_white,
    render_application_query,
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


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_openai_prompt_requests_white_background() -> None:
    prompt = build_openai_prompt("chair")

    assert "plain white background" in prompt
    assert "pure white" in prompt
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


def test_openai_input_flattens_transparency_to_white() -> None:
    crop = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
    for x in range(8, 12):
        for y in range(6, 14):
            crop.putpixel((x, y), (120, 80, 40, 255))

    flattened = flatten_transparency_on_white(render_target_square(crop, canvas_size=64))

    assert flattened.mode == "RGB"
    assert flattened.getpixel((0, 0)) == (255, 255, 255)
    assert flattened.getpixel((32, 32)) == (120, 80, 40)


def test_application_query_prompt_keeps_output_object_only() -> None:
    prompt = build_application_query_prompt("vase")

    assert "Application-Querying layout" in prompt
    assert "Extracted Object" in prompt
    assert "plain white background" in prompt
    assert "two-panel layout" in prompt
    assert "Do not include floor" in prompt


def test_render_application_query_writes_two_panel_context() -> None:
    context = Image.new("RGB", (80, 60), (90, 120, 140))
    target = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
    for x in range(10, 30):
        for y in range(8, 32):
            target.putpixel((x, y), (120, 80, 40, 255))

    rendered = render_application_query(context, target, label="chair", canvas_size=128)

    assert rendered.mode == "RGB"
    assert rendered.size[0] > rendered.size[1]
    assert rendered.getbbox() is not None


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


def test_image_edit_requests_opaque_background(tmp_path) -> None:
    input_path = tmp_path / "input.png"
    input_path.write_bytes(png_bytes(Image.new("RGBA", (8, 8), (0, 0, 0, 0))))
    result_image = Image.new("RGB", (8, 8), (245, 245, 240))
    calls = []

    class FakeImages:
        def edit(self, **kwargs):
            calls.append(dict(kwargs))
            return type(
                "Result",
                (),
                {
                    "data": [
                        type(
                            "Item",
                            (),
                            {"b64_json": base64.b64encode(png_bytes(result_image)).decode("ascii")},
                        )()
                    ]
                },
            )()

    client = type("Client", (), {"images": FakeImages()})()

    image = call_image_edit_api(
        client=client,
        model="gpt-image-1.5",
        prompt="complete object",
        input_path=input_path,
        reference_path=None,
        canvas_size=1024,
    )

    assert image.mode == "RGBA"
    assert [call["background"] for call in calls] == ["opaque"]
