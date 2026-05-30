from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ObjectReconstruction.hunyuan3d_objects import (
    apply_hunyuan3d_hf_cache_env,
    count_obj_faces,
    parse_removed_faces,
    prepare_completed_masks,
    prepare_paint_reference_image,
    should_remove_hunyuan_support_sheet,
    texture_object_dir,
    validate_hunyuan_texture_options,
)
from ObjectReconstruction.triposr_objects import clean_reconstruction_outputs, prepare_reconstruction_input
from ObjectReconstruction.triposr_objects import COMPLETED_MASK_METADATA_NAME, COMPLETED_MASK_NAME
from Segmentation.types import SegmentDetection


class FakeSam3Segmenter:
    def __init__(self) -> None:
        self.text_prompt = ""

    def detect(self, image: Image.Image) -> list[SegmentDetection]:
        return [
            SegmentDetection(
                bbox_xyxy=(20.0, 10.0, 90.0, 80.0),
                mask_polygon=[(20.0, 10.0), (90.0, 10.0), (90.0, 80.0), (20.0, 80.0)],
                detector_label="chair",
                detector_confidence=0.95,
                proposal_source="sam3_text_prompt",
            )
        ]


class BrokenSam3Segmenter:
    def detect(self, image: Image.Image) -> list[SegmentDetection]:
        raise RuntimeError("sam3 failed")


class FakePaintConfig:
    device = "cpu"
    image_caption = "test prompt"


class FakePaintPipeline:
    def __init__(self) -> None:
        self.config = FakePaintConfig()
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        *,
        mesh_path: str,
        image_path: str,
        output_mesh_path: str,
        use_remesh: bool,
        save_glb: bool,
    ) -> None:
        self.calls.append(
            {
                "mesh_path": mesh_path,
                "image_path": image_path,
                "output_mesh_path": output_mesh_path,
                "use_remesh": use_remesh,
                "save_glb": save_glb,
            }
        )
        Path(output_mesh_path).write_text("textured", encoding="utf-8")
        if save_glb:
            Path(output_mesh_path).with_suffix(".glb").write_text("textured glb", encoding="utf-8")


def test_completed_crop_uses_regenerated_sam3_mask(tmp_path: Path) -> None:
    object_dir = tmp_path / "01_chair"
    object_dir.mkdir()
    completed = object_dir / "completed_crop.png"
    Image.new("RGBA", (100, 100), (240, 240, 240, 255)).save(completed)

    stale_original = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    stale_original.putpixel((5, 5), (255, 255, 255, 255))
    stale_original.save(object_dir / "masked_crop.png")

    segmenter = FakeSam3Segmenter()
    prepared = prepare_reconstruction_input(
        completed,
        object_dir,
        completed_mask_backend="sam3",
        completed_mask_segmenter=segmenter,
        completed_mask_prompt="chair . foreground object .",
    )

    assert prepared.mask_source == "completed_sam3"
    assert prepared.completed_mask_path == COMPLETED_MASK_NAME
    assert (object_dir / COMPLETED_MASK_NAME).is_file()
    assert (object_dir / COMPLETED_MASK_METADATA_NAME).is_file()
    assert segmenter.text_prompt == "chair . foreground object ."

    values = np.asarray(prepared.mask.convert("L"), dtype=np.uint8)
    assert values[40, 40] > 0
    assert values[5, 5] == 0


def test_completed_crop_prefers_existing_completed_mask(tmp_path: Path) -> None:
    object_dir = tmp_path / "01_chair"
    object_dir.mkdir()
    completed = object_dir / "completed_crop.png"
    Image.new("RGBA", (50, 50), (240, 240, 240, 255)).save(completed)

    completed_mask = Image.new("L", (50, 50), 0)
    for x in range(10, 30):
        for y in range(12, 35):
            completed_mask.putpixel((x, y), 255)
    completed_mask_path = object_dir / COMPLETED_MASK_NAME
    completed_mask_path.parent.mkdir(parents=True, exist_ok=True)
    completed_mask.save(completed_mask_path)

    prepared = prepare_reconstruction_input(completed, object_dir, completed_mask_backend="auto")

    values = np.asarray(prepared.mask.convert("L"), dtype=np.uint8)
    assert prepared.mask_source == "completed_mask"
    assert values[20, 20] > 0
    assert values[3, 3] == 0


def test_explicit_sam3_refreshes_existing_completed_mask(tmp_path: Path) -> None:
    object_dir = tmp_path / "01_chair"
    object_dir.mkdir()
    completed = object_dir / "completed_crop.png"
    Image.new("RGBA", (100, 100), (240, 240, 240, 255)).save(completed)
    completed_mask_path = object_dir / COMPLETED_MASK_NAME
    completed_mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (100, 100), 0).save(completed_mask_path)

    prepared = prepare_reconstruction_input(
        completed,
        object_dir,
        completed_mask_backend="sam3",
        completed_mask_segmenter=FakeSam3Segmenter(),
        completed_mask_prompt="chair . foreground object .",
    )

    values = np.asarray(prepared.mask.convert("L"), dtype=np.uint8)
    assert prepared.mask_source == "completed_sam3"
    assert values[40, 40] > 0


def test_auto_completed_mask_falls_back_when_sam3_fails(tmp_path: Path) -> None:
    object_dir = tmp_path / "01_chair"
    object_dir.mkdir()
    Image.new("RGBA", (50, 50), (240, 240, 240, 255)).save(object_dir / "completed_crop.png")

    prepare_completed_masks(
        [object_dir],
        source="completed",
        completed_mask_backend="auto",
        completed_mask_segmenter=BrokenSam3Segmenter(),
        completed_mask_prompt="chair . foreground object .",
    )

    assert (object_dir / COMPLETED_MASK_NAME).is_file()
    prepared = prepare_reconstruction_input(object_dir / "completed_crop.png", object_dir)
    assert prepared.mask_source == "completed_mask"


def test_cleanup_removes_generated_reconstruction_outputs_only(tmp_path: Path) -> None:
    objects_dir = tmp_path / "objects"
    object_dir = objects_dir / "01_chair"
    object_dir.mkdir(parents=True)
    keep_names = ("completed_crop.png", "masked_crop.png", "metadata.json")
    remove_names = (
        COMPLETED_MASK_NAME,
        COMPLETED_MASK_METADATA_NAME,
        "artifacts/reconstruction/hunyuan3d_input.png",
        "artifacts/reconstruction/hunyuan3d_mask.png",
        "hunyuan3d_mesh.obj",
        "hunyuan3d_mesh.glb",
        "hunyuan3d_metadata.json",
        "hunyuan3d_textured.obj",
        "hunyuan3d_textured.glb",
        "hunyuan3d_textured.mtl",
        "hunyuan3d_textured.jpg",
        "hunyuan3d_textured_metallic.jpg",
        "hunyuan3d_textured_roughness.jpg",
        "artifacts/reconstruction/white_mesh_remesh.obj",
    )
    for name in (*keep_names, *remove_names):
        path = object_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    (objects_dir / "hunyuan3d_manifest.json").write_text("{}", encoding="utf-8")

    removed = clean_reconstruction_outputs(objects_dir, [object_dir], backend="hunyuan3d")

    assert removed == len(remove_names) + 1
    for name in keep_names:
        assert (object_dir / name).is_file()
    for name in remove_names:
        assert not (object_dir / name).exists()
    assert not (objects_dir / "hunyuan3d_manifest.json").exists()


def test_hunyuan_paint_cache_env_overrides_inherited_sam3_cache() -> None:
    env = {
        "HF_HOME": "Models/OpenVocabulary/SAM3/hf",
        "HF_MODULES_CACHE": "Models/OpenVocabulary/SAM3/hf/modules",
    }

    apply_hunyuan3d_hf_cache_env(env)

    assert env["HF_HUB_CACHE"].endswith("Models/Mesh/Hunyuan3D/hf-cache")
    assert env["HF_MODULES_CACHE"].endswith("Models/Mesh/Hunyuan3D/diffusers-modules")
    assert env["HF_HUB_OFFLINE"] == "1"
    assert "OpenVocabulary/SAM3" not in env["HF_HUB_CACHE"]
    assert "OpenVocabulary/SAM3" not in env["HF_MODULES_CACHE"]
    assert "TRANSFORMERS_CACHE" not in env
    assert "DIFFUSERS_CACHE" not in env
    assert Path(env["HF_HUB_CACHE"]).is_absolute()
    assert Path(env["HF_MODULES_CACHE"]).is_absolute()


def test_hunyuan_texture_options_reject_unsupported_quality_values() -> None:
    validate_hunyuan_texture_options(768, 9)

    try:
        validate_hunyuan_texture_options(640, 9)
    except ValueError as exc:
        assert "512 or 768" in str(exc)
    else:
        raise AssertionError("Expected unsupported texture resolution to fail.")

    try:
        validate_hunyuan_texture_options(768, 5)
    except ValueError as exc:
        assert "between 6 and 12" in str(exc)
    else:
        raise AssertionError("Expected unsupported texture view count to fail.")


def test_count_obj_faces_counts_only_faces(tmp_path: Path) -> None:
    obj_path = tmp_path / "mesh.obj"
    obj_path.write_text(
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "f 1 2 3\n"
        "vn 0 0 1\n"
        "f 1 3 2\n",
        encoding="utf-8",
    )

    assert count_obj_faces(obj_path) == 2
    assert count_obj_faces(tmp_path / "missing.obj") is None


def test_paint_reference_uses_mask_crop_without_bria_model(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    mask_path = tmp_path / "mask.png"
    output_path = tmp_path / "paint_input.png"
    image = Image.new("RGB", (100, 100), (245, 243, 239))
    for x in range(35, 65):
        for y in range(20, 80):
            image.putpixel((x, y), (120, 80, 50))
    image.save(image_path)
    mask = Image.new("L", (100, 100), 0)
    for x in range(35, 65):
        for y in range(20, 80):
            mask.putpixel((x, y), 255)
    mask.save(mask_path)

    paint_path, source = prepare_paint_reference_image(
        image_path,
        mask_path,
        output_path,
        matte_backend="auto",
        matte_model_dir=tmp_path / "missing-rmbg",
    )

    assert paint_path == output_path
    assert source == "hunyuan3d_mask"
    painted = Image.open(output_path).convert("RGBA")
    assert painted.size == (1024, 1024)
    assert painted.getchannel("A").getbbox() is not None


def test_texture_reference_mode_original_uses_hunyuan_input(tmp_path: Path) -> None:
    object_dir = tmp_path / "01_chair"
    object_dir.mkdir()
    Image.new("RGB", (20, 20), (250, 250, 250)).save(object_dir / "hunyuan3d_input.png")
    Image.new("L", (20, 20), 255).save(object_dir / "hunyuan3d_mask.png")
    (object_dir / "hunyuan3d_mesh.obj").write_text("v 0 0 0\nf 1 1 1\n", encoding="utf-8")
    pipeline = FakePaintPipeline()
    record = {"status": "ok"}

    texture_object_dir(
        object_dir,
        record,
        paint_pipeline=pipeline,
        use_remesh=True,
        reference_mode="original",
    )

    assert pipeline.calls[0]["image_path"].endswith("hunyuan3d_input.png")
    assert pipeline.calls[0]["save_glb"] is True
    assert record["texture_status"] == "ok"
    assert record["textured_glb"] == "hunyuan3d_textured.glb"
    assert record["paint_matte_source"] == "original_input"
    assert record["texture_reference_mode"] == "original"


def test_texture_reference_mode_masked_crop_writes_paint_input(tmp_path: Path) -> None:
    object_dir = tmp_path / "01_chair"
    object_dir.mkdir()
    Image.new("RGB", (20, 20), (250, 250, 250)).save(object_dir / "hunyuan3d_input.png")
    mask = Image.new("L", (20, 20), 0)
    for x in range(5, 15):
        for y in range(5, 15):
            mask.putpixel((x, y), 255)
    mask.save(object_dir / "hunyuan3d_mask.png")
    (object_dir / "hunyuan3d_mesh.obj").write_text("v 0 0 0\nf 1 1 1\n", encoding="utf-8")
    pipeline = FakePaintPipeline()
    record = {"status": "ok"}

    texture_object_dir(
        object_dir,
        record,
        paint_pipeline=pipeline,
        use_remesh=True,
        reference_mode="masked-crop",
        matte_backend="mask",
    )

    assert pipeline.calls[0]["image_path"].endswith("hunyuan3d_paint_input.png")
    assert (object_dir / "artifacts/reconstruction/hunyuan3d_paint_input.png").is_file()
    assert record["paint_matte_source"] == "hunyuan3d_mask"
    assert record["texture_reference_mode"] == "masked-crop"


def test_table_like_outputs_enable_support_sheet_cleanup(tmp_path: Path) -> None:
    table_dir = tmp_path / "03_round_table"
    table_dir.mkdir()
    assert should_remove_hunyuan_support_sheet(table_dir, {}) is True

    chair_dir = tmp_path / "01_chair"
    chair_dir.mkdir()
    (chair_dir / "metadata.json").write_text('{"detector_label": "chair"}', encoding="utf-8")
    assert should_remove_hunyuan_support_sheet(chair_dir, {}) is False

    metadata_table_dir = tmp_path / "object"
    metadata_table_dir.mkdir()
    (metadata_table_dir / "metadata.json").write_text('{"detector_label": "coffee table"}', encoding="utf-8")
    assert should_remove_hunyuan_support_sheet(metadata_table_dir, {}) is True


def test_parse_removed_faces_reads_blender_json_report() -> None:
    output = 'Blender 5.1\n{\n  "output": "clean.glb",\n  "removed_faces": 32435\n}\n'
    assert parse_removed_faces(output) == 32435
    assert parse_removed_faces("no json here") is None
