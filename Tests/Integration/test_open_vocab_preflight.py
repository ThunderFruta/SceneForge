from __future__ import annotations

import json
from pathlib import Path

from Tools.Integration.open_vocab_preflight import build_report, main


def make_sam3_layout(root: Path) -> tuple[Path, Path]:
    repo = root / "SAM3" / "repo"
    model = root / "SAM3" / "hf"
    (repo / "sam3" / "model").mkdir(parents=True)
    model.mkdir(parents=True)
    (repo / "sam3" / "model_builder.py").write_text("# fake", encoding="utf-8")
    (repo / "sam3" / "model" / "sam3_image_processor.py").write_text("# fake", encoding="utf-8")
    return repo, model


def make_groundingdino_layout(root: Path) -> tuple[Path, Path, Path]:
    repo = root / "GroundingDINO" / "repo"
    weights = root / "GroundingDINO" / "weights"
    (repo / "groundingdino" / "util").mkdir(parents=True)
    weights.mkdir(parents=True)
    config = repo / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
    config.parent.mkdir(parents=True)
    checkpoint = weights / "groundingdino_swint_ogc.pth"
    (repo / "groundingdino" / "util" / "inference.py").write_text("# fake", encoding="utf-8")
    config.write_text("# fake", encoding="utf-8")
    checkpoint.write_bytes(b"fake")
    return repo, config, checkpoint


def test_sam3_preflight_reports_ready_for_expected_layout(tmp_path: Path) -> None:
    sam3_repo, sam3_model = make_sam3_layout(tmp_path)

    report = build_report(
        backend="sam3",
        groundingdino_repo_dir=None,
        groundingdino_config=None,
        groundingdino_checkpoint=None,
        sam3_repo_dir=sam3_repo,
        sam3_model_dir=sam3_model,
    )

    assert report["ready"] is True
    assert report["backend"] == "sam3"
    assert all(check["ok"] for check in report["checks"])
    assert report["next_command"][report["next_command"].index("--backend") + 1] == "sam3"


def test_groundingdino_sam3_preflight_reports_missing_paths(tmp_path: Path) -> None:
    report = build_report(
        backend="groundingdino-sam3",
        groundingdino_repo_dir=tmp_path / "missing-gdino",
        groundingdino_config=tmp_path / "missing.py",
        groundingdino_checkpoint=tmp_path / "missing.pth",
        sam3_repo_dir=tmp_path / "missing-sam3",
        sam3_model_dir=tmp_path / "missing-model",
    )

    assert report["ready"] is False
    names = {check["name"]: check for check in report["checks"]}
    assert names["groundingdino_inference_api"]["ok"] is False
    assert names["sam3_model_builder"]["ok"] is False


def test_groundingdino_sam3_preflight_cli_writes_report(tmp_path: Path) -> None:
    sam3_repo, sam3_model = make_sam3_layout(tmp_path)
    gdino_repo, gdino_config, gdino_checkpoint = make_groundingdino_layout(tmp_path)
    output = tmp_path / "preflight.json"

    code = main(
        [
            "--backend",
            "groundingdino-sam3",
            "--groundingdino-repo-dir",
            str(gdino_repo),
            "--groundingdino-config",
            str(gdino_config),
            "--groundingdino-checkpoint",
            str(gdino_checkpoint),
            "--sam3-repo-dir",
            str(sam3_repo),
            "--sam3-model-dir",
            str(sam3_model),
            "--output",
            str(output),
        ]
    )

    assert code == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["ready"] is True
    assert data["backend"] == "groundingdino-sam3"
