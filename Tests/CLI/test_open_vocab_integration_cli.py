from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "run.py"), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


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


def test_cli_check_open_vocab_integration_writes_ready_report(tmp_path: Path) -> None:
    sam3_repo, sam3_model = make_sam3_layout(tmp_path)
    gdino_repo, gdino_config, gdino_checkpoint = make_groundingdino_layout(tmp_path)
    output = tmp_path / "preflight.json"

    result = run_cli(
        "check-open-vocab-integration",
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
    )

    assert result.returncode == 0
    assert "open-vocab integration ready" in result.stdout
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["ready"] is True
    assert data["backend"] == "groundingdino-sam3"


def test_cli_check_open_vocab_integration_missing_paths_fails_after_report(tmp_path: Path) -> None:
    output = tmp_path / "preflight.json"

    result = run_cli(
        "check-open-vocab-integration",
        "--backend",
        "groundingdino-sam3",
        "--groundingdino-repo-dir",
        str(tmp_path / "missing-gdino"),
        "--groundingdino-config",
        str(tmp_path / "missing.py"),
        "--groundingdino-checkpoint",
        str(tmp_path / "missing.pth"),
        "--sam3-repo-dir",
        str(tmp_path / "missing-sam3"),
        "--sam3-model-dir",
        str(tmp_path / "missing-model"),
        "--output",
        str(output),
    )

    assert result.returncode == 2
    assert output.is_file()
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["ready"] is False
    assert "Open-vocabulary integration is not ready" in result.stderr
