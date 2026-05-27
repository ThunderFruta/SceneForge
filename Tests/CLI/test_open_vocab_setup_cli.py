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


def test_cli_prepare_open_vocab_layout_writes_manifest_and_script(tmp_path: Path) -> None:
    root = tmp_path / "OpenVocabulary"

    result = run_cli("prepare-open-vocab-layout", "--root", str(root))

    assert result.returncode == 0
    assert "prepared open-vocab layout" in result.stdout
    manifest_path = root / "open_vocab_setup_manifest.json"
    assert manifest_path.is_file()
    assert (root / "setup_open_vocab_sources.sh").is_file()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["paths"]["sam3_model_dir"].endswith("SAM3/hf")
    assert data["smoke_test_command"][0:3] == ["python3", "run.py", "detect-shapes"]
    assert "Assets/Fixtures/OpenVocabulary/open_vocab_smoke_objects.png" in data["smoke_test_command"]
