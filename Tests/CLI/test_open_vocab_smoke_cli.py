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


def test_cli_run_open_vocab_smoke_writes_not_ready_report(tmp_path: Path) -> None:
    output = tmp_path / "smoke.json"

    result = run_cli(
        "run-open-vocab-smoke",
        "--root",
        str(tmp_path / "missing"),
        "--output",
        str(output),
    )

    assert result.returncode == 2
    assert output.is_file()
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["status"] == "not_ready"
    assert "Open-vocabulary smoke test did not pass" in result.stderr
