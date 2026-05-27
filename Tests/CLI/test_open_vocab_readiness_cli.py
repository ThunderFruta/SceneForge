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


def test_cli_audit_open_vocab_readiness_writes_not_ready_report(tmp_path: Path) -> None:
    output = tmp_path / "readiness.json"

    result = run_cli(
        "audit-open-vocab-readiness",
        "--root",
        str(tmp_path / "OpenVocabulary"),
        "--output",
        str(output),
    )

    assert result.returncode == 2
    assert output.is_file()
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["status"] == "layout_not_prepared"
    assert data["ready_for_smoke_test"] is False
    assert "Open-vocabulary integration is not ready for smoke test" in result.stderr
