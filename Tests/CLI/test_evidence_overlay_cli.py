from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from Tests.OutputWriter.test_evidence_overlay import write_reports


ROOT = Path(__file__).resolve().parents[2]


def test_render_evidence_overlay_cli_writes_png(tmp_path: Path) -> None:
    image_path, detections_path, enrichment_path = write_reports(tmp_path)
    output_path = tmp_path / "combined.png"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "run.py"),
            "render-evidence-overlay",
            "--image",
            str(image_path),
            "--detections",
            str(detections_path),
            "--enrichment",
            str(enrichment_path),
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert output_path.is_file()
    assert f"Wrote {output_path}" in result.stdout

