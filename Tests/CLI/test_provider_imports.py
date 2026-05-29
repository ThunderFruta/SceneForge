from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_run_py_does_not_import_real_provider_modules_on_startup() -> None:
    script = (
        "import sys; "
        "import run; "
        "assert 'EdgeDetection' not in sys.modules; "
        "assert 'MeshReconstruction' not in sys.modules; "
        "assert 'WireframeDetection' not in sys.modules; "
        "assert 'ObjectEnrichment' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
