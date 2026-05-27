from __future__ import annotations

import json
from pathlib import Path

from PrimitiveFitting.types import PrimitiveFitReport


def write_fit_report(report: PrimitiveFitReport, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
