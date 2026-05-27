from __future__ import annotations

import json
from pathlib import Path

from ObjectEnrichment.types import ObjectEnrichmentReport


def write_enrichment_report(report: ObjectEnrichmentReport, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
