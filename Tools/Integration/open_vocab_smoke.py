from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Callable, Sequence

from Tools.Integration.open_vocab_preflight import DEFAULT_TEXT_PROMPT
from Tools.Integration.open_vocab_readiness import build_report as build_readiness_report
from Tools.Integration.open_vocab_setup import OpenVocabLayout, SMOKE_IMAGE_PATH, smoke_test_command


SCHEMA_VERSION = 1
CompletedProcessFactory = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def build_command(*, root_dir: str | Path, backend: str = "groundingdino-sam3", text_prompt: str = DEFAULT_TEXT_PROMPT) -> list[str]:
    layout = OpenVocabLayout(Path(root_dir))
    command = smoke_test_command(layout)
    if backend != "groundingdino-sam3":
        command[command.index("--backend") + 1] = backend
    if text_prompt != DEFAULT_TEXT_PROMPT:
        command[command.index("--text-prompt") + 1] = text_prompt
    return command


def run_smoke_test(
    *,
    root_dir: str | Path,
    backend: str = "groundingdino-sam3",
    text_prompt: str = DEFAULT_TEXT_PROMPT,
    output: str | Path = "Output/Latest/open_vocab_smoke.json",
    runner: CompletedProcessFactory | None = None,
) -> dict:
    readiness = build_readiness_report(root_dir=root_dir, backend=backend, text_prompt=text_prompt, run_import_probe=True)
    command = build_command(root_dir=root_dir, backend=backend, text_prompt=text_prompt)
    command[0] = sys.executable
    report = {
        "schema_version": SCHEMA_VERSION,
        "backend": backend,
        "root_dir": str(root_dir),
        "ready_for_smoke_test": bool(readiness["ready_for_smoke_test"]),
        "smoke_image_path": SMOKE_IMAGE_PATH,
        "command": command,
        "readiness": readiness,
        "status": "not_ready",
        "returncode": None,
        "stdout_tail": "",
        "stderr_tail": "",
    }
    if not readiness["ready_for_smoke_test"]:
        write_report(report, output)
        return report

    process_runner = runner or default_runner
    result = process_runner(command)
    report.update(
        {
            "status": "passed" if result.returncode == 0 else "failed",
            "returncode": int(result.returncode),
            "stdout_tail": tail(result.stdout),
            "stderr_tail": tail(result.stderr),
        }
    )
    write_report(report, output)
    return report


def default_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )


def tail(value: str | None, max_chars: int = 4000) -> str:
    text = value or ""
    return text[-max_chars:]


def write_report(report: dict, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def print_summary(report: dict) -> None:
    print(f"open-vocab smoke: {report['status']}")
    print(f"image: {report['smoke_image_path']}")
    if report["returncode"] is not None:
        print(f"returncode: {report['returncode']}")
    if report["status"] == "not_ready":
        for step in report["readiness"].get("next_steps", []):
            print(f"next: {step}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the first guarded GroundingDINO/SAM3 detect-shapes smoke test.")
    parser.add_argument("--root", default="Models/OpenVocabulary")
    parser.add_argument("--backend", choices=("sam3", "groundingdino-sam3"), default="groundingdino-sam3")
    parser.add_argument("--text-prompt", default=DEFAULT_TEXT_PROMPT)
    parser.add_argument("--output", default="Output/Latest/open_vocab_smoke.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_smoke_test(
        root_dir=args.root,
        backend=args.backend,
        text_prompt=args.text_prompt,
        output=args.output,
    )
    print_summary(report)
    if report["status"] == "passed":
        return 0
    return 2


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        from Runtime.guided_cli import guided_tool_main

        raise SystemExit(guided_tool_main(Path(__file__), 'Run the guarded DINO/SAM smoke test.', [], main))
    raise SystemExit(main())
