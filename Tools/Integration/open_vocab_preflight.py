from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SCHEMA_VERSION = 1
DEFAULT_TEXT_PROMPT = "chair . table . box . sphere . cylinder . cone . plane . foreground object ."


@dataclass(frozen=True)
class PathCheck:
    name: str
    path: Path
    kind: str
    required: bool
    ok: bool
    detail: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": str(self.path),
            "kind": self.kind,
            "required": bool(self.required),
            "ok": bool(self.ok),
            "detail": self.detail,
        }


def build_report(
    *,
    backend: str,
    groundingdino_repo_dir: str | Path | None,
    groundingdino_config: str | Path | None,
    groundingdino_checkpoint: str | Path | None,
    sam3_repo_dir: str | Path | None,
    sam3_model_dir: str | Path | None,
    text_prompt: str = DEFAULT_TEXT_PROMPT,
) -> dict:
    checks: list[PathCheck] = []
    if backend not in {"sam3", "groundingdino-sam3"}:
        raise ValueError(f"Unsupported open-vocabulary backend: {backend}")

    checks.extend(sam3_checks(sam3_repo_dir=sam3_repo_dir, sam3_model_dir=sam3_model_dir))
    if backend == "groundingdino-sam3":
        checks.extend(
            groundingdino_checks(
                groundingdino_repo_dir=groundingdino_repo_dir,
                groundingdino_config=groundingdino_config,
                groundingdino_checkpoint=groundingdino_checkpoint,
            )
        )

    ready = all(check.ok for check in checks if check.required)
    return {
        "schema_version": SCHEMA_VERSION,
        "backend": backend,
        "ready": ready,
        "text_prompt": text_prompt,
        "checks": [check.to_dict() for check in checks],
        "next_command": build_next_command(
            backend=backend,
            groundingdino_repo_dir=groundingdino_repo_dir,
            groundingdino_config=groundingdino_config,
            groundingdino_checkpoint=groundingdino_checkpoint,
            sam3_repo_dir=sam3_repo_dir,
            sam3_model_dir=sam3_model_dir,
            text_prompt=text_prompt,
        ),
    }


def sam3_checks(*, sam3_repo_dir: str | Path | None, sam3_model_dir: str | Path | None) -> list[PathCheck]:
    repo = path_or_empty(sam3_repo_dir)
    model = path_or_empty(sam3_model_dir)
    return [
        check_dir("sam3_repo_dir", repo, required=True),
        check_file("sam3_model_builder", repo / "sam3" / "model_builder.py", required=True),
        check_file(
            "sam3_image_processor",
            repo / "sam3" / "model" / "sam3_image_processor.py",
            required=True,
        ),
        check_dir("sam3_model_dir", model, required=True),
    ]


def groundingdino_checks(
    *,
    groundingdino_repo_dir: str | Path | None,
    groundingdino_config: str | Path | None,
    groundingdino_checkpoint: str | Path | None,
) -> list[PathCheck]:
    repo = path_or_empty(groundingdino_repo_dir)
    return [
        check_dir("groundingdino_repo_dir", repo, required=True),
        check_file("groundingdino_inference_api", repo / "groundingdino" / "util" / "inference.py", required=True),
        check_file("groundingdino_config", path_or_empty(groundingdino_config), required=True),
        check_file("groundingdino_checkpoint", path_or_empty(groundingdino_checkpoint), required=True),
    ]


def check_dir(name: str, path: Path, *, required: bool) -> PathCheck:
    ok = path.is_dir()
    return PathCheck(
        name=name,
        path=path,
        kind="dir",
        required=required,
        ok=ok,
        detail="ok" if ok else "missing directory",
    )


def check_file(name: str, path: Path, *, required: bool) -> PathCheck:
    ok = path.is_file()
    return PathCheck(
        name=name,
        path=path,
        kind="file",
        required=required,
        ok=ok,
        detail="ok" if ok else "missing file",
    )


def path_or_empty(value: str | Path | None) -> Path:
    return Path(value) if value else Path("")


def build_next_command(
    *,
    backend: str,
    groundingdino_repo_dir: str | Path | None,
    groundingdino_config: str | Path | None,
    groundingdino_checkpoint: str | Path | None,
    sam3_repo_dir: str | Path | None,
    sam3_model_dir: str | Path | None,
    text_prompt: str,
) -> list[str]:
    command = [
        "python3",
        "run.py",
        "detect-shapes",
        "--backend",
        backend,
        "--image",
        "Input/Image/example.png",
        "--text-prompt",
        text_prompt,
        "--sam3-repo-dir",
        str(path_or_empty(sam3_repo_dir)),
        "--sam3-model-dir",
        str(path_or_empty(sam3_model_dir)),
        "--output",
        "Output/Latest/detect",
    ]
    if backend == "groundingdino-sam3":
        command[command.index("--sam3-repo-dir"):command.index("--output")] = [
            "--groundingdino-repo-dir",
            str(path_or_empty(groundingdino_repo_dir)),
            "--groundingdino-config",
            str(path_or_empty(groundingdino_config)),
            "--groundingdino-checkpoint",
            str(path_or_empty(groundingdino_checkpoint)),
            "--sam3-repo-dir",
            str(path_or_empty(sam3_repo_dir)),
            "--sam3-model-dir",
            str(path_or_empty(sam3_model_dir)),
        ]
    return command


def write_report(report: dict, output_path: str | Path | None) -> None:
    if output_path is None:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def print_summary(report: dict) -> None:
    status = "ready" if report["ready"] else "not_ready"
    print(f"open-vocab integration {status}: {report['backend']}")
    for check in report["checks"]:
        mark = "ok" if check["ok"] else "missing"
        print(f"[{mark}] {check['name']}: {check['path']}")
    if report["ready"]:
        print("next command:")
        print(shell_join(report["next_command"]))


def shell_join(parts: Iterable[str]) -> str:
    import shlex

    return " ".join(shlex.quote(str(part)) for part in parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preflight local GroundingDINO/SAM3 integration paths.")
    parser.add_argument("--backend", choices=("sam3", "groundingdino-sam3"), default="groundingdino-sam3")
    parser.add_argument("--groundingdino-repo-dir", default="Models/OpenVocabulary/GroundingDINO/repo")
    parser.add_argument(
        "--groundingdino-config",
        default="Models/OpenVocabulary/GroundingDINO/repo/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    )
    parser.add_argument(
        "--groundingdino-checkpoint",
        default="Models/OpenVocabulary/GroundingDINO/weights/groundingdino_swint_ogc.pth",
    )
    parser.add_argument("--sam3-repo-dir", default="Models/OpenVocabulary/SAM3/repo")
    parser.add_argument("--sam3-model-dir", default="Models/OpenVocabulary/SAM3/hf")
    parser.add_argument("--text-prompt", default=DEFAULT_TEXT_PROMPT)
    parser.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(
        backend=args.backend,
        groundingdino_repo_dir=args.groundingdino_repo_dir,
        groundingdino_config=args.groundingdino_config,
        groundingdino_checkpoint=args.groundingdino_checkpoint,
        sam3_repo_dir=args.sam3_repo_dir,
        sam3_model_dir=args.sam3_model_dir,
        text_prompt=args.text_prompt,
    )
    write_report(report, args.output)
    print_summary(report)
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
