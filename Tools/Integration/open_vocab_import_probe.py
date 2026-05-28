from __future__ import annotations

import argparse
import importlib
import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ImportCheck:
    name: str
    module: str
    repo_dir: Path
    required_symbols: tuple[str, ...]
    required_members: tuple[str, ...]
    ok: bool
    detail: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "module": self.module,
            "repo_dir": str(self.repo_dir),
            "required_symbols": list(self.required_symbols),
            "required_members": list(self.required_members),
            "ok": bool(self.ok),
            "detail": self.detail,
        }


def build_report(
    *,
    backend: str,
    groundingdino_repo_dir: str | Path | None,
    sam3_repo_dir: str | Path | None,
) -> dict:
    if backend not in {"sam3", "groundingdino-sam3"}:
        raise ValueError(f"Unsupported open-vocabulary backend: {backend}")

    checks: list[ImportCheck] = []
    checks.extend(sam3_import_checks(Path(sam3_repo_dir or "")))
    if backend == "groundingdino-sam3":
        checks.extend(groundingdino_import_checks(Path(groundingdino_repo_dir or "")))

    ready = all(check.ok for check in checks)
    return {
        "schema_version": SCHEMA_VERSION,
        "backend": backend,
        "ready": ready,
        "checks": [check.to_dict() for check in checks],
    }


def sam3_import_checks(repo_dir: Path) -> list[ImportCheck]:
    return [
        probe_import(
            name="sam3_model_builder",
            repo_dir=repo_dir,
            module_name="sam3.model_builder",
            required_symbols=("build_sam3_image_model",),
            required_members=(),
            clear_prefix="sam3",
        ),
        probe_import(
            name="sam3_image_processor",
            repo_dir=repo_dir,
            module_name="sam3.model.sam3_image_processor",
            required_symbols=("Sam3Processor",),
            required_members=("Sam3Processor.set_image", "Sam3Processor.set_text_prompt", "Sam3Processor.add_geometric_prompt"),
            clear_prefix="sam3",
        ),
    ]


def groundingdino_import_checks(repo_dir: Path) -> list[ImportCheck]:
    return [
        probe_import(
            name="groundingdino_inference_api",
            repo_dir=repo_dir,
            module_name="groundingdino.util.inference",
            required_symbols=("load_model", "load_image", "predict"),
            required_members=(),
            clear_prefix="groundingdino",
        )
    ]


def probe_import(
    *,
    name: str,
    repo_dir: Path,
    module_name: str,
    required_symbols: tuple[str, ...],
    required_members: tuple[str, ...],
    clear_prefix: str,
) -> ImportCheck:
    if not repo_dir.is_dir():
        return ImportCheck(
            name=name,
            module=module_name,
            repo_dir=repo_dir,
            required_symbols=required_symbols,
            required_members=required_members,
            ok=False,
            detail="missing repo directory",
        )
    with temp_sys_path(repo_dir):
        clear_module_tree(clear_prefix)
        importlib.invalidate_caches()
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            return ImportCheck(
                name=name,
                module=module_name,
                repo_dir=repo_dir,
                required_symbols=required_symbols,
                required_members=required_members,
                ok=False,
                detail=f"import failed: {type(exc).__name__}: {exc}",
            )
    missing = [symbol for symbol in required_symbols if not hasattr(module, symbol)]
    if missing:
        return ImportCheck(
            name=name,
            module=module_name,
            repo_dir=repo_dir,
            required_symbols=required_symbols,
            required_members=required_members,
            ok=False,
            detail=f"missing symbols: {', '.join(missing)}",
        )
    return ImportCheck(
        name=name,
        module=module_name,
        repo_dir=repo_dir,
        required_symbols=required_symbols,
        required_members=required_members,
        ok=True,
        detail="ok",
    )


def has_dotted_member(module, dotted_name: str) -> bool:
    current = module
    for part in dotted_name.split("."):
        if not hasattr(current, part):
            return False
        current = getattr(current, part)
    return True


@contextmanager
def temp_sys_path(path: Path):
    value = str(path.resolve())
    original = list(sys.path)
    if value not in sys.path:
        sys.path.insert(0, value)
    try:
        yield
    finally:
        sys.path[:] = original


def clear_module_tree(prefix: str) -> None:
    for name in list(sys.modules):
        if name == prefix or name.startswith(f"{prefix}."):
            del sys.modules[name]


def write_report(report: dict, output_path: str | Path | None) -> None:
    if output_path is None:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def print_summary(report: dict) -> None:
    status = "ready" if report["ready"] else "not_ready"
    print(f"open-vocab imports {status}: {report['backend']}")
    for check in report["checks"]:
        mark = "ok" if check["ok"] else "failed"
        print(f"[{mark}] {check['name']}: {check['detail']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe local GroundingDINO/SAM3 imports without loading checkpoints or running inference.")
    parser.add_argument("--backend", choices=("sam3", "groundingdino-sam3"), default="groundingdino-sam3")
    parser.add_argument("--groundingdino-repo-dir", default="Models/OpenVocabulary/GroundingDINO/repo")
    parser.add_argument("--sam3-repo-dir", default="Models/OpenVocabulary/SAM3/repo")
    parser.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(
        backend=args.backend,
        groundingdino_repo_dir=args.groundingdino_repo_dir,
        sam3_repo_dir=args.sam3_repo_dir,
    )
    write_report(report, args.output)
    print_summary(report)
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        from Runtime.guided_cli import guided_tool_main

        raise SystemExit(guided_tool_main(Path(__file__), 'Probe GroundingDINO/SAM3 imports.', [], main))
    raise SystemExit(main())
