from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from Tools.Integration.open_vocab_import_probe import build_report as build_import_probe_report
from Tools.Integration.open_vocab_preflight import DEFAULT_TEXT_PROMPT, build_report as build_preflight_report
from Tools.Integration.open_vocab_setup import OpenVocabLayout, SMOKE_IMAGE_PATH, SMOKE_PROMPT_PATH, smoke_test_command


SCHEMA_VERSION = 1


def build_report(
    *,
    root_dir: str | Path,
    backend: str = "groundingdino-sam3",
    text_prompt: str = DEFAULT_TEXT_PROMPT,
    run_import_probe: bool = True,
) -> dict:
    layout = OpenVocabLayout(Path(root_dir))
    manifest_exists = layout.manifest_path.is_file()
    setup_script_exists = layout.setup_script.is_file()
    smoke_image_exists = Path(SMOKE_IMAGE_PATH).is_file()
    smoke_prompt_exists = Path(SMOKE_PROMPT_PATH).is_file()
    preflight = build_preflight_report(
        backend=backend,
        groundingdino_repo_dir=layout.groundingdino_repo_dir,
        groundingdino_config=layout.groundingdino_config,
        groundingdino_checkpoint=layout.groundingdino_checkpoint,
        sam3_repo_dir=layout.sam3_repo_dir,
        sam3_model_dir=layout.sam3_model_dir,
        text_prompt=text_prompt,
    )
    import_probe = None
    if run_import_probe:
        import_probe = build_import_probe_report(
            backend=backend,
            groundingdino_repo_dir=layout.groundingdino_repo_dir,
            sam3_repo_dir=layout.sam3_repo_dir,
        )
    sam3_access = build_sam3_access_report(layout) if backend in {"sam3", "groundingdino-sam3"} else None
    ready_for_import_probe = bool(preflight["ready"])
    imports_ready = bool(import_probe is None or import_probe["ready"])
    sam3_access_ready = bool(sam3_access is None or sam3_access["ready"])
    ready_for_smoke_test = bool(preflight["ready"] and imports_ready and sam3_access_ready)
    return {
        "schema_version": SCHEMA_VERSION,
        "backend": backend,
        "root_dir": str(layout.root_dir),
        "status": readiness_status(
            manifest_exists=manifest_exists,
            setup_script_exists=setup_script_exists,
            preflight_ready=bool(preflight["ready"]),
            import_probe_ready=None if import_probe is None else bool(import_probe["ready"]),
            sam3_access_ready=None if sam3_access is None else bool(sam3_access["ready"]),
        ),
        "ready_for_import_probe": ready_for_import_probe,
        "ready_for_smoke_test": ready_for_smoke_test,
        "manifest_exists": manifest_exists,
        "setup_script_exists": setup_script_exists,
        "smoke_image_exists": smoke_image_exists,
        "smoke_prompt_exists": smoke_prompt_exists,
        "paths": {
            "manifest": str(layout.manifest_path),
            "setup_script": str(layout.setup_script),
            "smoke_image_path": SMOKE_IMAGE_PATH,
            "smoke_prompt_path": SMOKE_PROMPT_PATH,
            "groundingdino_repo_dir": str(layout.groundingdino_repo_dir),
            "groundingdino_config": str(layout.groundingdino_config),
            "groundingdino_checkpoint": str(layout.groundingdino_checkpoint),
            "sam3_repo_dir": str(layout.sam3_repo_dir),
            "sam3_model_dir": str(layout.sam3_model_dir),
        },
        "preflight": preflight,
        "import_probe": import_probe,
        "sam3_access": sam3_access,
        "first_smoke_test_command": smoke_test_command(layout),
        "next_steps": next_steps(
            manifest_exists=manifest_exists,
            setup_script_exists=setup_script_exists,
            preflight_ready=bool(preflight["ready"]),
            import_probe_ready=None if import_probe is None else bool(import_probe["ready"]),
            sam3_access_ready=None if sam3_access is None else bool(sam3_access["ready"]),
        ),
    }


def build_sam3_access_report(layout: OpenVocabLayout) -> dict:
    token_names = ["HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"]
    token_env_present = any(os.environ.get(name) for name in token_names)
    token_cache_present = huggingface_token_cache_present()
    cache_config_exists = any(layout.sam3_model_dir.rglob("config.json")) if layout.sam3_model_dir.exists() else False
    ready = bool(token_env_present or token_cache_present or cache_config_exists)
    if token_env_present:
        status = "token_env_present"
    elif token_cache_present:
        status = "token_cache_present"
    elif cache_config_exists:
        status = "cached_config_present"
    else:
        status = "auth_or_cache_missing"
    return {
        "ready": ready,
        "status": status,
        "repo_id": "facebook/sam3",
        "model_dir": str(layout.sam3_model_dir),
        "token_env_names": token_names,
        "token_env_present": token_env_present,
        "token_cache_present": token_cache_present,
        "cache_config_exists": cache_config_exists,
        "detail": "SAM3 is gated on Hugging Face; authenticate or pre-populate the local cache before smoke tests.",
    }


def huggingface_token_cache_present() -> bool:
    if os.environ.get("SCENEFORGE_DISABLE_HF_TOKEN_CACHE") == "1":
        return False
    try:
        from huggingface_hub import get_token

        return bool(get_token())
    except Exception:
        pass
    try:
        from huggingface_hub import HfFolder

        return bool(HfFolder.get_token())
    except Exception:
        return False


def readiness_status(
    *,
    manifest_exists: bool,
    setup_script_exists: bool,
    preflight_ready: bool,
    import_probe_ready: bool | None,
    sam3_access_ready: bool | None,
) -> str:
    if preflight_ready and (import_probe_ready is True or import_probe_ready is None) and sam3_access_ready is False:
        return "sam3_auth_required"
    if preflight_ready and (import_probe_ready is True or import_probe_ready is None):
        return "ready_for_smoke_test"
    if preflight_ready and import_probe_ready is False:
        return "imports_not_ready"
    if manifest_exists and setup_script_exists:
        return "layout_prepared_sources_missing"
    return "layout_not_prepared"


def next_steps(
    *,
    manifest_exists: bool,
    setup_script_exists: bool,
    preflight_ready: bool,
    import_probe_ready: bool | None,
    sam3_access_ready: bool | None,
) -> list[str]:
    if preflight_ready and (import_probe_ready is True or import_probe_ready is None) and sam3_access_ready is False:
        return ["Authenticate for gated SAM3 access with hf auth login or set HF_TOKEN/HUGGINGFACE_HUB_TOKEN, then rerun audit-open-vocab-readiness."]
    if preflight_ready and (import_probe_ready is True or import_probe_ready is None):
        return ["Run the first detect-shapes smoke test with --backend groundingdino-sam3."]
    if preflight_ready and import_probe_ready is False:
        return ["Fix Python environment/import errors from import_probe.checks, then rerun probe-open-vocab-imports."]
    if manifest_exists and setup_script_exists:
        return ["Review and run setup_open_vocab_sources.sh, handle Hugging Face auth for SAM3, then rerun check-open-vocab-integration."]
    return ["Run prepare-open-vocab-layout --root Models/OpenVocabulary, then review the generated setup script."]


def write_report(report: dict, output_path: str | Path | None) -> None:
    if output_path is None:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def print_summary(report: dict) -> None:
    print(f"open-vocab readiness: {report['status']}")
    print(f"root: {report['root_dir']}")
    print(f"path preflight: {'ready' if report['preflight']['ready'] else 'not_ready'}")
    import_probe = report.get("import_probe")
    if import_probe is not None:
        print(f"import probe: {'ready' if import_probe['ready'] else 'not_ready'}")
    sam3_access = report.get("sam3_access")
    if sam3_access is not None:
        print(f"sam3 access: {'ready' if sam3_access['ready'] else 'not_ready'}")
    for step in report["next_steps"]:
        print(f"next: {step}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit local readiness for the first GroundingDINO/SAM3 integration smoke test.")
    parser.add_argument("--root", default="Models/OpenVocabulary")
    parser.add_argument("--backend", choices=("sam3", "groundingdino-sam3"), default="groundingdino-sam3")
    parser.add_argument("--text-prompt", default=DEFAULT_TEXT_PROMPT)
    parser.add_argument("--skip-import-probe", action="store_true")
    parser.add_argument("--output", default="Output/Latest/open_vocab_readiness.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(
        root_dir=args.root,
        backend=args.backend,
        text_prompt=args.text_prompt,
        run_import_probe=not args.skip_import_probe,
    )
    write_report(report, args.output)
    print_summary(report)
    return 0 if report["ready_for_smoke_test"] else 2


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        from Runtime.guided_cli import guided_tool_main

        raise SystemExit(guided_tool_main(Path(__file__), 'Audit DINO/SAM readiness.', [], main))
    raise SystemExit(main())
