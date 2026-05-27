"""Evaluate a YOLO RGBD checkpoint across training datasets and synthetic blends.

This is an integration test runner, not a fast unit test. It runs:

1. `run.py eval-rgbd-yolo --split train` for every discovered `data_rgbd.yaml`.
2. `test_blends.py` for synthetic sample blend folders.

Example:
    .venv/bin/python Tests/Training/run_yolo_training_data_test.py \
      --weights Models/YOLO/sceneforge-yolo26l-rgbd-plane-context-easymild.pt \
      --device 0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = ROOT / "Models" / "YOLO" / "sceneforge-yolo26l-rgbd-plane-context-easymild-best.pt"
DEFAULT_OUTPUT_ROOT = ROOT / "Output" / "YoloTests"
DEFAULT_BLEND_ROOTS = (
    ROOT / "Assets" / "Samples" / "SyntheticNoPlane",
    ROOT / "Assets" / "Samples" / "SyntheticPlaneContext",
)


def main() -> int:
    args = parse_args()
    weights = resolve_path(args.weights)
    output_root = resolve_path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_yamls = discover_dataset_yamls(args.dataset_roots)
    blend_roots = discover_blend_roots(args.blend_roots)

    report: dict[str, Any] = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "weights": str(weights),
        "device": args.device,
        "output_root": str(output_root),
        "dataset_split": args.split,
        "dataset_results": [],
        "blend_results": [],
        "failures": [],
    }

    print(f"YOLO weights: {weights}")
    print(f"Output root: {output_root}")
    print(f"Dataset YAMLs: {len(dataset_yamls)}")
    for data_yaml in dataset_yamls:
        result = run_dataset_eval(
            data_yaml=data_yaml,
            weights=weights,
            output_root=output_root / "datasets",
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )
        report["dataset_results"].append(result)
        if result["status"] != "complete":
            report["failures"].append(result)

    print(f"Synthetic blend roots: {len(blend_roots)}")
    for blend_root in blend_roots:
        result = run_blend_eval(
            blend_root=blend_root,
            weights=weights,
            output_root=output_root / "blends",
            device=args.device,
            detector_confidence=args.detector_confidence,
            blender=args.blender,
        )
        report["blend_results"].append(result)
        if result["status"] != "complete":
            report["failures"].append(result)

    apply_threshold_failures(report, args)
    write_report(output_root, report)
    print(f"Wrote {output_root / 'summary.json'}")
    print(f"Wrote {output_root / 'summary.md'}")
    return 1 if report["failures"] else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO RGBD validation over all training datasets and synthetic blend samples.",
    )
    parser.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHTS),
        help=f"YOLO RGBD checkpoint. Defaults to {DEFAULT_WEIGHTS}.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_ROOT / datetime.now(UTC).strftime("%Y%m%d_%H%M%S")),
        help="Output directory for all eval artifacts.",
    )
    parser.add_argument(
        "--dataset-root",
        action="append",
        dest="dataset_roots",
        default=[],
        help="Dataset root to scan for data_rgbd.yaml. May be repeated. Defaults to Datasets.",
    )
    parser.add_argument(
        "--blend-root",
        action="append",
        dest="blend_roots",
        default=[],
        help="Synthetic blend folder or .blend file. May be repeated. Defaults to synthetic sample folders.",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", default="8")
    parser.add_argument("--device", default="0")
    parser.add_argument("--detector-confidence", type=float, default=0.20)
    parser.add_argument("--blender", default="blender")
    parser.add_argument(
        "--min-box-map50",
        type=float,
        default=0.0,
        help="Fail if any dataset box mAP50 is below this value. Default only records metrics.",
    )
    parser.add_argument(
        "--min-mask-map50",
        type=float,
        default=0.0,
        help="Fail if any dataset mask mAP50 is below this value. Default only records metrics.",
    )
    parser.add_argument(
        "--min-blend-good-rate",
        type=float,
        default=0.0,
        help="Fail if any blend-set object good rate is below this value.",
    )
    return parser.parse_args()


def discover_dataset_yamls(dataset_roots: list[str]) -> list[Path]:
    roots = [resolve_path(value) for value in dataset_roots] if dataset_roots else [ROOT / "Datasets"]
    yamls: list[Path] = []
    for root in roots:
        if root.is_file() and root.name == "data_rgbd.yaml":
            yamls.append(root)
        elif root.is_dir():
            yamls.extend(sorted(root.rglob("data_rgbd.yaml")))
    return sorted(dict.fromkeys(path.resolve() for path in yamls))


def discover_blend_roots(blend_roots: list[str]) -> list[Path]:
    if blend_roots:
        candidates = [resolve_path(value) for value in blend_roots]
    else:
        candidates = [path for path in DEFAULT_BLEND_ROOTS if path.exists()]
        if not candidates:
            candidates = sorted((ROOT / "Assets" / "Samples").glob("Synthetic*"))
    return sorted(dict.fromkeys(path.resolve() for path in candidates if path.exists()))


def run_dataset_eval(
    data_yaml: Path,
    weights: Path,
    output_root: Path,
    split: str,
    imgsz: int,
    batch: str,
    device: str,
) -> dict[str, Any]:
    name = safe_name(data_yaml.parent.relative_to(ROOT) if data_yaml.is_relative_to(ROOT) else data_yaml.parent)
    output_dir = output_root / name
    command = [
        sys.executable,
        "run.py",
        "eval-rgbd-yolo",
        "--data",
        str(data_yaml),
        "--weights",
        str(weights),
        "--output",
        str(output_dir),
        "--split",
        split,
        "--imgsz",
        str(imgsz),
        "--batch",
        str(batch),
        "--device",
        str(device),
    ]
    print(f"[dataset] {data_yaml}")
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    (output_dir / "command.txt").parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    (output_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")

    result: dict[str, Any] = {
        "kind": "dataset",
        "name": name,
        "data_yaml": str(data_yaml),
        "output_dir": str(output_dir),
        "status": "complete" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
    }
    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        result.update(json.loads(summary_path.read_text(encoding="utf-8")))
    else:
        result["error"] = short_error(completed.stderr or completed.stdout)
    print_dataset_result(result)
    return result


def run_blend_eval(
    blend_root: Path,
    weights: Path,
    output_root: Path,
    device: str,
    detector_confidence: float,
    blender: str,
) -> dict[str, Any]:
    name = safe_name(blend_root.relative_to(ROOT) if blend_root.is_relative_to(ROOT) else blend_root)
    output_dir = output_root / name
    command = [
        sys.executable,
        "test_blends.py",
        str(blend_root),
        "--detector-backend",
        "rgbd-yolo",
        "--detector-weights",
        str(weights),
        "--device",
        str(device),
        "--detector-confidence",
        str(detector_confidence),
        "--blender",
        blender,
        "--no-interactive",
        "--fix-summary-only",
        "--output",
        str(output_dir),
    ]
    print(f"[blends] {blend_root}")
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    (output_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")

    result: dict[str, Any] = {
        "kind": "blends",
        "name": name,
        "blend_root": str(blend_root),
        "output_dir": str(output_dir),
        "status": "complete" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
    }
    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        result.update(
            {
                "blend_count": summary.get("blend_count", 0),
                "completed_count": summary.get("completed_count", 0),
                "failed_count": summary.get("failed_count", 0),
                "object_count": sum(int(row.get("object_count", 0) or 0) for row in summary.get("runs", [])),
                "good_object_count": sum(int(row.get("good_object_count", 0) or 0) for row in summary.get("runs", [])),
                "bad_object_count": sum(int(row.get("bad_object_count", 0) or 0) for row in summary.get("runs", [])),
                "shapes": summary.get("shapes", []),
                "foreground_detection_recall": summary.get("foreground_detection_recall", []),
            }
        )
    else:
        result["error"] = short_error(completed.stderr or completed.stdout)
    print_blend_result(result)
    return result


def apply_threshold_failures(report: dict[str, Any], args: argparse.Namespace) -> None:
    for result in report["dataset_results"]:
        if result["status"] != "complete":
            continue
        if float(result.get("box_map50", 0.0)) < args.min_box_map50:
            report["failures"].append({**result, "error": "box_map50_below_threshold"})
        if float(result.get("mask_map50", 0.0)) < args.min_mask_map50:
            report["failures"].append({**result, "error": "mask_map50_below_threshold"})
    for result in report["blend_results"]:
        if result["status"] != "complete":
            continue
        object_count = int(result.get("object_count", 0) or 0)
        good_count = int(result.get("good_object_count", 0) or 0)
        good_rate = good_count / object_count if object_count else 0.0
        result["good_rate"] = round(good_rate, 6)
        if good_rate < args.min_blend_good_rate:
            report["failures"].append({**result, "error": "blend_good_rate_below_threshold"})


def write_report(output_root: Path, report: dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# SceneForge YOLO Training Data Test",
        "",
        f"- Weights: `{report['weights']}`",
        f"- Device: `{report['device']}`",
        f"- Dataset split: `{report['dataset_split']}`",
        f"- Dataset evals: {len(report['dataset_results'])}",
        f"- Blend evals: {len(report['blend_results'])}",
        f"- Failures: {len(report['failures'])}",
        "",
        "## Dataset results",
        "",
        "| Dataset | Status | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 | Output |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in report["dataset_results"]:
        lines.append(
            "| {name} | {status} | {box50:.4f} | {box:.4f} | {mask50:.4f} | {mask:.4f} | `{out}` |".format(
                name=item["name"],
                status=item["status"],
                box50=float(item.get("box_map50", 0.0)),
                box=float(item.get("box_map50_95", 0.0)),
                mask50=float(item.get("mask_map50", 0.0)),
                mask=float(item.get("mask_map50_95", 0.0)),
                out=item["output_dir"],
            )
        )
    lines.extend(
        [
            "",
            "## Synthetic blend results",
            "",
            "| Blend set | Status | Blends | Objects | Good | Bad | Good rate | Output |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for item in report["blend_results"]:
        lines.append(
            "| {name} | {status} | {blends} | {objects} | {good} | {bad} | {rate:.3f} | `{out}` |".format(
                name=item["name"],
                status=item["status"],
                blends=int(item.get("blend_count", 0) or 0),
                objects=int(item.get("object_count", 0) or 0),
                good=int(item.get("good_object_count", 0) or 0),
                bad=int(item.get("bad_object_count", 0) or 0),
                rate=float(item.get("good_rate", 0.0)),
                out=item["output_dir"],
            )
        )
    if report["failures"]:
        lines.extend(["", "## Failures", ""])
        for item in report["failures"]:
            lines.append(f"- `{item.get('name', item.get('output_dir', 'unknown'))}`: {item.get('error', item.get('status'))}")
    (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_dataset_result(result: dict[str, Any]) -> None:
    if result["status"] != "complete":
        print(f"  failed: {result.get('error', 'unknown error')}")
        return
    print(
        "  "
        f"box mAP50={float(result.get('box_map50', 0.0)):.4f} "
        f"mask mAP50={float(result.get('mask_map50', 0.0)):.4f}"
    )


def print_blend_result(result: dict[str, Any]) -> None:
    if result["status"] != "complete":
        print(f"  failed: {result.get('error', 'unknown error')}")
        return
    print(
        "  "
        f"blends={int(result.get('blend_count', 0) or 0)} "
        f"objects={int(result.get('object_count', 0) or 0)} "
        f"good={int(result.get('good_object_count', 0) or 0)} "
        f"bad={int(result.get('bad_object_count', 0) or 0)}"
    )


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def safe_name(value: object) -> str:
    text = str(value).strip().replace("\\", "/")
    safe = "".join(char if char.isalnum() else "_" for char in text)
    return safe.strip("_") or "root"


def short_error(text: str, limit: int = 600) -> str:
    cleaned = " ".join((text or "").strip().split())
    return cleaned[:limit] if cleaned else "missing summary output"


if __name__ == "__main__":
    raise SystemExit(main())
