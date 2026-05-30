from __future__ import annotations

import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ObjectReconstruction.triposr_objects import prepare_reconstruction_input, select_source_image


SCHEMA_VERSION = 1
ARTIFACTS_DIR = Path("artifacts") / "reconstruction"
INPUT_NAME = str(ARTIFACTS_DIR / "sam3d_objects_input.png")
MASK_NAME = str(ARTIFACTS_DIR / "sam3d_objects_mask.png")
OUTPUT_MESH_NAME = "sam3d_objects.glb"
METADATA_NAME = "sam3d_objects_metadata.json"
MANIFEST_NAME = "sam3d_objects_manifest.json"


def run_sam3d_objects_reconstruction(
    objects_dir: str | Path,
    *,
    repo_dir: str | Path | None = None,
    checkpoint: str | Path | None = None,
    command_template: str | None = None,
    model: str = "facebook/sam-3d-objects",
    device: str | None = "auto",
    source: str = "auto",
    max_objects: int = 0,
    completed_mask_backend: str = "auto",
) -> dict[str, Any]:
    root = Path(objects_dir)
    if not root.is_dir():
        return write_manifest(root, [], "missing_objects_dir", repo_dir=repo_dir, checkpoint=checkpoint, command_template=command_template, model=model, device=device, source=source, completed_mask_backend=completed_mask_backend)

    object_dirs = [path for path in sorted(root.iterdir()) if path.is_dir()]
    selected_dirs = object_dirs if max_objects <= 0 else object_dirs[:max_objects]
    if not selected_dirs:
        return write_manifest(root, [], "no_objects", repo_dir=repo_dir, checkpoint=checkpoint, command_template=command_template, model=model, device=device, source=source, completed_mask_backend=completed_mask_backend)

    records = [
        reconstruct_object_dir(
            object_dir,
            repo_dir=Path(repo_dir) if repo_dir else None,
            checkpoint=Path(checkpoint) if checkpoint else None,
            command_template=command_template,
            model=model,
            device=device,
            source=source,
            completed_mask_backend=completed_mask_backend,
            order_index=index,
        )
        for index, object_dir in enumerate(selected_dirs, start=1)
    ]
    return write_manifest(root, records, "complete", repo_dir=repo_dir, checkpoint=checkpoint, command_template=command_template, model=model, device=device, source=source, completed_mask_backend=completed_mask_backend)


def reconstruct_object_dir(
    object_dir: Path,
    *,
    repo_dir: Path | None,
    checkpoint: Path | None,
    command_template: str | None,
    model: str,
    device: str | None,
    source: str,
    completed_mask_backend: str,
    order_index: int,
) -> dict[str, Any]:
    source_path, source_kind = select_source_image(object_dir, source)
    if source_path is None:
        return write_object_metadata(object_dir, {"object_dir": str(object_dir), "status": "skipped", "reason": "missing_source_image", "backend": "sam3d-objects", "model": model, "order_index": order_index})

    input_path = object_dir / INPUT_NAME
    mask_path = object_dir / MASK_NAME
    output_mesh_path = object_dir / OUTPUT_MESH_NAME
    input_path.parent.mkdir(parents=True, exist_ok=True)

    prepared = prepare_reconstruction_input(source_path, object_dir, completed_mask_backend=completed_mask_backend)
    prepared.image.save(input_path)
    prepared.mask.save(mask_path)

    record = {
        "schema_version": SCHEMA_VERSION,
        "object_dir": str(object_dir),
        "status": "prepared",
        "reason": None,
        "backend": "sam3d-objects",
        "model": model,
        "device": device,
        "source": source_kind,
        "source_image": source_path.name,
        "mask_source": prepared.mask_source,
        "completed_mask": prepared.completed_mask_path,
        "sam3d_objects_input": INPUT_NAME,
        "sam3d_objects_mask": MASK_NAME,
        "mesh": OUTPUT_MESH_NAME if output_mesh_path.is_file() else None,
        "repo_dir": str(repo_dir) if repo_dir is not None else None,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "command_template": command_template,
        "order_index": order_index,
    }

    skip_reason = sam3d_configuration_skip_reason(repo_dir=repo_dir, checkpoint=checkpoint, command_template=command_template)
    if output_mesh_path.is_file():
        record["status"] = "ok"
        record["reason"] = "existing_output"
    elif skip_reason is not None:
        record["status"] = "skipped"
        record["reason"] = skip_reason
    else:
        command = build_command(command_template, object_dir=object_dir, input_path=input_path, mask_path=mask_path, output_mesh_path=output_mesh_path, repo_dir=repo_dir, checkpoint=checkpoint, device=device)
        record["command"] = command
        result = subprocess.run(command, cwd=repo_dir or object_dir, check=False, capture_output=True, text=True)
        record["returncode"] = int(result.returncode)
        record["stdout_tail"] = result.stdout[-2000:]
        record["stderr_tail"] = result.stderr[-2000:]
        if result.returncode == 0 and output_mesh_path.is_file():
            record["status"] = "ok"
            record["reason"] = None
            record["mesh"] = OUTPUT_MESH_NAME
        else:
            record["status"] = "failed"
            record["reason"] = "sam3d_objects_command_failed"

    return write_object_metadata(object_dir, record)


def sam3d_configuration_skip_reason(*, repo_dir: Path | None, checkpoint: Path | None, command_template: str | None) -> str | None:
    if repo_dir is not None and not repo_dir.is_dir():
        return "missing_sam3d_objects_repo_dir"
    if checkpoint is not None and not checkpoint.is_file():
        return "missing_sam3d_objects_checkpoint"
    if not command_template:
        return "sam3d_objects_command_required"
    return None


def build_command(
    command_template: str | None,
    *,
    object_dir: Path,
    input_path: Path,
    mask_path: Path,
    output_mesh_path: Path,
    repo_dir: Path | None,
    checkpoint: Path | None,
    device: str | None,
) -> list[str]:
    if not command_template:
        raise ValueError("command_template is required")
    values = {
        "object_dir": str(object_dir),
        "image": str(input_path),
        "mask": str(mask_path),
        "output": str(output_mesh_path),
        "repo_dir": str(repo_dir or ""),
        "checkpoint": str(checkpoint or ""),
        "device": str(device or "auto"),
    }
    return [piece.format(**values) for piece in shlex.split(command_template)]


def write_object_metadata(object_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
    (object_dir / METADATA_NAME).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def write_manifest(
    objects_dir: Path,
    records: list[dict[str, Any]],
    status: str,
    *,
    repo_dir: str | Path | None,
    checkpoint: str | Path | None,
    command_template: str | None,
    model: str,
    device: str | None,
    source: str,
    completed_mask_backend: str,
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "backend": "sam3d-objects",
        "model": model,
        "device": device,
        "source": source,
        "completed_mask_backend": completed_mask_backend,
        "repo_dir": str(repo_dir) if repo_dir is not None else None,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "command_template": command_template,
        "object_count": len(records),
        "objects": records,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    objects_dir.mkdir(parents=True, exist_ok=True)
    (objects_dir / MANIFEST_NAME).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
