from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path
from typing import Callable, Sequence


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def shell_join(parts: Sequence[str | Path]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def ask_text(label: str, default: str | Path | None = None, *, required: bool = False) -> str:
    suffix = f" [{default}]" if default not in (None, "") else ""
    while True:
        try:
            value = input(f"{label}{suffix}: ").strip()
        except EOFError:
            value = ""
        if not value and default not in (None, ""):
            return str(default)
        if value or not required:
            return value
        print("Required.")


def ask_choice(title: str, choices: Sequence[tuple[str, str]], *, default_index: int = 0) -> int:
    print(title)
    for index, (label, detail) in enumerate(choices, start=1):
        marker = " [default]" if index - 1 == default_index else ""
        print(f"  {index}. {label}{marker} - {detail}")
    while True:
        raw = ask_text("Choose", str(default_index + 1))
        try:
            choice = int(raw) - 1
        except ValueError:
            print("Enter a number.")
            continue
        if 0 <= choice < len(choices):
            return choice
        print("Choice out of range.")


def confirm(label: str, *, default: bool = True) -> bool:
    suffix = "Y/n" if default else "y/N"
    raw = ask_text(f"{label} ({suffix})", "" if default else "n").lower()
    if raw == "":
        return default
    return raw in {"y", "yes", "1", "true"}


def print_command(command: Sequence[str | Path]) -> None:
    print("Equivalent command:")
    print(shell_join(command))


def run_after_confirmation(command: Sequence[str | Path], runner: Callable[[list[str]], int] | None = None) -> int:
    print_command(command)
    if not confirm("Run now", default=True):
        return 0
    if runner is not None:
        return int(runner([str(part) for part in command]))
    return subprocess.run([str(part) for part in command], cwd=repo_root(), check=False).returncode


def guided_tool_main(
    script_path: str | Path,
    description: str,
    default_args: Sequence[str | Path],
    runner: Callable[[list[str]], int],
) -> int:
    command = [sys.executable, str(script_path), *[str(part) for part in default_args]]
    print(description)
    return run_after_confirmation(command, lambda _cmd: runner([str(part) for part in default_args]))


def guided_blender_tool_main(
    script_path: str | Path,
    description: str,
    script_args: Sequence[str | Path],
    *,
    blend_path: str | Path | None = None,
    blender: str = "blender",
) -> int:
    command: list[str | Path] = [blender, "--background"]
    if blend_path:
        command.append(str(blend_path))
    command.extend(["--python", str(script_path), "--", *[str(part) for part in script_args]])
    print(description)
    return run_after_confirmation(command)


def likely_open_vocab_ready(root: Path | None = None) -> bool:
    root = root or repo_root() / "Models" / "OpenVocabulary"
    return all(
        path.exists()
        for path in (
            root / "open_vocab_setup_manifest.json",
            root / "GroundingDINO" / "repo",
            root / "GroundingDINO" / "weights" / "groundingdino_swint_ogc.pth",
            root / "SAM3" / "repo",
            root / "SAM3" / "hf",
        )
    )


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def is_blend_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() == ".blend"


def is_image_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_SUFFIXES


def reconstruct_args_for_blend(blend: str | Path, output: str | Path) -> list[str]:
    return [
        "reconstruct-scene",
        "--reference-blend", str(blend),
        "--detector-backend", "groundingdino-sam3",
        "--open-vocab-root", "Models/OpenVocabulary",
        "--text-prompt-preset", "scene-primitives-v1",
        "--edge-backend", "simple",
        "--wireframe-backend", "none",
        "--mesh-backend", "none",
        "--output", str(output),
        "--device", "auto",
    ]


def guided_scene_main(execute: Callable[[list[str]], int]) -> int:
    root = repo_root()
    default_choice = 1 if likely_open_vocab_ready() else 2
    choices = [
        ("Detect objects from image", "DINO/SAM proposal masks to detections.json and overlay.png"),
        ("Reconstruct scene from .blend", "Render RGBD, detect with DINO/SAM, enrich, fit, and export fitted_scene.blend"),
        ("Turn .blend into PNG", "Render the active Blender camera to one PNG without depth or detection"),
        ("Check DINO/SAM readiness", "Run non-inference setup/import/auth audit"),
        ("Run DINO/SAM smoke test", "Run guarded smoke fixture through real DINO/SAM"),
        ("Inspect latest outputs", "Render preview views from Output/Latest/fitted_scene.blend"),
        ("Show command recipes", "Print common explicit commands and exit"),
    ]
    selected = ask_choice("SceneForge guided mode", choices, default_index=default_choice)
    if selected == 0:
        image = ask_text("Image path", root / "Assets" / "Fixtures" / "OpenVocabulary" / "open_vocab_smoke_objects.png", required=True)
        if is_blend_path(image):
            print("That path is a .blend file, so guided mode will run reconstruction instead of image detection.")
            output = ask_text("Output directory", "Output/Latest", required=True)
            args = reconstruct_args_for_blend(image, output)
            if Path(output).exists() and confirm(f"Use --force for existing output {output}", default=True):
                args.append("--force")
            return _run_scene_args(args, execute)
        if not is_image_path(image):
            print("Image detection needs an image file such as .png, .jpg, .jpeg, .webp, .bmp, .tif, or .tiff.")
            return 2
        output = ask_text("Output directory", "Output/Latest/detect", required=True)
        args = [
            "detect-shapes",
            "--backend", "groundingdino-sam3",
            "--image", image,
            "--open-vocab-root", "Models/OpenVocabulary",
            "--text-prompt-preset", "scene-primitives-v1",
            "--output", output,
            "--device", "auto",
        ]
        return _run_scene_args(args, execute)
    if selected == 1:
        blend = ask_text("Reference .blend", root / "Assets" / "Samples" / "shapes.blend", required=True)
        if not is_blend_path(blend):
            print("Reconstruction needs a .blend file. Use option 1 for image files.")
            return 2
        output = ask_text("Output directory", "Output/Latest", required=True)
        args = reconstruct_args_for_blend(blend, output)
        if Path(output).exists() and confirm(f"Use --force for existing output {output}", default=True):
            args.append("--force")
        return _run_scene_args(args, execute)
    if selected == 2:
        blend = ask_text("Reference .blend", root / "Assets" / "Samples" / "shapes.blend", required=True)
        if not is_blend_path(blend):
            print("PNG rendering needs a .blend file.")
            return 2
        output = ask_text("PNG output", "Output/Latest/render/image.png", required=True)
        width = ask_text("Width", "1280", required=True)
        height = ask_text("Height", "720", required=True)
        exposure = ask_text("Exposure", "auto", required=True)
        return _run_scene_args([
            "render-blend-png",
            "--reference-blend", blend,
            "--output", output,
            "--width", width,
            "--height", height,
            "--render-samples", "8",
            "--exposure", exposure,
        ], execute)
    if selected == 3:
        root_dir = ask_text("Open vocabulary root", "Models/OpenVocabulary", required=True)
        return _run_scene_args(["audit-open-vocab-readiness", "--root", root_dir, "--backend", "groundingdino-sam3"], execute)
    if selected == 4:
        root_dir = ask_text("Open vocabulary root", "Models/OpenVocabulary", required=True)
        return _run_scene_args(["run-open-vocab-smoke", "--root", root_dir, "--backend", "groundingdino-sam3"], execute)
    if selected == 5:
        blend = ask_text("Blend to inspect", "Output/Latest/fitted_scene.blend", required=True)
        return run_after_confirmation([sys.executable, "Tools/Scripts/view_blend.py", "--blend", blend, "--views", "front,iso", "--no-gltf"])
    print_recipes()
    return 0


def _run_scene_args(args: list[str], execute: Callable[[list[str]], int]) -> int:
    print_command([sys.executable, "run.py", *args])
    if not confirm("Run now", default=True):
        return 0
    return int(execute(args))


def print_recipes() -> None:
    recipes = [
        [sys.executable, "run.py", "audit-open-vocab-readiness", "--root", "Models/OpenVocabulary", "--backend", "groundingdino-sam3"],
        [sys.executable, "run.py", "run-open-vocab-smoke", "--root", "Models/OpenVocabulary", "--backend", "groundingdino-sam3"],
        [sys.executable, "run.py", "detect-shapes", "--backend", "groundingdino-sam3", "--image", "path/to/image.png", "--open-vocab-root", "Models/OpenVocabulary", "--text-prompt-preset", "scene-primitives-v1", "--output", "Output/Latest/detect", "--device", "auto"],
        [sys.executable, "run.py", "render-blend-png", "--reference-blend", "path/to/file.blend", "--output", "Output/Latest/render/image.png", "--width", "1280", "--height", "720", "--exposure", "auto"],
        [sys.executable, "run.py", "reconstruct-scene", "--reference-blend", "path/to/file.blend", "--detector-backend", "groundingdino-sam3", "--open-vocab-root", "Models/OpenVocabulary", "--text-prompt-preset", "scene-primitives-v1", "--edge-backend", "simple", "--wireframe-backend", "none", "--mesh-backend", "none", "--output", "Output/Latest", "--device", "auto", "--force"],
    ]
    for command in recipes:
        print(shell_join(command))
