from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Callable, Sequence


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def shell_join(parts: Sequence[str | Path]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def normalize_prompt_value(value: str) -> str:
    if not value:
        return value
    try:
        parts = shlex.split(value)
    except ValueError:
        return value
    if len(parts) == 1:
        return parts[0]
    return value


def ask_text(label: str, default: str | Path | None = None, *, required: bool = False) -> str:
    suffix = f" [{default}]" if default not in (None, "") else ""
    while True:
        try:
            value = normalize_prompt_value(input(f"{label}{suffix}: ").strip())
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


def render_args_for_blend(blend: str | Path, output: str | Path) -> list[str]:
    output_path = Path(output)
    if output_path.suffix.lower() != ".png":
        output_path = output_path / "render" / "image.png"
    return [
        "render-blend-png",
        "--reference-blend", str(blend),
        "--output", str(output_path),
        "--width", "1280",
        "--height", "720",
        "--render-samples", "1024",
        "--exposure", "auto",
    ]


def detect_args_for_image(image: str | Path, output: str | Path) -> list[str]:
    args = [
        "detect-shapes",
        "--backend", "groundingdino-sam3",
        "--image", str(image),
        "--open-vocab-root", "Models/OpenVocabulary",
        "--text-prompt-preset", "scene-primitives-v1",
        "--output", str(output),
        "--device", "auto",
    ]
    backend, ram_args = open_vocab_backend_with_ram_fallback()
    args[args.index("--backend") + 1] = backend
    if backend == "ram-groundingdino-sam3":
        args = remove_option_with_value(args, "--text-prompt-preset")
    args.extend(ram_args)
    return args


def completion_args_if_available() -> list[str]:
    preferred_backend = os.environ.get("SCENEFORGE_COMPLETION_BACKEND", "openai-image").strip().lower()
    flux_model_dir = repo_root() / "Models" / "Completion" / "FluxFill"
    disabled_values = {"", "none", "0", "false", "off"}
    if preferred_backend in disabled_values:
        print("SceneForge guided completion is disabled by SCENEFORGE_COMPLETION_BACKEND.")
        return []
    if preferred_backend in {"openai", "openai-image", "gpt-5.5"}:
        max_objects = os.environ.get(
            "SCENEFORGE_OPENAI_COMPLETION_MAX_OBJECTS",
            os.environ.get("SCENEFORGE_COMPLETION_MAX_OBJECTS", "0"),
        )
        return [
            "--completion-backend", "openai-image",
            "--completion-model", os.environ.get("SCENEFORGE_OPENAI_COMPLETION_MODEL", "gpt-5.5"),
            "--completion-guidance-scale", "6.0",
            "--completion-steps", "28",
            "--completion-canvas-size", os.environ.get("SCENEFORGE_OPENAI_COMPLETION_CANVAS_SIZE", "1024"),
            "--completion-max-objects", max_objects,
        ]
    if preferred_backend not in {"flux", "flux-fill"}:
        print(f"Info: SCENEFORGE_COMPLETION_BACKEND={preferred_backend} is ignored; use openai-image or flux-fill.")
        return []
    if flux_model_dir.is_dir():
        return [
            "--completion-backend", "flux-fill",
            "--completion-model", "Models/Completion/FluxFill",
            "--completion-device", "auto",
            "--completion-quantization", "4bit",
            "--completion-guidance-scale", "6.0",
            "--completion-steps", "28",
        ]
    print(f"Info: FLUX completion skipped; expected model at {flux_model_dir}.")
    return []


def object_reconstruction_args_if_available() -> list[str]:
    preferred_backend = os.environ.get("SCENEFORGE_OBJECT_RECON_BACKEND", "hunyuan3d").strip().lower()
    disabled_values = {"", "none", "0", "false", "off"}
    if preferred_backend in disabled_values:
        print("SceneForge object reconstruction is disabled by SCENEFORGE_OBJECT_RECON_BACKEND.")
        return []
    if preferred_backend == "hunyuan3d":
        args = [
            "--backend", "hunyuan3d",
            "--model", os.environ.get("SCENEFORGE_HUNYUAN3D_MODEL", "tencent/Hunyuan3D-2.1"),
            "--device", os.environ.get("SCENEFORGE_OBJECT_RECON_DEVICE", os.environ.get("SCENEFORGE_HUNYUAN3D_DEVICE", "auto")),
            "--source", os.environ.get("SCENEFORGE_OBJECT_RECON_SOURCE", "completed"),
            "--max-objects", os.environ.get("SCENEFORGE_OBJECT_RECON_MAX_OBJECTS", "0"),
        ]
        if os.environ.get("SCENEFORGE_HUNYUAN3D_TEXTURE", "").strip().lower() in {"1", "true", "yes", "on"}:
            args.extend([
                "--with-texture",
                "--texture-resolution", os.environ.get("SCENEFORGE_HUNYUAN3D_TEXTURE_RESOLUTION", "512"),
                "--texture-views", os.environ.get("SCENEFORGE_HUNYUAN3D_TEXTURE_VIEWS", "6"),
            ])
        return args
    if preferred_backend != "triposr":
        print(f"Info: SCENEFORGE_OBJECT_RECON_BACKEND={preferred_backend} is ignored; use hunyuan3d or triposr.")
        return []
    return [
        "--backend", "triposr",
        "--model-dir", os.environ.get("SCENEFORGE_TRIPOSR_MODEL_DIR", "Models/Mesh/TripoSR"),
        "--device", os.environ.get("SCENEFORGE_OBJECT_RECON_DEVICE", os.environ.get("SCENEFORGE_TRIPOSR_DEVICE", "auto")),
        "--source", os.environ.get("SCENEFORGE_OBJECT_RECON_SOURCE", os.environ.get("SCENEFORGE_TRIPOSR_SOURCE", "completed")),
        "--max-objects", os.environ.get("SCENEFORGE_OBJECT_RECON_MAX_OBJECTS", os.environ.get("SCENEFORGE_TRIPOSR_MAX_OBJECTS", "0")),
    ]


def guided_scene_main(execute: Callable[[list[str]], int]) -> int:
    root = repo_root()
    default_choice = 1 if likely_open_vocab_ready() else 2
    choices = [
        ("Detect objects from image", "DINO/SAM proposal masks to detections.json and overlay.png"),
        ("Detect objects from image", "Use SAM3/GroundingDINO-SAM3 proposals for the new object lane"),
        ("Turn .blend into PNG", "Render the active Blender camera to one PNG without depth or detection"),
        ("Check DINO/SAM readiness", "Run non-inference setup/import/auth audit"),
        ("Run DINO/SAM smoke test", "Run guarded smoke fixture through real DINO/SAM"),
        ("Inspect a .blend file", "Render preview views from a selected Blender file"),
        ("Complete latest object crops", "Run OpenAI or FLUX completion over Output/Latest/objects"),
        ("Reconstruct latest object meshes", "Run Hunyuan3D/TripoSR over Output/Latest/objects crops"),
        ("Show command recipes", "Print common explicit commands and exit"),
    ]
    selected = ask_choice("SceneForge guided mode", choices, default_index=default_choice)
    if selected == 0:
        image = ask_text("Image path", root / "Assets" / "Samples" / "Chairs.jpg", required=True)
        if is_blend_path(image):
            print("That path is a .blend file. Guided mode will render a PNG preview instead.")
            output = ask_text("PNG output", "Output/Latest/render/image.png", required=True)
            args = render_args_for_blend(image, output)
            return _run_scene_args(args, execute)
        if not is_image_path(image):
            print("Image detection needs an image file such as .png, .jpg, .jpeg, .webp, .bmp, .tif, or .tiff.")
            return 2
        output = ask_text("Output directory", "Output/Latest/detect", required=True)
        args = detect_args_for_image(image, output)
        return _run_detection_then_completion(args, output)
    if selected == 1:
        source = ask_text("Reference .blend or image", root / "Assets" / "Samples" / "roomScene.blend", required=True)
        if is_image_path(source):
            print("Image input has no depth source, so guided mode will run RAM/DINO/SAM image detection.")
            output = ask_text("Output directory", "Output/Latest/detect", required=True)
            args = detect_args_for_image(source, output)
            return _run_detection_then_completion(args, output)
        if not is_blend_path(source):
            print("Option 2 needs a .blend or image file such as .png, .jpg, .jpeg, .webp, .bmp, .tif, or .tiff.")
            return 2
        output = ask_text("PNG output", "Output/Latest/render/image.png", required=True)
        args = render_args_for_blend(source, output)
        return _run_scene_args(args, execute)
    if selected == 2:
        blend = ask_text("Reference .blend", root / "Assets" / "Samples" / "roomScene.blend", required=True)
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
            "--render-samples", "1024",
            "--exposure", exposure,
        ], execute)
    if selected == 3:
        root_dir = ask_text("Open vocabulary root", "Models/OpenVocabulary", required=True)
        backend, ram_args = open_vocab_backend_with_ram_fallback()
        args = ["audit-open-vocab-readiness", "--root", root_dir, "--backend", backend]
        args.extend(ram_args)
        return _run_scene_args(args, execute)
    if selected == 4:
        root_dir = ask_text("Open vocabulary root", "Models/OpenVocabulary", required=True)
        args = ["run-open-vocab-smoke", "--root", root_dir, "--backend", "groundingdino-sam3"]
        backend, ram_args = open_vocab_backend_with_ram_fallback()
        args[4] = backend
        args.extend(ram_args)
        return _run_scene_args(args, execute)
    if selected == 5:
        blend = ask_text("Blend to inspect", "path/to/output.blend", required=True)
        return run_after_confirmation([sys.executable, "Tools/Scripts/view_blend.py", "--blend", blend, "--views", "front,iso", "--no-gltf"])
    if selected == 6:
        objects = ask_text("Objects directory", "Output/Latest/objects", required=True)
        args = ["complete-objects", "--objects", objects]
        completion_args = completion_args_if_available()
        if completion_args:
            args.extend(completion_args)
        return _run_scene_args(args, execute)
    if selected == 7:
        objects = ask_text("Objects directory", "Output/Latest/objects", required=True)
        args = [
            "reconstruct-objects",
            "--objects", objects,
            *object_reconstruction_args_if_available(),
        ]
        return _run_scene_args(args, execute)
    print_recipes()
    return 0


def _run_scene_args(args: list[str], execute: Callable[[list[str]], int]) -> int:
    print_command([sys.executable, "run.py", *args])
    if not confirm("Run now", default=True):
        return 0
    return int(execute(args))


def _run_detection_then_completion(args: list[str], output: str | Path) -> int:
    print_command([sys.executable, "run.py", *args])
    if not confirm("Run now", default=True):
        return 0
    status = _run_completion_process([sys.executable, "run.py", *args], banner="Running detection in an isolated process to release resources.")
    if status != 0:
        return status
    completion_args = completion_args_if_available()
    if not completion_args:
        return 0
    objects_dir = object_masks_dir_for_detect_output(Path(output))
    print("Running object completion in a fresh process to release detector GPU memory.")
    status = _run_completion_process(
        [sys.executable, "run.py", "complete-objects", "--objects", str(objects_dir), *completion_args]
    )
    if status != 0:
        fallback_args = lower_memory_completion_args(completion_args)
        if fallback_args != completion_args:
            status = _run_completion_process(
                [sys.executable, "run.py", "complete-objects", "--objects", str(objects_dir), *fallback_args],
                banner="Object completion failed; retrying with lower-memory settings.",
            )
    if status != 0:
        return status
    reconstruction_args = object_reconstruction_args_if_available()
    if not reconstruction_args:
        return 0
    print("Running object mesh reconstruction over completed object crops.")
    return _run_completion_process(
        [sys.executable, "run.py", "reconstruct-objects", "--objects", str(objects_dir), *reconstruction_args]
    )


def _run_completion_process(command: list[str], banner: str | None = None) -> int:
    print_command(command)
    if banner:
        print(banner)
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return subprocess.run([str(part) for part in command], cwd=repo_root(), check=False, env=env).returncode


def lower_memory_completion_args(args: list[str]) -> list[str]:
    if "--completion-backend" not in args or "flux-fill" not in args:
        return args
    lowered = replace_option_value(args, "--completion-device", "cpu")
    lowered = replace_option_value(lowered, "--completion-canvas-size", "768")
    lowered = replace_option_value(lowered, "--completion-steps", "16")
    lowered = replace_option_value(lowered, "--completion-max-objects", "1")
    lowered = replace_option_value(lowered, "--completion-quantization", "none")
    return lowered


def replace_option_value(args: list[str], option: str, value: str) -> list[str]:
    if option in args:
        index = args.index(option)
        return args[:index + 1] + [value] + args[index + 2 :]
    return [*args, option, value]


def object_masks_dir_for_detect_output(output: Path) -> Path:
    if output.name == "detect":
        return output.parent / "objects"
    return output / "objects"


def print_recipes() -> None:
    backend, ram_args = open_vocab_backend_with_ram_fallback()
    detect_prompt_args = [] if backend == "ram-groundingdino-sam3" else ["--text-prompt-preset", "scene-primitives-v1"]
    completion_args = completion_args_if_available()
    reconstruction_args = object_reconstruction_args_if_available()
    recipes = [
        [sys.executable, "run.py", "audit-open-vocab-readiness", "--root", "Models/OpenVocabulary", "--backend", backend] + ram_args,
        [sys.executable, "run.py", "run-open-vocab-smoke", "--root", "Models/OpenVocabulary", "--backend", backend] + ram_args,
        [sys.executable, "run.py", "detect-shapes", "--backend", backend, "--image", "path/to/image.png", "--open-vocab-root", "Models/OpenVocabulary"] + detect_prompt_args + ["--output", "Output/Latest/detect", "--device", "auto"] + ram_args,
        [sys.executable, "run.py", "complete-objects", "--objects", "Output/Latest/objects"] + completion_args,
        [sys.executable, "run.py", "reconstruct-objects", "--objects", "Output/Latest/objects"] + reconstruction_args,
        [sys.executable, "run.py", "render-blend-png", "--reference-blend", "path/to/file.blend", "--output", "Output/Latest/render/image.png", "--width", "1280", "--height", "720", "--exposure", "auto"],
    ]
    for command in recipes:
        print(shell_join(command))


def remove_option_with_value(args: list[str], option: str) -> list[str]:
    if option not in args:
        return args
    index = args.index(option)
    return args[:index] + args[index + 2 :]


def open_vocab_backend_with_ram_fallback() -> tuple[str, list[str]]:
    ram_repo, ram_checkpoint = _resolve_ram_locations()
    if ram_repo and ram_checkpoint:
        return "ram-groundingdino-sam3", [
            "--ram-repo-dir", str(ram_repo),
            "--ram-checkpoint", str(ram_checkpoint),
        ]
    if os.environ.get("SCENEFORGE_DISABLE_RAM", "0") == "1":
        return "groundingdino-sam3", []
    print("Warning: RAM paths not found; falling back to groundingdino-sam3.")
    return "groundingdino-sam3", []


def _resolve_ram_locations() -> tuple[Path | None, Path | None]:
    env_repo = os.environ.get("SCENEFORGE_RAM_REPO_DIR")
    env_checkpoint = os.environ.get("SCENEFORGE_RAM_CHECKPOINT")
    if env_repo and env_checkpoint:
        candidate_repo = Path(env_repo)
        candidate_checkpoint = Path(env_checkpoint)
        if candidate_repo.is_dir() and candidate_checkpoint.is_file():
            return candidate_repo, candidate_checkpoint

    repo_candidates = (
        Path("Models/RAM"),
        Path("Models/OpenVocabulary/RAM"),
    )
    for base in repo_candidates:
        ram_repo_dir = base / "repo"
        if not ram_repo_dir.is_dir():
            continue
        weights_dir = base / "weights"
        ram_checkpoints = sorted(weights_dir.glob("*.pth")) if weights_dir.is_dir() else []
        if not ram_checkpoints:
            ram_checkpoints = sorted(base.glob("*.pth"))
        if ram_checkpoints:
            return ram_repo_dir, ram_checkpoints[0]
    return None, None
