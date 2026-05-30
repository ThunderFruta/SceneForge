from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from Input.Image.image_loader import load_rgb_image
from SceneGeometry.coordinate_contract import camera_fusion_contract


SCHEMA_VERSION = 1


@dataclass
class VggtResult:
    depth: np.ndarray
    points: np.ndarray
    confidence: np.ndarray | None
    camera: dict[str, Any]
    model_info: dict[str, Any]


def run_vggt_image_geometry(
    image_path: str | Path,
    output_dir: str | Path,
    *,
    backend: str = "vggt",
    model: str = "facebook/VGGT-1B",
    repo_dir: str | Path | None = None,
    checkpoint: str | Path | None = None,
    device: str = "auto",
    local_only: bool = False,
    cache_dir: str | Path = "Models/Geometry/VGGT/hf-cache",
    obj_stride: int = 8,
    mesh_stem: str = "vggt_mesh",
) -> dict[str, Any]:
    image_path = Path(image_path)
    image = load_rgb_image(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if backend == "fake":
        result = run_fake_vggt(image, model=model)
    elif backend == "vggt":
        result = run_real_vggt(
            image_path=image_path,
            image_size=image.size,
            model=model,
            repo_dir=repo_dir,
            checkpoint=checkpoint,
            device=device,
            local_only=local_only,
            cache_dir=cache_dir,
        )
    else:
        raise ValueError(f"Unsupported VGGT backend: {backend}")

    depth = normalize_depth(result.depth, image.size)
    points = normalize_points(result.points, image.size)
    confidence = normalize_confidence(result.confidence, image.size)

    depth_npy_path = output_dir / "vggt_depth.npy"
    points_npy_path = output_dir / "vggt_points.npy"
    depth_png_path = output_dir / "vggt_depth.png"
    points_xyz_path = output_dir / "vggt_points.xyz"
    mesh_stem = safe_mesh_stem(mesh_stem)
    mesh_obj_path = output_dir / f"{mesh_stem}.obj"
    mesh_glb_path = output_dir / f"{mesh_stem}.glb"
    confidence_png_path = output_dir / "vggt_confidence.png"
    camera_path = output_dir / "vggt_camera.json"
    report_path = output_dir / "vggt_geometry.json"

    np.save(depth_npy_path, depth.astype(np.float32))
    np.save(points_npy_path, points.astype(np.float32))
    write_depth_png(depth, depth_png_path)
    write_points_xyz(points, points_xyz_path)
    mesh_data = build_sampled_point_mesh(points, image=image, stride=obj_stride)
    mesh_stats = write_points_obj(mesh_data, mesh_obj_path)
    glb_stats = write_points_glb(mesh_data, mesh_glb_path)
    if confidence is not None:
        write_confidence_png(confidence, confidence_png_path)

    camera = build_camera_report(result.camera, image.width, image.height)
    camera_path.write_text(json.dumps(camera, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    valid_depth = np.isfinite(depth) & (depth > 0)
    finite_points = np.isfinite(points).all(axis=2)
    valid_points = valid_depth & finite_points
    report = {
        "schema_version": SCHEMA_VERSION,
        "image_path": str(image_path),
        "image_width": image.width,
        "image_height": image.height,
        "backend": backend,
        "artifacts": {
            "depth_npy": depth_npy_path.name,
            "depth_png": depth_png_path.name,
            "points_npy": points_npy_path.name,
            "points_xyz": points_xyz_path.name,
            "mesh_obj": mesh_obj_path.name,
            "mesh_glb": mesh_glb_path.name,
            "confidence_png": confidence_png_path.name if confidence is not None else None,
            "camera": camera_path.name,
        },
        "coordinate_contract": camera_fusion_contract(image_width=image.width, image_height=image.height),
        "summary": {
            "depth_min": float(np.nanmin(depth[valid_depth])) if bool(valid_depth.any()) else None,
            "depth_max": float(np.nanmax(depth[valid_depth])) if bool(valid_depth.any()) else None,
            "valid_depth_ratio": float(valid_depth.mean()) if depth.size else 0.0,
            "valid_point_ratio": float(valid_points.mean()) if valid_points.size else 0.0,
            "point_count": int(valid_points.sum()),
            "obj_vertex_count": mesh_stats["vertex_count"],
            "obj_face_count": mesh_stats["face_count"],
            "obj_stride": mesh_stats["stride"],
            "obj_winding": mesh_stats["winding"],
            "obj_point_source": "sceneforge_camera_points",
            "glb_vertex_count": glb_stats["vertex_count"],
            "glb_face_count": glb_stats["face_count"],
            "glb_stride": glb_stats["stride"],
            "glb_point_source": "sceneforge_camera_points",
            "glb_axis_transform": glb_stats["axis_transform"],
            "glb_vertex_colors": glb_stats["vertex_colors"],
        },
        "model_info": result.model_info,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def run_fake_vggt(image: Image.Image, *, model: str) -> VggtResult:
    width, height = image.size
    xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    ys = np.linspace(1.0, -1.0, height, dtype=np.float32)
    grid_x, grid_z = np.meshgrid(xs, ys)
    rgb = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    depth = 1.0 + (1.0 - rgb) * 2.0
    points = np.stack([grid_x * depth, depth, grid_z * depth], axis=2).astype(np.float32)
    confidence = np.ones((height, width), dtype=np.float32)
    return VggtResult(
        depth=depth,
        points=points,
        confidence=confidence,
        camera={
            "source": "fake_vggt",
            "intrinsics": None,
            "extrinsics": None,
        },
        model_info={
            "backend": "fake",
            "model": model,
            "notes": "Deterministic test backend; not real VGGT geometry.",
        },
    )


def run_real_vggt(
    *,
    image_path: Path,
    image_size: tuple[int, int],
    model: str,
    repo_dir: str | Path | None,
    checkpoint: str | Path | None,
    device: str,
    local_only: bool,
    cache_dir: str | Path,
) -> VggtResult:
    hf_home = Path("Models/Geometry/VGGT/hf").resolve()
    hf_cache = Path(cache_dir).resolve()
    hf_home.mkdir(parents=True, exist_ok=True)
    hf_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_cache))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_cache))

    if repo_dir is not None:
        resolved_repo = Path(repo_dir).resolve()
        if not resolved_repo.is_dir():
            raise ValueError(f"--vggt-repo-dir does not exist or is not a directory: {resolved_repo}")
        if str(resolved_repo) not in sys.path:
            sys.path.insert(0, str(resolved_repo))

    try:
        import torch
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
    except Exception as exc:
        raise RuntimeError(
            "VGGT is not importable. Install/clone VGGT and pass --vggt-repo-dir, "
            "or run this stage with --backend fake for contract tests."
        ) from exc

    resolved_device = resolve_torch_device(torch, device)
    with torch.no_grad():
        vggt_model = load_vggt_model(VGGT, model=model, checkpoint=checkpoint, local_only=local_only, cache_dir=hf_cache)
        vggt_model = vggt_model.to(resolved_device)
        if hasattr(vggt_model, "eval"):
            vggt_model.eval()
        images = load_and_preprocess_images([str(image_path)]).to(resolved_device)
        with maybe_autocast(torch, resolved_device):
            predictions = vggt_model(images)

    predictions_cpu = to_cpu_tree(predictions)
    depth = extract_depth(predictions_cpu, image_size)
    points = convert_vggt_points_to_sceneforge_camera(extract_points(predictions_cpu, image_size))
    confidence = extract_confidence(predictions_cpu, image_size)
    camera = extract_camera(predictions_cpu, image_size)
    return VggtResult(
        depth=depth,
        points=points,
        confidence=confidence,
        camera=camera,
        model_info={
            "backend": "vggt",
            "model": model,
            "checkpoint": str(checkpoint) if checkpoint is not None else None,
            "device": resolved_device,
            "local_only": local_only,
            "cache_dir": str(hf_cache),
        },
    )


def load_vggt_model(
    vggt_class,
    *,
    model: str,
    checkpoint: str | Path | None,
    local_only: bool,
    cache_dir: Path,
):
    if checkpoint is not None:
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("Loading a VGGT checkpoint requires torch.") from exc
        instance = vggt_class()
        state = torch.load(checkpoint, map_location="cpu")
        state_dict = state.get("model", state.get("state_dict", state)) if isinstance(state, dict) else state
        instance.load_state_dict(state_dict)
        return instance
    if hasattr(vggt_class, "from_pretrained"):
        if local_only:
            local_snapshot = find_local_hf_snapshot(model, cache_dir)
            if local_snapshot is None:
                raise RuntimeError(
                    f"VGGT model {model!r} is not available in local cache {cache_dir}. "
                    "Run without --vggt-local-only once, or download it with hf download."
                )
            return vggt_class.from_pretrained(str(local_snapshot))
        return vggt_class.from_pretrained(model)
    return vggt_class()


def find_local_hf_snapshot(model: str, cache_dir: Path) -> Path | None:
    repo_cache = cache_dir / f"models--{model.replace('/', '--')}" / "snapshots"
    if not repo_cache.is_dir():
        return None
    candidates = []
    for snapshot in repo_cache.iterdir():
        if not snapshot.is_dir():
            continue
        if (snapshot / "model.safetensors").exists() or (snapshot / "model.pt").exists():
            candidates.append(snapshot)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def maybe_autocast(torch, device: str):
    if str(device).startswith("cuda"):
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast("cuda", dtype=torch.bfloat16)
        return torch.cuda.amp.autocast(dtype=torch.bfloat16)
    try:
        from contextlib import nullcontext
    except ImportError:
        return None
    return nullcontext()


def resolve_torch_device(torch, device: str) -> str:
    if device in {"", "auto", None}:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if str(device).isdigit():
        return f"cuda:{device}"
    return str(device)


def to_cpu_tree(value):
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().float().numpy()
    if isinstance(value, dict):
        return {key: to_cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_cpu_tree(item) for item in value]
    return value


def extract_depth(predictions: dict[str, Any], image_size: tuple[int, int]) -> np.ndarray:
    for key in ("depth", "depths", "depth_map", "depth_maps"):
        if key in predictions:
            return squeeze_image_array(predictions[key], image_size, channels=1)
    raise RuntimeError("VGGT output did not include a recognized depth field.")


def extract_points(predictions: dict[str, Any], image_size: tuple[int, int]) -> np.ndarray:
    for key in ("world_points", "points", "point_map", "points3d", "xyz"):
        if key in predictions:
            return squeeze_image_array(predictions[key], image_size, channels=3)
    depth = extract_depth(predictions, image_size)
    return points_from_depth(depth)


def convert_vggt_points_to_sceneforge_camera(points: np.ndarray) -> np.ndarray:
    """Map VGGT/OpenCV-style points into SceneForge x-right, y-depth, z-up axes."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise RuntimeError(f"Expected VGGT point map with final channel size 3, got shape {points.shape}")
    converted = np.empty_like(points, dtype=np.float32)
    converted[..., 0] = points[..., 0]
    converted[..., 1] = points[..., 2]
    converted[..., 2] = -points[..., 1]
    return converted


def extract_confidence(predictions: dict[str, Any], image_size: tuple[int, int]) -> np.ndarray | None:
    for key in ("depth_conf", "confidence", "conf", "depth_confidence"):
        if key in predictions:
            return squeeze_image_array(predictions[key], image_size, channels=1)
    return None


def extract_camera(predictions: dict[str, Any], image_size: tuple[int, int]) -> dict[str, Any]:
    camera: dict[str, Any] = {
        "source": "vggt",
        "image_width": image_size[0],
        "image_height": image_size[1],
    }
    for key in ("intrinsic", "intrinsics", "K"):
        if key in predictions:
            camera["intrinsics"] = numpy_to_jsonable(predictions[key])
            break
    for key in ("extrinsic", "extrinsics", "camera_pose", "pose"):
        if key in predictions:
            camera["extrinsics"] = numpy_to_jsonable(predictions[key])
            break
    return camera


def squeeze_image_array(value: Any, image_size: tuple[int, int], *, channels: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    array = np.squeeze(array)
    width, height = image_size
    if channels == 1:
        if array.ndim == 3:
            array = array[..., 0]
        if array.shape != (height, width):
            array = resize_float_image(array, (width, height))
        return array.astype(np.float32)
    if channels == 3:
        if array.ndim != 3:
            raise RuntimeError(f"Expected point map with 3 channels, got shape {array.shape}")
        if array.shape[0] == 3 and array.shape[-1] != 3:
            array = np.moveaxis(array, 0, -1)
        if array.shape[-1] != 3:
            raise RuntimeError(f"Expected point map final channel size 3, got shape {array.shape}")
        if array.shape[:2] != (height, width):
            planes = [resize_float_image(array[..., channel], (width, height)) for channel in range(3)]
            array = np.stack(planes, axis=2)
        return array.astype(np.float32)
    raise ValueError(f"Unsupported channel count: {channels}")


def normalize_depth(depth: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    depth = squeeze_image_array(depth, image_size, channels=1)
    return np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def normalize_points(points: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    points = squeeze_image_array(points, image_size, channels=3)
    return np.nan_to_num(points.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def normalize_confidence(confidence: np.ndarray | None, image_size: tuple[int, int]) -> np.ndarray | None:
    if confidence is None:
        return None
    return np.clip(squeeze_image_array(confidence, image_size, channels=1), 0.0, 1.0)


def resize_float_image(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(values.astype(np.float32), mode="F")
    return np.asarray(image.resize(size, Image.Resampling.BILINEAR), dtype=np.float32)


def points_from_depth(depth: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    zs = np.linspace(1.0, -1.0, height, dtype=np.float32)
    grid_x, grid_z = np.meshgrid(xs, zs)
    return np.stack([grid_x * depth, depth, grid_z * depth], axis=2).astype(np.float32)


def write_depth_png(depth: np.ndarray, output_path: Path) -> None:
    values = depth.astype(np.float32)
    valid = np.isfinite(values) & (values > 0)
    if bool(valid.any()):
        near = float(values[valid].min())
        far = float(values[valid].max())
        if far > near:
            normalized = (far - values) / (far - near)
        else:
            normalized = np.ones_like(values)
    else:
        normalized = np.zeros_like(values)
    image = Image.fromarray(np.rint(np.clip(normalized, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    image.save(output_path)


def write_confidence_png(confidence: np.ndarray, output_path: Path) -> None:
    image = Image.fromarray(np.rint(np.clip(confidence, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    image.save(output_path)


def write_points_xyz(points: np.ndarray, output_path: Path, *, stride: int = 8) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# x y z\n")
        sampled = points[::stride, ::stride, :]
        for point in sampled.reshape(-1, 3):
            if np.isfinite(point).all():
                handle.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")


def safe_mesh_stem(value: str) -> str:
    stem = "".join(character if character.isalnum() or character in {"_", "-"} else "_" for character in str(value).strip())
    return stem.strip("._-") or "vggt_mesh"


def build_sampled_point_mesh(points: np.ndarray, *, image: Image.Image, stride: int = 8) -> dict[str, Any]:
    stride = max(1, int(stride))
    sampled = points[::stride, ::stride, :]
    finite = np.isfinite(sampled).all(axis=2)
    vertex_indices = np.zeros(sampled.shape[:2], dtype=np.int32)
    sampled_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)[::stride, ::stride, :]
    vertices_obj: list[tuple[float, float, float]] = []
    vertices_gltf: list[tuple[float, float, float]] = []
    vertex_colors: list[tuple[int, int, int, int]] = []
    faces: list[tuple[int, int, int]] = []

    for row in range(sampled.shape[0]):
        for col in range(sampled.shape[1]):
            if not finite[row, col]:
                continue
            vertex_indices[row, col] = len(vertices_obj) + 1
            point = sampled[row, col]
            vertices_obj.append(scene_point_to_blender_obj_vertex(point))
            vertices_gltf.append(scene_point_to_gltf_vertex(point))
            rgb = sampled_rgb[row, col]
            vertex_colors.append((int(rgb[0]), int(rgb[1]), int(rgb[2]), 255))

    for row in range(sampled.shape[0] - 1):
        for col in range(sampled.shape[1] - 1):
            v00 = int(vertex_indices[row, col])
            v10 = int(vertex_indices[row, col + 1])
            v01 = int(vertex_indices[row + 1, col])
            v11 = int(vertex_indices[row + 1, col + 1])
            if not (v00 and v10 and v01 and v11):
                continue
            faces.append((v00 - 1, v11 - 1, v10 - 1))
            faces.append((v00 - 1, v01 - 1, v11 - 1))

    return {
        "stride": stride,
        "vertices_obj": np.asarray(vertices_obj, dtype=np.float32),
        "vertices_gltf": np.asarray(vertices_gltf, dtype=np.float32),
        "vertex_colors": np.asarray(vertex_colors, dtype=np.uint8),
        "faces": np.asarray(faces, dtype=np.int64),
    }


def write_points_obj(mesh_data: dict[str, Any], output_path: Path) -> dict[str, Any]:
    vertices = np.asarray(mesh_data["vertices_obj"], dtype=np.float32)
    faces = np.asarray(mesh_data["faces"], dtype=np.int64)
    colors = np.asarray(mesh_data["vertex_colors"], dtype=np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# SceneForge VGGT sampled point-map mesh\n")
        handle.write("# point_source sceneforge_camera_points\n")
        handle.write("# obj_source_axes x=image_right y=image_up z=back_toward_camera\n")
        handle.write("# blender_default_obj_import lands as x=image_right y=depth_away_from_camera z=image_up\n")
        handle.write("# source VGGT world point-map artifacts are also preserved in vggt_points.npy and vggt_points.xyz\n")
        handle.write(f"# stride {int(mesh_data['stride'])}\n")
        for index, vertex in enumerate(vertices):
            color = colors[index] if len(colors) > index else (255, 255, 255, 255)
            handle.write(
                f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f} "
                f"{color[0] / 255.0:.6f} {color[1] / 255.0:.6f} {color[2] / 255.0:.6f}\n"
            )
        for face in faces:
            handle.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n")

    return {
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
        "stride": int(mesh_data["stride"]),
        "winding": "camera_facing",
    }


def write_points_glb(mesh_data: dict[str, Any], output_path: Path) -> dict[str, Any]:
    try:
        import trimesh
    except Exception as exc:
        raise RuntimeError("Exporting VGGT GLB meshes requires trimesh from requirements.txt.") from exc

    vertices = np.asarray(mesh_data["vertices_gltf"], dtype=np.float32)
    faces = np.asarray(mesh_data["faces"], dtype=np.int64)
    colors = np.asarray(mesh_data["vertex_colors"], dtype=np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors, process=False)
    mesh.export(output_path)
    return {
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
        "stride": int(mesh_data["stride"]),
        "axis_transform": "gltf_x_image_right_y_image_up_z_back_toward_camera",
        "vertex_colors": bool(len(colors) == len(vertices) and len(colors) > 0),
    }


def scene_point_to_blender_obj_vertex(point: np.ndarray) -> tuple[float, float, float]:
    return (float(point[0]), float(-point[1]), float(-point[2]))


def scene_point_to_gltf_vertex(point: np.ndarray) -> tuple[float, float, float]:
    return (float(point[0]), float(point[2]), float(-point[1]))


def build_camera_report(camera: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "image_width": int(width),
        "image_height": int(height),
        "coordinate_contract": camera_fusion_contract(image_width=width, image_height=height),
        "vggt_camera": camera,
    }


def numpy_to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: numpy_to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [numpy_to_jsonable(item) for item in value]
    try:
        return value.item()
    except AttributeError:
        return value
