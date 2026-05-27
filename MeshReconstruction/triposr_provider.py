from __future__ import annotations

import os
from pathlib import Path
import sys
import types

import numpy as np
from PIL import Image

from MeshReconstruction.types import MeshProvider, MeshResult


class TripoSRMeshProvider(MeshProvider):
    backend = "triposr"

    def __init__(self, model_dir: str | Path, device: str | None = None) -> None:
        self.model_dir = Path(model_dir)
        self.device = device
        if not self.model_dir.is_dir():
            raise ValueError(f"--mesh-model-dir does not exist or is not a directory: {self.model_dir}")
        self.repo_dir = self._find_repo_dir()
        self.weights_dir = self._find_weights_dir()
        self.dino_dir = self._find_dino_dir()
        self.cache_dir = self.model_dir / "cache"
        self._model = None
        self._resolved_device: str | None = None

    def _find_repo_dir(self) -> Path:
        candidates = [
            self.model_dir / "repo",
            self.model_dir,
        ]
        for candidate in candidates:
            if (candidate / "tsr" / "system.py").is_file():
                return candidate
        raise ValueError(
            "TripoSR model directory is missing the source repo. "
            "Expected --mesh-model-dir/repo/tsr/system.py."
        )

    def _find_weights_dir(self) -> Path:
        candidates = [
            self.model_dir / "hf",
            self.model_dir,
        ]
        for candidate in candidates:
            if (candidate / "config.yaml").is_file() and (candidate / "model.ckpt").is_file():
                return candidate
        raise ValueError(
            "TripoSR model directory is missing config.yaml and model.ckpt. "
            "Expected them under --mesh-model-dir/hf/."
        )

    def _find_dino_dir(self) -> Path:
        candidates = [
            self.model_dir / "dino-vitb16",
            self.model_dir / "facebook" / "dino-vitb16",
        ]
        for candidate in candidates:
            if (candidate / "config.json").is_file() and (
                (candidate / "pytorch_model.bin").is_file()
                or (candidate / "model.safetensors").is_file()
            ):
                return candidate
        raise ValueError(
            "TripoSR model directory is missing the local facebook/dino-vitb16 dependency. "
            "Expected it under --mesh-model-dir/dino-vitb16/."
        )

    def _load_model(self):
        if self._model is not None:
            return self._model

        self._install_torchmcubes_fallback()
        self._install_rembg_fallback()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(self.cache_dir))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(self.cache_dir / "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(self.cache_dir / "transformers"))
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("TripoSR mesh provider requires torch.") from exc

        if str(self.repo_dir) not in sys.path:
            sys.path.insert(0, str(self.repo_dir))
        self._patch_local_dino_download()
        from tsr.system import TSR

        requested_device = self.device or "cuda:0"
        if requested_device.isdigit():
            requested_device = f"cuda:{requested_device}"
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"

        local_config_name = self._write_local_config()
        model = TSR.from_pretrained(
            str(self.weights_dir),
            config_name=local_config_name,
            weight_name="model.ckpt",
        )
        model.renderer.set_chunk_size(4096)
        model.to(requested_device)
        model.eval()
        self._model = model
        self._resolved_device = requested_device
        return model

    def reconstruct(self, rgb_crop_path: Path, mask_path: Path, output_path: Path) -> MeshResult:
        try:
            import torch
        except ImportError as exc:
            return MeshResult(status="failed", path=None, reason=f"torch is unavailable: {exc}")

        try:
            model = self._load_model()
            image = self._prepare_input(rgb_crop_path, mask_path)
            with torch.no_grad():
                scene_codes = model([image], device=self._resolved_device or "cpu")
                meshes = model.extract_mesh(scene_codes, has_vertex_color=True, resolution=96)
            if not meshes:
                return MeshResult(status="invalid", path=None, reason="TripoSR returned no meshes.")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            meshes[0].export(output_path)
            return MeshResult(status="ok", path=output_path)
        except Exception as exc:
            return MeshResult(status="failed", path=None, reason=str(exc))

    @staticmethod
    def _prepare_input(rgb_crop_path: Path, mask_path: Path) -> Image.Image:
        rgb = Image.open(rgb_crop_path).convert("RGB")
        mask = Image.open(mask_path).convert("L").resize(rgb.size)
        rgb_array = np.asarray(rgb, dtype=np.float32) / 255.0
        alpha = (np.asarray(mask, dtype=np.float32) / 255.0)[..., None]
        composited = rgb_array * alpha + (1.0 - alpha) * 0.5
        return Image.fromarray(np.clip(composited * 255.0, 0, 255).astype(np.uint8), mode="RGB")

    @staticmethod
    def _install_torchmcubes_fallback() -> None:
        if "torchmcubes" in sys.modules:
            return

        module = types.ModuleType("torchmcubes")

        def marching_cubes(level, threshold):
            import torch
            from skimage import measure

            volume = level.detach().cpu().numpy().astype(np.float32)
            try:
                verts, faces, _, _ = measure.marching_cubes(volume, level=float(threshold))
            except ValueError:
                verts = np.zeros((0, 3), dtype=np.float32)
                faces = np.zeros((0, 3), dtype=np.int64)
            return (
                torch.as_tensor(verts.copy(), dtype=torch.float32, device=level.device),
                torch.as_tensor(faces.copy(), dtype=torch.long, device=level.device),
            )

        module.marching_cubes = marching_cubes
        sys.modules["torchmcubes"] = module

    @staticmethod
    def _install_rembg_fallback() -> None:
        if "rembg" in sys.modules:
            return

        module = types.ModuleType("rembg")

        def new_session(*args, **kwargs):
            del args, kwargs
            return None

        def remove(image, *args, **kwargs):
            del args, kwargs
            return image

        module.new_session = new_session
        module.remove = remove
        sys.modules["rembg"] = module

    def _patch_local_dino_download(self) -> None:
        import importlib

        image_tokenizer = importlib.import_module("tsr.models.tokenizers.image")
        original_download = image_tokenizer.hf_hub_download
        local_dino = self.dino_dir.resolve()

        def local_first_download(repo_id: str, filename: str, *args, **kwargs):
            repo_path = Path(repo_id)
            if repo_path.is_dir():
                candidate = repo_path / filename
                if candidate.is_file():
                    return str(candidate)
            if repo_id == "facebook/dino-vitb16":
                candidate = local_dino / filename
                if candidate.is_file():
                    return str(candidate)
            kwargs.setdefault("local_files_only", True)
            return original_download(repo_id=repo_id, filename=filename, *args, **kwargs)

        image_tokenizer.hf_hub_download = local_first_download

    def _write_local_config(self) -> str:
        source = self.weights_dir / "config.yaml"
        target = self.weights_dir / "config.local.yaml"
        text = source.read_text(encoding="utf-8")
        text = text.replace('"facebook/dino-vitb16"', f'"{self.dino_dir.resolve()}"')
        if not target.is_file() or target.read_text(encoding="utf-8") != text:
            target.write_text(text, encoding="utf-8")
        return target.name
