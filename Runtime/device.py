from __future__ import annotations


def resolve_torch_device(device: str | None) -> str | None:
    """Resolve a CLI device string to a torch-compatible device.

    This is intentionally backend-neutral so open-vocabulary detectors, object
    reconstruction backends, and future VGGT/Hunyuan-style stages can share the
    same CUDA/CPU selection.
    """
    if device is None:
        return None
    value = str(device).strip()
    if not value:
        return None
    if value.lower() == "auto":
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    lowered = value.lower()
    if lowered in {"cpu", "cuda", "auto"}:
        if lowered in {"cuda", "auto"}:
            try:
                import torch
            except ImportError:
                return "cpu"
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cpu"

    if lowered.isdigit():
        try:
            import torch
        except ImportError:
            return "cpu"
        return f"cuda:{value}" if torch.cuda.is_available() else "cpu"

    try:
        import torch  # noqa: F401
    except ImportError:
        return "cpu"
    return value
