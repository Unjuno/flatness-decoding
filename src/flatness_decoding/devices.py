from __future__ import annotations

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve a requested device name to a torch.device.

    Priority for "auto": cuda > mps > cpu.
    """
    requested = requested.lower()

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")

    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is False.")
        return torch.device("mps")

    if requested == "cpu":
        return torch.device("cpu")

    if requested != "auto":
        raise ValueError(f"Unknown device request: {requested}")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def dtype_from_name(dtype_name: str, device: torch.device) -> torch.dtype:
    """Return a dtype for model loading.

    The default favors stability on CPU/MPS and speed on CUDA.
    """
    dtype_name = dtype_name.lower()

    if dtype_name == "auto":
        if device.type == "cuda":
            return torch.float16
        return torch.float32

    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16

    raise ValueError(f"Unknown dtype: {dtype_name}")


def describe_torch_devices() -> dict[str, object]:
    """Return simple device diagnostics for CLI checks."""
    return {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
    }
