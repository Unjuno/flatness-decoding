from __future__ import annotations

import gc

import torch


def cleanup_device_cache(device: torch.device | str | None = None, *, collect_python: bool = False) -> dict[str, bool]:
    """Release unoccupied accelerator cache without changing live tensors.

    This is a memory-hygiene helper for long-running generation jobs. It does
    not alter model weights, logits, RNG state, prompts, candidates, or the
    decoding criterion. It only asks the backend allocator to release cached
    memory that is no longer occupied by live tensors.
    """
    if collect_python:
        gc.collect()

    device_type = getattr(device, "type", None) if device is not None else None
    if isinstance(device, str):
        device_type = device.split(":", 1)[0]

    ran_mps = False
    ran_cuda = False

    if device_type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
        ran_mps = True
    elif device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        ran_cuda = True

    return {"mps_empty_cache": ran_mps, "cuda_empty_cache": ran_cuda, "python_gc": collect_python}


def cleanup_config(cfg: dict) -> dict:
    """Return runtime cleanup configuration with conservative defaults."""
    runtime_cfg = cfg.get("runtime", {}) or {}
    cleanup_cfg = runtime_cfg.get("cleanup", {}) or {}
    return {
        "device_cache_after_prompt": bool(cleanup_cfg.get("device_cache_after_prompt", False)),
        "device_cache_after_group": bool(cleanup_cfg.get("device_cache_after_group", True)),
        "python_gc_after_group": bool(cleanup_cfg.get("python_gc_after_group", False)),
    }
