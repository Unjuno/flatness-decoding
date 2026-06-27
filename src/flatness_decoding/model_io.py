from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .devices import dtype_from_name, resolve_device


DEFAULT_MODEL_CACHE_DIR = "models/hf"


def normalize_cache_dir(cache_dir: str | None = None) -> str | None:
    """Return a local cache directory for Hugging Face downloads.

    Model weights are intentionally not tracked by Git. By default, this project
    stores downloaded model artifacts under `models/hf/`, which is ignored by
    `.gitignore`.
    """
    if cache_dir is None or cache_dir == "":
        cache_dir = DEFAULT_MODEL_CACHE_DIR
    if cache_dir in {"none", "None", "null"}:
        return None
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return cache_dir


def load_causal_lm(
    model_name: str,
    device_name: str = "auto",
    dtype_name: str = "auto",
    cache_dir: str | None = None,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
):
    """Load a causal LM and tokenizer for custom decoding.

    If the model is not already cached, `transformers` will download it on first
    use. The downloaded files stay in the local cache directory and are not
    committed to the repository.

    Returns `(model, tokenizer, device)`.
    """
    device = resolve_device(device_name)
    dtype = dtype_from_name(dtype_name, device)
    resolved_cache_dir = normalize_cache_dir(cache_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=resolved_cache_dir,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        cache_dir=resolved_cache_dir,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    model.eval()

    return model, tokenizer, device
