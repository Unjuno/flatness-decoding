from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .devices import dtype_from_name, resolve_device


def load_causal_lm(
    model_name: str,
    device_name: str = "auto",
    dtype_name: str = "auto",
):
    """Load a causal LM and tokenizer for custom decoding.

    Returns `(model, tokenizer, device)`.
    """
    device = resolve_device(device_name)
    dtype = dtype_from_name(dtype_name, device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()

    return model, tokenizer, device
