from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flatness_decoding.devices import dtype_from_name, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Hugging Face causal LM compatibility.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-tokenizer", action="store_true")
    parser.add_argument("--run-forward", action="store_true")
    parser.add_argument(
        "--prompt",
        default="Write one short sentence about a traveler at a train station.",
    )
    return parser.parse_args()


def load_tokenizer(model_id: str, local_files_only: bool, trust_remote_code: bool):
    return AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )


def load_model(model_id: str, device: torch.device, dtype_name: str, local_files_only: bool, trust_remote_code: bool):
    dtype = dtype_from_name(dtype_name, device)
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
    )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = dtype_from_name(args.dtype, device)

    print(f"model_id: {args.model_id}")
    print(f"device: {device}")
    print(f"dtype: {dtype}")

    tokenizer = load_tokenizer(args.model_id, args.local_files_only, args.trust_remote_code)
    print(f"tokenizer_class: {tokenizer.__class__.__name__}")
    print(f"vocab_size: {getattr(tokenizer, 'vocab_size', None)}")
    print(f"eos_token: {getattr(tokenizer, 'eos_token', None)}")
    print(f"eos_token_id: {getattr(tokenizer, 'eos_token_id', None)}")
    print(f"pad_token: {getattr(tokenizer, 'pad_token', None)}")
    print(f"pad_token_id: {getattr(tokenizer, 'pad_token_id', None)}")
    print(f"bos_token: {getattr(tokenizer, 'bos_token', None)}")
    print(f"bos_token_id: {getattr(tokenizer, 'bos_token_id', None)}")
    print(f"padding_side: {getattr(tokenizer, 'padding_side', None)}")
    print(f"model_max_length: {getattr(tokenizer, 'model_max_length', None)}")
    print(f"chat_template_available: {getattr(tokenizer, 'chat_template', None) is not None}")
    if getattr(tokenizer, "pad_token_id", None) is None:
        print("warning: pad_token_id is missing; using eos_token_id as pad may be needed.")
    if hasattr(tokenizer, "apply_chat_template"):
        print("note: apply_chat_template is available; instruct prompt formatting should be documented.")

    if args.run_tokenizer:
        encoded = tokenizer(args.prompt, return_tensors="pt")
        print(f"tokenizer_input_keys: {sorted(encoded.keys())}")
        print(f"tokenizer_input_shape: {tuple(encoded['input_ids'].shape)}")

    if not args.run_forward:
        return

    model = load_model(args.model_id, device, args.dtype, args.local_files_only, args.trust_remote_code)
    print(f"model_class: {model.__class__.__name__}")
    print(f"model_dtype: {next(model.parameters()).dtype}")
    max_positions = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if max_positions is None:
        max_positions = getattr(getattr(model, "config", None), "max_sequence_length", None)
    print(f"max_position_embeddings: {max_positions}")
    print(f"config_use_cache: {getattr(getattr(model, 'config', None), 'use_cache', None)}")

    encoded = tokenizer(args.prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    print(f"logits_shape: {tuple(logits.shape)}")
    if logits.dim() == 3:
        print("logits_shape_ok: yes")
    else:
        print("logits_shape_ok: no")


if __name__ == "__main__":
    main()
