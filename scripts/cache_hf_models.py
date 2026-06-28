from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from huggingface_hub import model_info
from transformers import AutoModelForCausalLM, AutoTokenizer

from flatness_decoding.devices import dtype_from_name, resolve_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect or cache Hugging Face causal LM models.")
    p.add_argument("--model-id", action="append", required=True, help="Repeatable model id.")
    p.add_argument("--cache-dir", default="models/hf")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--download", action="store_true")
    p.add_argument("--tokenizer-only", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    p.add_argument("--device", default="cpu")
    p.add_argument("--run-forward", action="store_true")
    p.add_argument("--prompt", default="Hello.")
    return p.parse_args()


def cache_root(cache_dir: str) -> Path:
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root


def local_cache_status(model_id: str, cache_dir: Path) -> str:
    path = cache_dir / f"models--{model_id.replace('/', '--')}"
    return "present" if path.exists() else "absent"


def load_tokenizer(model_id: str, cache_dir: Path, local_files_only: bool, trust_remote_code: bool):
    return AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=str(cache_dir),
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )


def load_model(model_id: str, cache_dir: Path, local_files_only: bool, trust_remote_code: bool, dtype_name: str, device_name: str):
    device = resolve_device(device_name)
    dtype = dtype_from_name(dtype_name, device)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=str(cache_dir),
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()
    return model, device, dtype


def model_metadata(model_id: str) -> None:
    try:
        info = model_info(model_id)
        print(f"hf_gated: {getattr(info, 'gated', None)}")
        print(f"hf_private: {getattr(info, 'private', None)}")
        print(f"hf_pipeline_tag: {getattr(info, 'pipeline_tag', None)}")
        print(f"hf_library_name: {getattr(info, 'library_name', None)}")
        print(f"hf_license: {info.cardData.get('license') if getattr(info, 'cardData', None) else None}")
        print(f"hf_tags: {getattr(info, 'tags', None)}")
    except Exception as exc:  # pragma: no cover
        print(f"warning: unable to fetch hub metadata: {exc}")


def main() -> None:
    args = parse_args()
    root = cache_root(args.cache_dir)
    for model_id in args.model_id:
        print(f"model_id: {model_id}")
        model_metadata(model_id)
        print(f"local_cache_status: {local_cache_status(model_id, root)}")
        tokenizer = None
        try:
            tokenizer = load_tokenizer(model_id, root, args.local_files_only, args.trust_remote_code)
            print(f"tokenizer_class: {tokenizer.__class__.__name__}")
            print(f"vocab_size: {getattr(tokenizer, 'vocab_size', None)}")
            print(f"eos_token_id: {tokenizer.eos_token_id}")
            print(f"pad_token_id: {tokenizer.pad_token_id}")
            print(f"bos_token_id: {tokenizer.bos_token_id}")
            print(f"chat_template_available: {bool(getattr(tokenizer, 'chat_template', None))}")
        except Exception as exc:
            print(f"tokenizer_error: {exc}")
            if args.tokenizer_only:
                continue
        if tokenizer is None and args.tokenizer_only:
            continue
        if not args.download:
            continue
        try:
            model, device, dtype = load_model(
                model_id,
                root,
                args.local_files_only,
                args.trust_remote_code,
                args.dtype,
                args.device,
            )
            print(f"model_class: {model.__class__.__name__}")
            print(f"config_type: {model.config.__class__.__name__}")
            print(f"device: {device}")
            print(f"dtype: {dtype}")
            print(f"max_position_embeddings: {getattr(model.config, 'max_position_embeddings', None)}")
            print(f"model_max_length: {getattr(tokenizer, 'model_max_length', None) if tokenizer is not None else None}")
            if args.run_forward and tokenizer is not None:
                inputs = tokenizer(args.prompt, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs)
                print(f"logits_shape: {tuple(outputs.logits.shape)}")
        except Exception as exc:
            print(f"model_error: {exc}")


if __name__ == "__main__":
    main()
