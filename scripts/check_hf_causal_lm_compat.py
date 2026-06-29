from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def resolve_device(name: str):
    import torch

    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def dtype_from_name(name: str, device):
    import torch

    if name == "auto":
        return torch.float32 if str(device) == "cpu" else torch.float16
    return getattr(torch, name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Hugging Face causal LM compatibility.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-dir", default="results/analysis")
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--model-cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-gemma", action="store_true")
    parser.add_argument("--run-forward", action="store_true")
    parser.add_argument("--run-chat-template", action="store_true")
    return parser.parse_args()


def load_tokenizer(model_id: str, local_files_only: bool, trust_remote_code: bool, cache_dir: str | None = None):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )


def load_model(
    model_id: str,
    device: torch.device,
    dtype_name: str,
    local_files_only: bool,
    trust_remote_code: bool,
    cache_dir: str | None = None,
):
    import torch
    from transformers import AutoModelForCausalLM

    dtype = dtype_from_name(dtype_name, device)
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
    )


def resolve_prefix(model_id: str, prefix: str | None) -> str:
    if prefix:
        return prefix
    if "qwen" in model_id.lower():
        return "model_tokenizer_compatibility_qwen"
    if "gemma" in model_id.lower():
        return "model_tokenizer_compatibility_gemma"
    return "model_tokenizer_compatibility"


def vocab_size_stats(tokenizer, model) -> dict[str, object]:
    input_embed = getattr(model, "get_input_embeddings", None)
    output_embed = getattr(model, "get_output_embeddings", None)
    input_vocab = None
    output_vocab = None
    if callable(input_embed):
        emb = input_embed()
        if emb is not None and hasattr(emb, "weight"):
            input_vocab = int(emb.weight.shape[0])
    if callable(output_embed):
        head = output_embed()
        if head is not None and hasattr(head, "weight"):
            output_vocab = int(head.weight.shape[0])
    return {
        "tokenizer_vocab_size": getattr(tokenizer, "vocab_size", None),
        "len_tokenizer": len(tokenizer),
        "model_input_embedding_vocab_size": input_vocab,
        "model_output_head_vocab_size": output_vocab,
    }


def smoke_texts() -> list[str]:
    return [
        "Hello.",
        "[ WP ] Write a road trip story.",
        "Continue the following WritingPrompts story prompt as narrative prose.",
    ]


def short_generate(model, tokenizer, device, prompt: str) -> dict[str, object]:
    import torch

    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=32,
            do_sample=False,
        )
    text = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
    return {"prompt": prompt, "generated_text": text}


def main() -> None:
    args = parse_args()
    import torch

    device = resolve_device(args.device)
    dtype = dtype_from_name(args.dtype, device)
    model_id_lower = args.model_id.lower()
    if "gemma" in model_id_lower and not args.allow_gemma:
        raise SystemExit("refusing Gemma diagnostics without --allow-gemma")
    if "qwen" not in model_id_lower and "gemma" not in model_id_lower:
        print("warning: model id is not Qwen/Gemma; diagnostics still run")

    prefix = resolve_prefix(args.model_id, args.prefix)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{prefix}.md"
    json_path = out_dir / f"{prefix}.json"

    tokenizer = load_tokenizer(args.model_id, args.local_files_only, args.trust_remote_code, args.model_cache_dir)
    model = None
    model_error = None
    try:
        model = load_model(args.model_id, device, args.dtype, args.local_files_only, args.trust_remote_code, args.model_cache_dir)
    except Exception as exc:
        model_error = str(exc)

    report: dict[str, object] = {
        "requested_model_name": args.model_id,
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", None),
        "model_config_name_or_path": getattr(getattr(model, "config", None), "name_or_path", None),
        "tokenizer_class": tokenizer.__class__.__name__,
        "model_class": model.__class__.__name__ if model is not None else None,
        "device": str(device),
        "dtype": str(dtype),
        "model_dtype": str(next(model.parameters()).dtype) if model is not None else None,
        "model_cache_dir": args.model_cache_dir,
        "chat_template_exists": getattr(tokenizer, "chat_template", None) is not None,
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "padding_side": getattr(tokenizer, "padding_side", None),
        "model_max_length": getattr(tokenizer, "model_max_length", None),
    }
    if model is not None:
        report.update(vocab_size_stats(tokenizer, model))
    else:
        report.update(
            {
                "tokenizer_vocab_size": getattr(tokenizer, "vocab_size", None),
                "len_tokenizer": len(tokenizer),
                "model_input_embedding_vocab_size": None,
                "model_output_head_vocab_size": None,
            }
        )

    emb_in = report["model_input_embedding_vocab_size"]
    vocab = report["len_tokenizer"]
    report["vocab_compatible"] = emb_in in {None, vocab}
    report["name_or_path_matches_requested"] = args.model_id in {
        str(report["tokenizer_name_or_path"]),
        str(report["model_config_name_or_path"]),
    }
    report["cache_path_warning"] = None
    if report["tokenizer_name_or_path"] and args.model_id not in str(report["tokenizer_name_or_path"]):
        report["cache_path_warning"] = f"tokenizer.name_or_path={report['tokenizer_name_or_path']}"
    if report["model_config_name_or_path"] and args.model_id not in str(report["model_config_name_or_path"]):
        report["cache_path_warning"] = f"model.config.name_or_path={report['model_config_name_or_path']}"
    report["model_error"] = model_error

    smoke = []
    for text in smoke_texts():
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        decoded = tokenizer.decode(encoded["input_ids"][0].tolist(), skip_special_tokens=True)
        smoke.append(
            {
                "text": text,
                "token_count": int(encoded["input_ids"].shape[-1]),
                "roundtrip_ok": decoded == text,
                "decoded": decoded,
            }
        )
    report["smoke_roundtrip"] = smoke
    if model is not None:
        report["generation_smoke"] = short_generate(model, tokenizer, device, "Hello.")
    else:
        report["generation_smoke"] = {"prompt": "Hello.", "generated_text": None}

    if model is not None and getattr(tokenizer, "chat_template", None) is not None and args.run_chat_template:
        chat_messages = [{"role": "user", "content": "Write one short sentence about a traveler at a train station."}]
        chat_prompt = tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
        report["chat_template_smoke"] = short_generate(model, tokenizer, device, chat_prompt)
        report["chat_template_recommendation"] = (
            "Qwen-style chat_template exists; consider a separate chat-template condition."
            if "qwen" in model_id_lower
            else "Chat template exists; test a separate chat-formatted condition before comparing to raw prompts."
        )
    else:
        if report["chat_template_exists"]:
            report["chat_template_recommendation"] = (
                "Qwen-style chat_template exists; consider a separate chat-template condition."
                if "qwen" in model_id_lower
                else "Chat template exists; consider a separate chat-formatted condition."
            )
        else:
            report["chat_template_recommendation"] = "No chat template available."

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        f"# Tokenizer / Model Compatibility",
        "",
        f"- requested_model_name: `{report['requested_model_name']}`",
        f"- tokenizer.name_or_path: `{report['tokenizer_name_or_path']}`",
        f"- model.config.name_or_path: `{report['model_config_name_or_path']}`",
        f"- tokenizer class: `{report['tokenizer_class']}`",
        f"- model class: `{report['model_class']}`",
        f"- tokenizer vocab_size: `{report['tokenizer_vocab_size']}`",
        f"- len(tokenizer): `{report['len_tokenizer']}`",
        f"- model input embedding vocab size: `{report['model_input_embedding_vocab_size']}`",
        f"- model output head vocab size: `{report['model_output_head_vocab_size']}`",
        f"- pad/eos/bos token ids: `{report['pad_token_id']}` / `{report['eos_token_id']}` / `{report['bos_token_id']}`",
        f"- chat_template exists: `{report['chat_template_exists']}`",
        f"- vocab compatible: `{report['vocab_compatible']}`",
        f"- cache path warning: `{report['cache_path_warning']}`",
        f"- model error: `{report['model_error']}`",
        "",
        "## Smoke Tests",
    ]
    for item in smoke:
        lines.extend(
            [
                f"- `{item['text']}` -> roundtrip `{item['roundtrip_ok']}` ({item['token_count']} tokens)",
            ]
        )
    lines.extend(["", "## Generation Smoke", f"- plain causal prompt: `{report['generation_smoke']['prompt']}`", f"- generated text: `{report['generation_smoke']['generated_text']}`"])
    if "chat_template_smoke" in report:
        lines.extend(
            [
                "",
                "## Chat Template Smoke",
                f"- generated text: `{report['chat_template_smoke']['generated_text']}`",
                f"- recommendation: {report['chat_template_recommendation']}",
            ]
        )
    elif report["chat_template_exists"]:
        lines.extend(
            [
                "",
                "## Chat Template",
                f"- recommendation: {report['chat_template_recommendation']}",
            ]
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
