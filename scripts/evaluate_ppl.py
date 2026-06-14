from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from tqdm import tqdm

from flatness_decoding.io_utils import read_jsonl, write_csv
from flatness_decoding.metrics import analysis_text_from_row
from flatness_decoding.model_io import load_causal_lm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Directory containing generation .jsonl files.")
    parser.add_argument("--model", default="distilgpt2", help="Reference causal LM for scoring.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--model-cache-dir", default="models/hf")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--output", default="results/summary/ppl_metrics.csv")
    return parser.parse_args()


@torch.no_grad()
def conditional_continuation_nll(
    model,
    tokenizer,
    prompt: str,
    full_text: str,
    device: torch.device,
    max_length: int,
) -> tuple[float, int]:
    """Score generated continuation tokens conditioned on the prompt.

    Prompt tokens provide context but are masked out of the loss. Use full_text
    rather than normalized continuation text so spacing/tokenization stays faithful
    to the decoded sequence.
    """
    if full_text == "":
        return 0.0, 0

    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
    encoded = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    if input_ids.shape[1] < 2:
        return 0.0, 0

    prompt_len = min(int(prompt_ids.shape[1]), int(input_ids.shape[1]))
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100

    scored_tokens = int((labels[:, 1:] != -100).sum().detach().cpu().item())
    if scored_tokens == 0:
        return 0.0, 0

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return float(outputs.loss.detach().cpu().item()) * scored_tokens, scored_tokens


def full_text_for_ppl(row: dict) -> str:
    text = str(row.get("text", ""))
    if text:
        return text
    prompt = str(row.get("prompt", ""))
    continuation = analysis_text_from_row(row)
    if not prompt:
        return continuation
    if not continuation:
        return prompt
    if continuation[0].isspace():
        return prompt + continuation
    return prompt + " " + continuation


def main() -> None:
    args = parse_args()
    model, tokenizer, device = load_causal_lm(
        args.model,
        args.device,
        args.dtype,
        cache_dir=args.model_cache_dir,
    )
    rows: list[dict] = []

    for path in sorted(Path(args.input_dir).glob("*.jsonl")):
        generations = read_jsonl(path)
        total_nll = 0.0
        total_tokens = 0
        first = generations[0] if generations else {}
        for row in tqdm(generations, desc=f"PPL {path.name}"):
            prompt = str(row.get("prompt", ""))
            full_text = full_text_for_ppl(row)
            nll, count = conditional_continuation_nll(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                full_text=full_text,
                device=device,
                max_length=args.max_length,
            )
            total_nll += nll
            total_tokens += count
        mean_nll = total_nll / total_tokens if total_tokens > 0 else None
        rows.append(
            {
                "file": str(path),
                "reference_model": args.model,
                "scored_text": "continuation_conditioned_on_prompt",
                "experiment_name": first.get("experiment_name"),
                "model": first.get("model"),
                "method": first.get("method"),
                "method_family": first.get("method_family"),
                "n_trials": first.get("n_trials"),
                "seed": first.get("seed"),
                "num_outputs": len(generations),
                "scored_tokens": total_tokens,
                "mean_nll": mean_nll,
                "ppl": math.exp(mean_nll) if mean_nll is not None and mean_nll < 50 else None,
            }
        )

    write_csv(args.output, rows)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
