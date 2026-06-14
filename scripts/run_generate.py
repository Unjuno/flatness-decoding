from __future__ import annotations

import argparse
from pathlib import Path

from flatness_decoding.decoding import generate_flatness_controlled
from flatness_decoding.io_utils import write_jsonl
from flatness_decoding.model_io import load_causal_lm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="distilgpt2")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--method", choices=["minflat", "maxflat", "random"], default="minflat")
    parser.add_argument("--n-trials", type=int, default=8)
    parser.add_argument("--proposal-temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--model-cache-dir", default="models/hf")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--token-log", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, tokenizer, device = load_causal_lm(
        args.model,
        args.device,
        args.dtype,
        cache_dir=args.model_cache_dir,
    )

    text, records = generate_flatness_controlled(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        n_trials=args.n_trials,
        proposal_temperature=args.proposal_temperature,
        method=args.method,
        device=device,
        seed=args.seed,
    )

    print(text)

    output_row = {
        "model": args.model,
        "method": args.method,
        "n_trials": args.n_trials,
        "proposal_temperature": args.proposal_temperature,
        "seed": args.seed,
        "prompt": args.prompt,
        "text": text,
    }

    if args.output:
        write_jsonl(Path(args.output), [output_row])
    if args.token_log:
        write_jsonl(Path(args.token_log), [record.to_dict() for record in records])


if __name__ == "__main__":
    main()
