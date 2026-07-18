from __future__ import annotations

import argparse
from pathlib import Path

from flatness_decoding.decoding import generate_flatness_controlled
from flatness_decoding.io_utils import write_jsonl
from flatness_decoding.metrics import continuation_from_text
from flatness_decoding.model_io import load_causal_lm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="distilgpt2")
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--method",
        choices=[
            "minflat",
            "maxflat",
            "random",
            "midflat",
            "centerflat_q25",
            "centerflat_q50",
            "centerflat_q75",
            "highprob_centerflat",
            "low_spread",
            "mid_spread",
            "high_spread",
            "low_concentration",
            "mid_concentration",
            "high_concentration",
            "thin_tail",
            "mid_tail",
            "heavy_tail",
            "small_gap",
            "mid_gap",
            "large_gap",
            "rank_slope_flat",
            "rank_slope_mid",
            "rank_slope_steep",
            "spiky_top",
            "smooth_top",
            "multi_peak",
        ],
        default="minflat",
    )
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
    continuation_text = continuation_from_text(text, args.prompt)

    output_row = {
        "model": args.model,
        "method": args.method,
        "n_trials": args.n_trials,
        "proposal_temperature": args.proposal_temperature,
        "seed": args.seed,
        "prompt": args.prompt,
        "text": text,
        "continuation_text": continuation_text,
    }

    if args.output:
        write_jsonl(Path(args.output), [output_row])
    if args.token_log:
        write_jsonl(Path(args.token_log), [record.to_dict() for record in records])


if __name__ == "__main__":
    main()
