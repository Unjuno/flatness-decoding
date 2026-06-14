from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from flatness_decoding.decoding import generate_flatness_controlled
from flatness_decoding.io_utils import read_jsonl, write_jsonl
from flatness_decoding.model_io import load_causal_lm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="distilgpt2")
    parser.add_argument("--prompts", required=True, help="JSONL with a 'prompt' field.")
    parser.add_argument("--methods", nargs="+", choices=["random", "minflat", "maxflat"], default=["random", "minflat"])
    parser.add_argument("--n-values", nargs="+", type=int, default=[1, 4, 8, 16])
    parser.add_argument("--proposal-temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--output-dir", default="outputs/generations/sweep")
    parser.add_argument("--token-log-dir", default="outputs/token_logs/sweep")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = read_jsonl(args.prompts)
    if args.limit is not None:
        prompts = prompts[: args.limit]

    model, tokenizer, device = load_causal_lm(args.model, args.device, args.dtype)

    output_dir = Path(args.output_dir)
    token_log_dir = Path(args.token_log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    token_log_dir.mkdir(parents=True, exist_ok=True)

    for method in args.methods:
        for n_trials in args.n_values:
            for seed in args.seeds:
                generations: list[dict] = []
                token_records: list[dict] = []

                iterator = tqdm(prompts, desc=f"{method} N={n_trials} seed={seed}")
                for idx, row in enumerate(iterator):
                    prompt = str(row["prompt"])
                    prompt_id = row.get("id", idx)
                    text, records = generate_flatness_controlled(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        max_new_tokens=args.max_new_tokens,
                        n_trials=n_trials,
                        proposal_temperature=args.proposal_temperature,
                        method=method,
                        device=device,
                        seed=seed + idx,
                    )

                    generations.append(
                        {
                            "prompt_id": prompt_id,
                            "model": args.model,
                            "method": method,
                            "n_trials": n_trials,
                            "proposal_temperature": args.proposal_temperature,
                            "seed": seed,
                            "prompt": prompt,
                            "text": text,
                        }
                    )
                    for record in records:
                        record_row = record.to_dict()
                        record_row["prompt_id"] = prompt_id
                        record_row["seed"] = seed
                        record_row["model"] = args.model
                        token_records.append(record_row)

                stem = f"model={args.model.replace('/', '__')}_method={method}_n={n_trials}_temp={args.proposal_temperature}_seed={seed}"
                write_jsonl(output_dir / f"{stem}.jsonl", generations)
                write_jsonl(token_log_dir / f"{stem}.tokenlog.jsonl", token_records)


if __name__ == "__main__":
    main()
