from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from flatness_decoding.decoding import generate_flatness_controlled
from flatness_decoding.io_utils import read_jsonl, write_jsonl
from flatness_decoding.model_io import load_causal_lm

METHOD_CHOICES = [
    "random",
    "minflat",
    "maxflat",
    "midflat",
    "centerflat_q25",
    "centerflat_q50",
    "centerflat_q75",
    "highprob_centerflat",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="distilgpt2")
    parser.add_argument("--prompts", required=True, help="JSONL with a 'prompt' field.")
    parser.add_argument("--methods", nargs="+", choices=METHOD_CHOICES, default=["random", "minflat"])
    parser.add_argument("--n-values", nargs="+", type=int, default=[1, 4, 8, 16])
    parser.add_argument("--proposal-temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--repeat-count", type=int, default=None, help="Generate this many seed replicates when --seeds is not provided.")
    parser.add_argument("--seed-start", type=int, default=0, help="First seed to use when synthesizing a repeat grid.")
    parser.add_argument("--resume", action="store_true", help="Skip generation groups whose output files already look complete.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--output-dir", default="outputs/generations/sweep")
    parser.add_argument("--token-log-dir", default="outputs/token_logs/sweep")
    return parser.parse_args()


def is_group_complete(output_path: Path, token_log_path: Path, expected_rows: int) -> bool:
    if not output_path.exists() or not token_log_path.exists():
        return False
    try:
        gen_rows = read_jsonl(output_path)
        tok_rows = read_jsonl(token_log_path)
    except Exception:
        return False
    return len(gen_rows) == expected_rows and len(tok_rows) > 0


def main() -> None:
    args = parse_args()
    prompts = read_jsonl(args.prompts)
    if args.limit is not None:
        prompts = prompts[: args.limit]

    if args.repeat_count is not None and args.repeat_count <= 0:
        raise SystemExit("--repeat-count must be positive")
    if args.seeds == [0] and args.repeat_count is not None:
        seeds = [args.seed_start + i for i in range(args.repeat_count)]
    else:
        seeds = args.seeds

    model, tokenizer, device = load_causal_lm(args.model, args.device, args.dtype)

    output_dir = Path(args.output_dir)
    token_log_dir = Path(args.token_log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    token_log_dir.mkdir(parents=True, exist_ok=True)

    for method in args.methods:
        for n_trials in args.n_values:
            for repeat_index, seed in enumerate(seeds):
                stem = f"model={args.model.replace('/', '__')}_method={method}_n={n_trials}_temp={args.proposal_temperature}_seed={seed}"
                output_path = output_dir / f"{stem}.jsonl"
                token_log_path = token_log_dir / f"{stem}.tokenlog.jsonl"
                if args.resume and is_group_complete(output_path, token_log_path, len(prompts)):
                    print(f"skip complete group: {output_path}")
                    continue

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
                            "repeat_index": repeat_index,
                            "repeat_count": len(seeds),
                            "prompt": prompt,
                            "text": text,
                        }
                    )
                    for record in records:
                        record_row = record.to_dict()
                        record_row["prompt_id"] = prompt_id
                        record_row["seed"] = seed
                        record_row["repeat_index"] = repeat_index
                        record_row["repeat_count"] = len(seeds)
                        record_row["model"] = args.model
                        token_records.append(record_row)

                write_jsonl(output_path, generations)
                write_jsonl(token_log_path, token_records)


if __name__ == "__main__":
    main()
