from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml
from tqdm import tqdm

from flatness_decoding.baselines import generate_baseline
from flatness_decoding.decoding import generate_flatness_controlled
from flatness_decoding.io_utils import read_jsonl, write_jsonl
from flatness_decoding.model_io import load_causal_lm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML experiment-set config.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_baseline_grid(cfg: dict, model, tokenizer, device, prompts: list[dict]) -> None:
    model_name = cfg["model_name"]
    exp_name = cfg["experiment_name"]
    gen_cfg = cfg["generation"]
    baseline_cfg = cfg.get("baselines", {})
    output_dir = Path(cfg["output"]["generations_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = baseline_cfg.get("methods", [])
    seeds = cfg.get("seeds", [0])

    for method in methods:
        for seed in seeds:
            rows: list[dict] = []
            desc = f"baseline {method} seed={seed}"
            for idx, row in enumerate(tqdm(prompts, desc=desc)):
                prompt_id = row.get("id", idx)
                prompt = str(row["prompt"])
                started = time.perf_counter()
                result = generate_baseline(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    method=method,
                    max_new_tokens=int(gen_cfg.get("max_new_tokens", 100)),
                    temperature=float(baseline_cfg.get("temperature", 1.0)),
                    top_p=float(baseline_cfg.get("top_p", 0.9)),
                    top_k=int(baseline_cfg.get("top_k", 50)),
                    device=device,
                    seed=seed + idx,
                )
                elapsed = time.perf_counter() - started
                rows.append(
                    {
                        "experiment_name": exp_name,
                        "prompt_id": prompt_id,
                        "model": model_name,
                        "method": method,
                        "method_family": "baseline",
                        "n_trials": None,
                        "proposal_temperature": None,
                        "baseline_temperature": baseline_cfg.get("temperature", 1.0),
                        "top_p": baseline_cfg.get("top_p", None),
                        "top_k": baseline_cfg.get("top_k", None),
                        "seed": seed,
                        "prompt": prompt,
                        "text": result.text,
                        "num_input_tokens": result.num_input_tokens,
                        "num_output_tokens": result.num_output_tokens,
                        "runtime_seconds": elapsed,
                        "tokens_per_second": result.num_output_tokens / elapsed if elapsed > 0 else None,
                    }
                )

            stem = f"experiment={exp_name}_model={model_name.replace('/', '__')}_method={method}_seed={seed}"
            write_jsonl(output_dir / f"{stem}.jsonl", rows)


def run_n_trial_grid(cfg: dict, model, tokenizer, device, prompts: list[dict]) -> None:
    model_name = cfg["model_name"]
    exp_name = cfg["experiment_name"]
    gen_cfg = cfg["generation"]
    ntrial_cfg = cfg.get("n_trial_methods", {})
    output_dir = Path(cfg["output"]["generations_dir"])
    token_log_dir = Path(cfg["output"]["token_log_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    token_log_dir.mkdir(parents=True, exist_ok=True)

    methods = ntrial_cfg.get("methods", [])
    n_values = ntrial_cfg.get("n_values", [1, 4, 8, 16])
    seeds = cfg.get("seeds", [0])
    proposal_temperature = float(ntrial_cfg.get("proposal_temperature", 1.0))

    for method in methods:
        for n_trials in n_values:
            for seed in seeds:
                rows: list[dict] = []
                token_rows: list[dict] = []
                desc = f"n-trial {method} N={n_trials} seed={seed}"
                for idx, row in enumerate(tqdm(prompts, desc=desc)):
                    prompt_id = row.get("id", idx)
                    prompt = str(row["prompt"])
                    started = time.perf_counter()
                    text, records = generate_flatness_controlled(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        max_new_tokens=int(gen_cfg.get("max_new_tokens", 100)),
                        n_trials=int(n_trials),
                        proposal_temperature=proposal_temperature,
                        method=method,
                        device=device,
                        seed=seed + idx,
                    )
                    elapsed = time.perf_counter() - started
                    num_output_tokens = len(records)
                    rows.append(
                        {
                            "experiment_name": exp_name,
                            "prompt_id": prompt_id,
                            "model": model_name,
                            "method": method,
                            "method_family": "n_trial",
                            "n_trials": int(n_trials),
                            "proposal_temperature": proposal_temperature,
                            "baseline_temperature": None,
                            "top_p": None,
                            "top_k": None,
                            "seed": seed,
                            "prompt": prompt,
                            "text": text,
                            "num_input_tokens": None,
                            "num_output_tokens": num_output_tokens,
                            "runtime_seconds": elapsed,
                            "tokens_per_second": num_output_tokens / elapsed if elapsed > 0 else None,
                        }
                    )
                    for record in records:
                        record_row = record.to_dict()
                        record_row.update(
                            {
                                "experiment_name": exp_name,
                                "prompt_id": prompt_id,
                                "seed": seed,
                                "model": model_name,
                            }
                        )
                        token_rows.append(record_row)

                stem = (
                    f"experiment={exp_name}_model={model_name.replace('/', '__')}_"
                    f"method={method}_n={n_trials}_temp={proposal_temperature}_seed={seed}"
                )
                write_jsonl(output_dir / f"{stem}.jsonl", rows)
                write_jsonl(token_log_dir / f"{stem}.tokenlog.jsonl", token_rows)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    prompts = read_jsonl(cfg["prompts"]["path"])
    limit = cfg.get("prompts", {}).get("limit")
    if limit is not None:
        prompts = prompts[: int(limit)]

    model, tokenizer, device = load_causal_lm(
        model_name=cfg["model_name"],
        device_name=cfg.get("device", "auto"),
        dtype_name=cfg.get("dtype", "auto"),
    )

    run_baseline_grid(cfg, model, tokenizer, device, prompts)
    run_n_trial_grid(cfg, model, tokenizer, device, prompts)


if __name__ == "__main__":
    main()
