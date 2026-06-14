from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from flatness_decoding.io_utils import read_jsonl


STANDARD_METHODS = {"greedy", "sampling", "temperature", "top_p", "top_k"}
N_TRIAL_METHODS = {"random", "minflat", "maxflat"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit(f"SANITY CHECK FAILED: {message}")


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        fail(f"config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    required_top = ["experiment_name", "model_name", "generation", "prompts", "output"]
    for key in required_top:
        if key not in cfg:
            fail(f"missing top-level key: {key}")

    prompt_path = Path(cfg["prompts"]["path"])
    if not prompt_path.exists():
        fail(f"prompt file not found: {prompt_path}")
    prompts = read_jsonl(prompt_path)
    if not prompts:
        fail(f"prompt file has no rows: {prompt_path}")
    for i, row in enumerate(prompts[:10]):
        if "prompt" not in row:
            fail(f"prompt row {i} has no 'prompt' field")

    baseline_methods = set(cfg.get("baselines", {}).get("methods", []))
    unknown_baselines = baseline_methods - STANDARD_METHODS
    if unknown_baselines:
        fail(f"unknown baseline methods: {sorted(unknown_baselines)}")

    n_trial_methods = set(cfg.get("n_trial_methods", {}).get("methods", []))
    unknown_n_trial = n_trial_methods - N_TRIAL_METHODS
    if unknown_n_trial:
        fail(f"unknown N-trial methods: {sorted(unknown_n_trial)}")

    n_values = cfg.get("n_trial_methods", {}).get("n_values", [])
    if n_trial_methods and not n_values:
        fail("N-trial methods specified but n_values is empty")
    if any(int(n) <= 0 for n in n_values):
        fail("all n_values must be positive")

    output = cfg["output"]
    for key in ["generations_dir", "token_log_dir", "summary_csv", "table_csv", "figures_dir"]:
        if key not in output:
            fail(f"missing output key: {key}")

    print("SANITY CHECK PASSED")
    print(f"experiment: {cfg['experiment_name']}")
    print(f"model: {cfg['model_name']}")
    print(f"prompts: {prompt_path} ({len(prompts)} rows before config limit)")
    print(f"baseline methods: {sorted(baseline_methods)}")
    print(f"N-trial methods: {sorted(n_trial_methods)}")
    print(f"N values: {n_values}")


if __name__ == "__main__":
    main()
