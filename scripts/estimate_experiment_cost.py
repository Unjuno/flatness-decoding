from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from flatness_decoding.io_utils import read_jsonl


PARAMETER_SCALE = {
    # Very rough relative parameter/compute scale for GPT-2 family.
    "distilgpt2": 1.0,
    "gpt2": 2.6,
    "gpt2-medium": 7.8,
}

DEVICE_SPEED_NOTES = {
    "cuda": "fastest if memory fits; batched candidate evaluation helps",
    "mps": "usually usable but can be slower or memory-limited with multiple workers",
    "cpu": "safe but likely slow for full N-sweep experiments",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--unit-seconds",
        type=float,
        default=None,
        help=(
            "Optional measured seconds per relative forward unit. "
            "If omitted, only relative cost units are reported."
        ),
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def count_prompts(cfg: dict) -> int:
    prompt_path = cfg["prompts"]["path"]
    limit = cfg.get("prompts", {}).get("limit")
    try:
        prompts = read_jsonl(prompt_path)
        count = len(prompts)
    except FileNotFoundError:
        count = int(limit) if limit is not None else 0
    if limit is not None:
        count = min(count, int(limit)) if count else int(limit)
    return count


def estimate_cost(cfg: dict) -> dict[str, float | int | str]:
    model_name = str(cfg["model_name"])
    model_scale = PARAMETER_SCALE.get(model_name, 2.6 if "gpt2" in model_name else 1.0)
    prompt_count = count_prompts(cfg)
    seed_count = len(cfg.get("seeds", [0]))
    max_new_tokens = int(cfg.get("generation", {}).get("max_new_tokens", 100))

    baseline_methods = cfg.get("baselines", {}).get("methods", [])
    baseline_count = len(baseline_methods)
    # Beam search is more expensive than greedy, but less dominant than N-trial sweeps here.
    baseline_units = prompt_count * seed_count * max_new_tokens * baseline_count
    if "beam" in baseline_methods:
        num_beams = int(cfg.get("baselines", {}).get("num_beams", 4))
        baseline_units += prompt_count * seed_count * max_new_tokens * max(0, num_beams - 1)

    n_trial_methods = cfg.get("n_trial_methods", {}).get("methods", [])
    n_values = [int(v) for v in cfg.get("n_trial_methods", {}).get("n_values", [])]
    # Each Flatness-N step performs one current-state forward and one batched candidate forward.
    # Treat a batched candidate forward of size N as approximately N relative units.
    n_trial_units = (
        prompt_count
        * seed_count
        * max_new_tokens
        * len(n_trial_methods)
        * sum(1 + n for n in n_values)
    )

    total_units = baseline_units + n_trial_units
    scaled_units = total_units * model_scale

    return {
        "experiment_name": cfg.get("experiment_name", "unknown"),
        "model_name": model_name,
        "model_scale": model_scale,
        "prompt_count": prompt_count,
        "seed_count": seed_count,
        "max_new_tokens": max_new_tokens,
        "baseline_methods": baseline_count,
        "n_trial_methods": len(n_trial_methods),
        "n_values": ",".join(str(n) for n in n_values),
        "baseline_relative_units": baseline_units,
        "n_trial_relative_units": n_trial_units,
        "total_relative_units": total_units,
        "model_scaled_relative_units": scaled_units,
    }


def print_estimate(result: dict[str, float | int | str], unit_seconds: float | None) -> None:
    print("Experiment cost estimate")
    for key, value in result.items():
        print(f"{key}: {value}")
    if unit_seconds is not None:
        seconds = float(result["model_scaled_relative_units"]) * unit_seconds
        print(f"estimated_seconds: {seconds:.1f}")
        print(f"estimated_minutes: {seconds / 60:.1f}")
        print(f"estimated_hours: {seconds / 3600:.2f}")
    print("\nNotes:")
    print("- These are relative compute units, not guaranteed wall-clock time.")
    print("- Sequence length growth, batching efficiency, memory bandwidth, and device choice can change runtime substantially.")
    for device, note in DEVICE_SPEED_NOTES.items():
        print(f"- {device}: {note}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    result = estimate_cost(cfg)
    print_estimate(result, args.unit_seconds)


if __name__ == "__main__":
    main()
