from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CONDITION_SPECS = {
    "gpt2": {
        "model": "gpt2",
        "prompt_mode": "raw_writingprompts",
        "prompt_path": "prompts/writingprompts_fixed_100.jsonl",
        "generations_dir": None,
        "token_log_dir": None,
        "requires_allow_gemma": False,
    },
    "gemma_raw": {
        "model": "google/gemma-3-4b-it",
        "prompt_mode": "raw_writingprompts",
        "prompt_path": "prompts/writingprompts_fixed_100.jsonl",
        "generations_dir": "results/gemma3_4b_it_smoke/generations",
        "token_log_dir": "results/gemma3_4b_it_smoke/token_logs",
        "requires_allow_gemma": True,
    },
    "gemma_wrapped": {
        "model": "google/gemma-3-4b-it",
        "prompt_mode": "explicit_story_continuation_wrapper",
        "prompt_path": "prompts/writingprompts_fixed_100_gemma_wrapped.jsonl",
        "generations_dir": "results/gemma3_4b_it_wrapped/10/generations",
        "token_log_dir": "results/gemma3_4b_it_wrapped/10/token_logs",
        "requires_allow_gemma": True,
    },
    "qwen_raw": {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "prompt_mode": "raw_writingprompts",
        "prompt_path": "prompts/writingprompts_fixed_100.jsonl",
        "generations_dir": "results/qwen2_5_1_5b_instruct_raw/10/generations",
        "token_log_dir": "results/qwen2_5_1_5b_instruct_raw/10/token_logs",
        "requires_allow_gemma": False,
    },
    "qwen_wrapped": {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "prompt_mode": "explicit_story_continuation_wrapper",
        "prompt_path": "prompts/writingprompts_fixed_100_gemma_wrapped.jsonl",
        "generations_dir": "results/qwen2_5_1_5b_instruct_wrapped/10/generations",
        "token_log_dir": "results/qwen2_5_1_5b_instruct_wrapped/10/token_logs",
        "requires_allow_gemma": False,
    },
}


REPORT_COLUMNS = [
    "condition",
    "model",
    "prompt_mode",
    "method",
    "prompt_id",
    "prompt_index",
    "step",
    "top_k",
    "candidate_batch_size",
    "context_input_tokens",
    "flat_min",
    "flat_p10",
    "flat_p25",
    "flat_p50",
    "flat_p75",
    "flat_p90",
    "flat_max",
    "rank_at_flat_min",
    "rank_at_flat_p50",
    "rank_at_flat_p75",
    "rank_at_flat_max",
    "logprob_at_flat_min",
    "logprob_at_flat_p50",
    "logprob_at_flat_p75",
    "logprob_at_flat_max",
    "token_id_at_flat_min",
    "token_id_at_flat_p50",
    "token_id_at_flat_p75",
    "token_id_at_flat_max",
    "token_at_flat_min",
    "token_at_flat_p50",
    "token_at_flat_p75",
    "token_at_flat_max",
    "prob_mass_below_p25",
    "prob_mass_p25_p75",
    "prob_mass_above_p75",
    "flatness_range",
    "gap_p25_p50",
    "gap_p50_p75",
    "runtime_seconds",
]


CONDITION_SUMMARY_COLUMNS = [
    "condition",
    "model",
    "prompt_mode",
    "method",
    "n_rows",
    "top_k",
    "mean_flat_p50",
    "mean_flat_p75",
    "mean_rank_at_flat_p50",
    "median_rank_at_flat_p50",
    "mean_rank_at_flat_p75",
    "median_rank_at_flat_p75",
    "mean_prob_mass_p25_p75",
    "mean_prob_mass_below_p25",
    "mean_prob_mass_above_p75",
    "mean_flatness_range",
    "mean_gap_p25_p50",
    "mean_gap_p50_p75",
]


@dataclass(frozen=True)
class RunTarget:
    condition: str
    model: str
    prompt_mode: str
    prompt_path: Path
    generations_dir: Path
    token_log_dir: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze untempered successor-dispersion distributions from raw top-K logits.",
    )
    p.add_argument("--conditions", action="append", default=[], help="Condition name; repeatable or comma-separated.")
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=16)
    p.add_argument("--max-prompts", type=int, default=10)
    p.add_argument("--methods", action="append", default=["top_p"], help="Method name; repeatable or comma-separated.")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--output-dir", default="results/analysis")
    p.add_argument("--model-cache-dir", default=None)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--candidate-batch-size", type=int, default=1)
    p.add_argument("--flush-every", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true", default=True)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--allow-gemma", action="store_true")
    p.add_argument("--allow-top-k-gt-100", action="store_true")
    p.add_argument("--verbose-candidates", action="store_true")
    return p.parse_args()


def split_csv(values: list[str]) -> list[str]:
    items: list[str] = []
    for value in values:
        items.extend(part.strip() for part in value.split(",") if part.strip())
    return items


def normalize_names(names: list[str]) -> list[str]:
    return sorted(dict.fromkeys(split_csv(names)))


def resolve_targets(conditions: Iterable[str], model_cache_dir: str | None) -> list[RunTarget]:
    targets: list[RunTarget] = []
    for condition in conditions:
        spec = CONDITION_SPECS.get(condition)
        if spec is None:
            raise SystemExit(f"unknown condition: {condition}")
        prompt_path = Path(spec["prompt_path"])
        gen_dir = spec["generations_dir"]
        tok_dir = spec["token_log_dir"]
        if gen_dir is None or tok_dir is None:
            raise SystemExit(
                f"condition {condition} has ambiguous data paths; add an explicit mapping before running."
            )
        targets.append(
            RunTarget(
                condition=condition,
                model=spec["model"],
                prompt_mode=spec["prompt_mode"],
                prompt_path=prompt_path,
                generations_dir=Path(gen_dir),
                token_log_dir=Path(tok_dir),
            )
        )
    return targets


def load_prompts(prompt_path: Path, max_prompts: int) -> list[dict[str, object]]:
    rows = []
    if not prompt_path.exists():
        raise SystemExit(f"prompt file not found: {prompt_path}")
    with prompt_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= max_prompts:
                break
    return rows


def load_completed_keys(output_dir: Path) -> set[tuple[str, str, str, int]]:
    completed: set[tuple[str, str, str, int]] = set()
    csv_path = output_dir / "successor_dispersion_untempered_topk_distribution.csv"
    if not csv_path.exists():
        return completed
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                completed.add(
                    (
                        row["condition"],
                        row["method"],
                        row["prompt_id"],
                        int(row["step"]),
                    )
                )
            except Exception:
                continue
    return completed


def estimate_passes(n_conditions: int, n_methods: int, n_prompts: int, n_steps: int, top_k: int) -> tuple[int, int]:
    return (
        n_conditions * n_methods * n_prompts * n_steps,
        n_conditions * n_methods * n_prompts * n_steps * top_k,
    )


def print_plan(args: argparse.Namespace, targets: list[RunTarget], methods: list[str]) -> None:
    n_curr, n_succ = estimate_passes(len(targets), len(methods), args.max_prompts, args.max_steps, args.top_k)
    print("conditions:", ", ".join(t.condition for t in targets))
    print("methods:", ", ".join(methods))
    print("top_k:", args.top_k)
    print("max_steps:", args.max_steps)
    print("max_prompts:", args.max_prompts)
    print("candidate_batch_size:", args.candidate_batch_size)
    print("estimated_current_forward_passes:", n_curr)
    print("estimated_successor_forward_passes:", n_succ)
    for target in targets:
        print(f"target[{target.condition}]: model={target.model}")
        print(f"  prompt_path={target.prompt_path}")
        print(f"  generations_dir={target.generations_dir}")
        print(f"  token_log_dir={target.token_log_dir}")
    out_dir = Path(args.output_dir)
    print("output_dir:", out_dir)
    print("report_csv:", out_dir / "successor_dispersion_untempered_topk_distribution.csv")
    print("condition_csv:", out_dir / "successor_dispersion_untempered_topk_by_condition.csv")
    print("report_md:", out_dir / "successor_dispersion_untempered_topk_report.md")


def maybe_warn(args: argparse.Namespace, targets: list[RunTarget]) -> None:
    if args.top_k > 100 and not args.allow_top_k_gt_100:
        raise SystemExit("refusing --top-k > 100 without --allow-top-k-gt-100")
    if args.max_steps > 32:
        raise SystemExit("refusing --max-steps > 32")
    if args.max_prompts > 20:
        raise SystemExit("refusing --max-prompts > 20")
    if any(t.condition.startswith("gemma") for t in targets) and not args.allow_gemma:
        raise SystemExit("refusing Gemma conditions without --allow-gemma")
    if any(t.condition.startswith("gemma") for t in targets) and args.top_k > 50:
        print("warning: Gemma condition with top_k > 50 may be slow", file=sys.stderr)


def clear_memory() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def write_csv_rows(path: Path, rows: list[dict[str, object]], columns: list[str], write_header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if path.exists() and not write_header else "w"
    with path.open(mode, encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    conditions = normalize_names(args.conditions or ["qwen_raw"])
    methods = normalize_names(args.methods or ["top_p"])
    targets = resolve_targets(conditions, args.model_cache_dir)
    maybe_warn(args, targets)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print_plan(args, targets, methods)
    if not args.execute:
        return
    import torch

    raise SystemExit(
        "execute mode is intentionally not enabled yet in this scaffold; "
        "use the dry-run plan to validate paths before wiring execution."
    )


if __name__ == "__main__":
    main()
