from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flatness_decoding.io_utils import read_jsonl

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


def load_generation_rows(gen_dir: Path, max_prompts: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(gen_dir.glob("*.jsonl")):
        rows.extend(read_jsonl(path))
    rows.sort(key=lambda r: (str(r.get("prompt_id", "")), int(r.get("prompt_index", 0)), str(r.get("method", ""))))
    return rows[: max_prompts * max(1, len({r.get("method") for r in rows}))]


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


def stable_quantile_index(values: list[tuple[float, int]], quantile: float) -> int:
    if not values:
        raise ValueError("no values")
    ordered = sorted(values, key=lambda item: (item[0], item[1]))
    pos = int(round((len(ordered) - 1) * quantile))
    return ordered[max(0, min(pos, len(ordered) - 1))][1]


def summarize_band(probabilities: torch.Tensor, flatness: torch.Tensor) -> tuple[float, float, float]:
    flat_cpu = flatness.detach().float().cpu()
    prob_cpu = probabilities.detach().float().cpu()
    pairs = [(float(flat_cpu[i].item()), int(i)) for i in range(int(flat_cpu.shape[0]))]
    p25 = stable_quantile_index(pairs, 0.25)
    p75 = stable_quantile_index(pairs, 0.75)
    lo = min(p25, p75)
    hi = max(p25, p75)
    mass_below = float(prob_cpu[:lo].sum().item())
    mass_mid = float(prob_cpu[lo : hi + 1].sum().item())
    mass_above = float(prob_cpu[hi + 1 :].sum().item())
    total = mass_below + mass_mid + mass_above
    if total > 0:
        return mass_below / total, mass_mid / total, mass_above / total
    return 0.0, 0.0, 0.0


def decode_token(tokenizer, token_id: int) -> str:
    text = tokenizer.decode([int(token_id)], skip_special_tokens=True)
    return text.replace("\n", "\\n")


def run_condition(
    target: RunTarget,
    methods: list[str],
    args: argparse.Namespace,
    output_dir: Path,
) -> list[dict[str, object]]:
    import torch

    from flatness_decoding.decoding import current_candidate_distribution_stats, evaluate_candidates_next_flatness
    from flatness_decoding.model_io import load_causal_lm

    model, tokenizer, device = load_causal_lm(
        target.model,
        device_name=args.device,
        dtype_name=args.dtype,
        cache_dir=args.model_cache_dir,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )
    try:
        generations = load_generation_rows(target.generations_dir, args.max_prompts)
        if not generations:
            raise SystemExit(f"no generations found in {target.generations_dir}")
        completed = load_completed_keys(output_dir) if args.resume else set()
        rows_out: list[dict[str, object]] = []
        by_condition_rows: list[dict[str, object]] = []
        for gen_row in generations:
            method = str(gen_row.get("method", ""))
            if method not in methods:
                continue
            prompt_id = str(gen_row.get("prompt_id", ""))
            prompt_index = int(gen_row.get("prompt_index", 0))
            if not prompt_id:
                continue
            prompt = str(gen_row.get("prompt", ""))
            continuation = str(gen_row.get("continuation_text", ""))
            prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
            cont_ids = tokenizer(continuation, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
            max_steps = min(args.max_steps, len(cont_ids))
            for step in range(max_steps):
                key = (target.condition, method, prompt_id, step)
                if key in completed:
                    continue
                started = time.perf_counter()
                prefix_ids = torch.tensor([cont_ids[:step]], dtype=torch.long, device=device)
                if prefix_ids.numel() == 0:
                    input_ids = prompt_ids
                else:
                    input_ids = torch.cat([prompt_ids, prefix_ids], dim=1)
                with torch.inference_mode():
                    current_logits = model(input_ids=input_ids).logits[0, -1, :]
                    candidate_ids = torch.topk(current_logits, k=min(args.top_k, current_logits.shape[-1])).indices.detach().long().cpu()
                    candidate_logprobs, candidate_probs, candidate_ranks = current_candidate_distribution_stats(current_logits, candidate_ids)
                    candidate_flatness = evaluate_candidates_next_flatness(
                        model=model,
                        input_ids=input_ids,
                        candidate_ids=candidate_ids,
                        candidate_eval_batch_size=args.candidate_batch_size,
                        past_key_values=None,
                    ).detach().float().cpu()

                flat_pairs = [(float(candidate_flatness[i].item()), int(i)) for i in range(int(candidate_flatness.shape[0]))]
                order = sorted(range(len(flat_pairs)), key=lambda i: (flat_pairs[i][0], i))
                p10 = order[max(0, min(int(round((len(order) - 1) * 0.10)), len(order) - 1))]
                p25 = order[max(0, min(int(round((len(order) - 1) * 0.25)), len(order) - 1))]
                p50 = order[max(0, min(int(round((len(order) - 1) * 0.50)), len(order) - 1))]
                p75 = order[max(0, min(int(round((len(order) - 1) * 0.75)), len(order) - 1))]
                p90 = order[max(0, min(int(round((len(order) - 1) * 0.90)), len(order) - 1))]
                flat_min = order[0]
                flat_max = order[-1]
                prob_mass_below_p25, prob_mass_p25_p75, prob_mass_above_p75 = summarize_band(candidate_probs, candidate_flatness)
                row = {
                    "condition": target.condition,
                    "model": target.model,
                    "prompt_mode": target.prompt_mode,
                    "method": method,
                    "prompt_id": prompt_id,
                    "prompt_index": prompt_index,
                    "step": step,
                    "top_k": min(args.top_k, current_logits.shape[-1]),
                    "candidate_batch_size": args.candidate_batch_size,
                    "context_input_tokens": int(input_ids.shape[-1]),
                    "flat_min": float(candidate_flatness[flat_min].item()),
                    "flat_p10": float(candidate_flatness[p10].item()),
                    "flat_p25": float(candidate_flatness[p25].item()),
                    "flat_p50": float(candidate_flatness[p50].item()),
                    "flat_p75": float(candidate_flatness[p75].item()),
                    "flat_p90": float(candidate_flatness[p90].item()),
                    "flat_max": float(candidate_flatness[flat_max].item()),
                    "rank_at_flat_min": int(candidate_ranks[flat_min].item()),
                    "rank_at_flat_p50": int(candidate_ranks[p50].item()),
                    "rank_at_flat_p75": int(candidate_ranks[p75].item()),
                    "rank_at_flat_max": int(candidate_ranks[flat_max].item()),
                    "logprob_at_flat_min": float(candidate_logprobs[flat_min].item()),
                    "logprob_at_flat_p50": float(candidate_logprobs[p50].item()),
                    "logprob_at_flat_p75": float(candidate_logprobs[p75].item()),
                    "logprob_at_flat_max": float(candidate_logprobs[flat_max].item()),
                    "token_id_at_flat_min": int(candidate_ids[flat_min].item()),
                    "token_id_at_flat_p50": int(candidate_ids[p50].item()),
                    "token_id_at_flat_p75": int(candidate_ids[p75].item()),
                    "token_id_at_flat_max": int(candidate_ids[flat_max].item()),
                    "token_at_flat_min": decode_token(tokenizer, int(candidate_ids[flat_min].item())),
                    "token_at_flat_p50": decode_token(tokenizer, int(candidate_ids[p50].item())),
                    "token_at_flat_p75": decode_token(tokenizer, int(candidate_ids[p75].item())),
                    "token_at_flat_max": decode_token(tokenizer, int(candidate_ids[flat_max].item())),
                    "prob_mass_below_p25": prob_mass_below_p25,
                    "prob_mass_p25_p75": prob_mass_p25_p75,
                    "prob_mass_above_p75": prob_mass_above_p75,
                    "flatness_range": float(candidate_flatness.max().item() - candidate_flatness.min().item()),
                    "gap_p25_p50": float(candidate_flatness[p50].item() - candidate_flatness[p25].item()),
                    "gap_p50_p75": float(candidate_flatness[p75].item() - candidate_flatness[p50].item()),
                    "runtime_seconds": time.perf_counter() - started,
                }
                rows_out.append(row)
                by_condition_rows.append(row)
                if len(rows_out) >= args.flush_every:
                    write_csv_rows(output_dir / "successor_dispersion_untempered_topk_distribution.csv", rows_out, REPORT_COLUMNS, not (output_dir / "successor_dispersion_untempered_topk_distribution.csv").exists())
                    rows_out.clear()
                    clear_memory()
        if rows_out:
            write_csv_rows(output_dir / "successor_dispersion_untempered_topk_distribution.csv", rows_out, REPORT_COLUMNS, not (output_dir / "successor_dispersion_untempered_topk_distribution.csv").exists())
            rows_out.clear()
        return by_condition_rows
    finally:
        del model
        clear_memory()


def aggregate_by_condition(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["condition"]), str(row["model"]), str(row["prompt_mode"]), str(row["method"]))
        grouped.setdefault(key, []).append(row)
    out: list[dict[str, object]] = []
    for (condition, model, prompt_mode, method), group in sorted(grouped.items()):
        def mean(name: str) -> float:
            vals = [float(r[name]) for r in group]
            return sum(vals) / len(vals)
        def median(name: str) -> float:
            vals = sorted(float(r[name]) for r in group)
            mid = len(vals) // 2
            return vals[mid] if len(vals) % 2 else 0.5 * (vals[mid - 1] + vals[mid])
        out.append({
            "condition": condition,
            "model": model,
            "prompt_mode": prompt_mode,
            "method": method,
            "n_rows": len(group),
            "top_k": int(group[0]["top_k"]),
            "mean_flat_p50": mean("flat_p50"),
            "mean_flat_p75": mean("flat_p75"),
            "mean_rank_at_flat_p50": mean("rank_at_flat_p50"),
            "median_rank_at_flat_p50": median("rank_at_flat_p50"),
            "mean_rank_at_flat_p75": mean("rank_at_flat_p75"),
            "median_rank_at_flat_p75": median("rank_at_flat_p75"),
            "mean_prob_mass_p25_p75": mean("prob_mass_p25_p75"),
            "mean_prob_mass_below_p25": mean("prob_mass_below_p25"),
            "mean_prob_mass_above_p75": mean("prob_mass_above_p75"),
            "mean_flatness_range": mean("flatness_range"),
            "mean_gap_p25_p50": mean("gap_p25_p50"),
            "mean_gap_p50_p75": mean("gap_p50_p75"),
        })
    return out


def write_report(output_dir: Path, condition_rows: list[dict[str, object]], summary_rows: list[dict[str, object]]) -> None:
    report = output_dir / "successor_dispersion_untempered_topk_report.md"
    lines = [
        "# Untempered Successor Dispersion Report",
        "",
        "This report is generated from raw current-logit top-K candidates and is intended to test whether the successor-dispersion median remains tied to the high-probability region.",
        "",
        "## Questions",
        "",
        "1. Does successor-dispersion p50 overlap with high-probability current-token regions?",
        "2. Is p50 low probability even when candidates are selected from raw top-K logits?",
        "3. Does Gemma place high-probability candidates closer to p75 than p50?",
        "4. Does Qwen show a weaker or different overlap pattern?",
        "5. Was earlier q50 behavior mainly caused by proposal-temperature sampling?",
        "6. Are min/max extremes separated by large flatness gaps, or is the distribution narrow?",
        "7. Does p25-p75 probability mass differ across GPT-2, Gemma, and Qwen?",
        "",
        "## Status",
        "",
        f"- distribution rows: {len(condition_rows)}",
        f"- condition summary rows: {len(summary_rows)}",
        "",
        "If no execute run has been performed, these fields remain placeholders only.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    if "qwen_raw" not in conditions or any(c != "qwen_raw" for c in conditions):
        raise SystemExit("execute mode currently supports qwen_raw only; other conditions remain scaffolded")
    all_rows: list[dict[str, object]] = []
    for target in targets:
        all_rows.extend(run_condition(target, methods, args, out_dir))
    summary_rows = aggregate_by_condition(all_rows)
    by_condition_path = out_dir / "successor_dispersion_untempered_topk_by_condition.csv"
    if summary_rows:
        with by_condition_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=CONDITION_SUMMARY_COLUMNS)
            writer.writeheader()
            writer.writerows(summary_rows)
    write_report(out_dir, all_rows, summary_rows)


if __name__ == "__main__":
    main()
