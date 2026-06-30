from __future__ import annotations

import argparse
import csv
import gc
import json
import math
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
    "qwen_raw": {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "prompt_mode": "raw_writingprompts",
        "prompt_path": "prompts/writingprompts_fixed_100.jsonl",
        "generations_dir": "results/qwen2_5_1_5b_instruct_raw/10/generations",
        "requires_allow_gemma": False,
    },
    "qwen_wrapped": {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "prompt_mode": "explicit_story_continuation_wrapper",
        "prompt_path": "prompts/writingprompts_fixed_100_gemma_wrapped.jsonl",
        "generations_dir": "results/qwen2_5_1_5b_instruct_wrapped/10/generations",
        "requires_allow_gemma": False,
    },
    "gemma_raw": {
        "model": "google/gemma-3-4b-it",
        "prompt_mode": "raw_writingprompts",
        "prompt_path": "prompts/writingprompts_fixed_100.jsonl",
        "generations_dir": "results/gemma3_4b_it_smoke/generations",
        "requires_allow_gemma": True,
    },
    "gemma_wrapped": {
        "model": "google/gemma-3-4b-it",
        "prompt_mode": "explicit_story_continuation_wrapper",
        "prompt_path": "prompts/writingprompts_fixed_100_gemma_wrapped.jsonl",
        "generations_dir": "results/gemma3_4b_it_wrapped/10/generations",
        "requires_allow_gemma": True,
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
    "mode",
    "top_k",
    "candidate_batch_size",
    "context_input_tokens",
    "candidate_rank",
    "candidate_token_id",
    "candidate_token",
    "candidate_current_logprob",
    "candidate_current_rank",
    "candidate_current_prob",
    "candidate_successor_logit_std",
    "candidate_successor_logit_iqr",
    "candidate_successor_top1_prob",
    "candidate_successor_top5_mass",
    "candidate_successor_top10_mass",
    "candidate_successor_softmax_entropy",
    "candidate_successor_nucleus_size_p90",
    "candidate_successor_tail_mass_after_top50",
    "candidate_successor_gap_1_2",
    "candidate_successor_gap_1_5",
    "candidate_successor_max_gap_top50",
    "candidate_successor_large_gap_count_top50",
    "candidate_successor_markup_mass_top20",
    "candidate_successor_newline_or_bullet_mass_top20",
    "candidate_successor_prompt_overlap_mass_top20",
    "candidate_successor_subword_fragment_mass_top20",
    "runtime_seconds",
]

SUMMARY_COLUMNS = [
    "condition",
    "model",
    "prompt_mode",
    "method",
    "mode",
    "n_rows",
    "top_k",
    "mean_candidate_successor_logit_std",
    "mean_candidate_successor_logit_iqr",
    "mean_candidate_successor_top1_prob",
    "mean_candidate_successor_top5_mass",
    "mean_candidate_successor_top10_mass",
    "mean_candidate_successor_softmax_entropy",
    "mean_candidate_successor_nucleus_size_p90",
    "mean_candidate_successor_tail_mass_after_top50",
    "mean_candidate_successor_gap_1_2",
    "mean_candidate_successor_gap_1_5",
    "mean_candidate_successor_max_gap_top50",
    "mean_candidate_successor_large_gap_count_top50",
    "mean_candidate_successor_markup_mass_top20",
    "mean_candidate_successor_newline_or_bullet_mass_top20",
    "mean_candidate_successor_prompt_overlap_mass_top20",
    "mean_candidate_successor_subword_fragment_mass_top20",
]


@dataclass(frozen=True)
class RunTarget:
    condition: str
    model: str
    prompt_mode: str
    prompt_path: Path
    generations_dir: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze successor distribution morphology proxies.")
    p.add_argument("--conditions", action="append", default=[], help="Repeatable or comma-separated.")
    p.add_argument("--methods", action="append", default=["top_p"], help="Repeatable or comma-separated.")
    p.add_argument("--mode", choices=["raw", "successor"], default="successor")
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=4)
    p.add_argument("--max-prompts", type=int, default=2)
    p.add_argument("--candidate-batch-size", type=int, default=1)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--output-dir", default="results/analysis")
    p.add_argument("--model-cache-dir", default=None)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--flush-every", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true", default=True)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--allow-gemma", action="store_true")
    return p.parse_args()


def split_csv(values: list[str]) -> list[str]:
    items: list[str] = []
    for value in values:
        items.extend(part.strip() for part in value.split(",") if part.strip())
    return items


def normalize(values: list[str]) -> list[str]:
    return sorted(dict.fromkeys(split_csv(values)))


def resolve_targets(conditions: Iterable[str]) -> list[RunTarget]:
    out: list[RunTarget] = []
    for condition in conditions:
        spec = CONDITION_SPECS.get(condition)
        if spec is None:
            raise SystemExit(f"unknown condition: {condition}")
        if spec["generations_dir"] is None:
            raise SystemExit(f"condition {condition} has no generation mapping")
        out.append(
            RunTarget(
                condition=condition,
                model=spec["model"],
                prompt_mode=spec["prompt_mode"],
                prompt_path=Path(spec["prompt_path"]),
                generations_dir=Path(spec["generations_dir"]),
            )
        )
    return out


def load_generation_rows(gen_dir: Path, max_prompts: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(gen_dir.glob("*.jsonl")):
        rows.extend(read_jsonl(path))
    rows.sort(key=lambda r: (str(r.get("prompt_id", "")), int(r.get("prompt_index", 0)), str(r.get("method", ""))))
    return rows[: max_prompts * max(1, len({r.get("method") for r in rows}))]


def load_completed_keys(out_path: Path) -> set[tuple[str, str, str, int, str]]:
    if not out_path.exists():
        return set()
    seen: set[tuple[str, str, str, int, str]] = set()
    with out_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                seen.add((row["condition"], row["method"], row["prompt_id"], int(row["step"]), row["mode"]))
            except Exception:
                continue
    return seen


def resolve_device(name: str):
    import torch

    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def dtype_from_name(name: str, device):
    import torch

    if name == "auto":
        return torch.float32 if str(device) == "cpu" else torch.float16
    return getattr(torch, name)


def print_plan(args: argparse.Namespace, targets: list[RunTarget], methods: list[str]) -> None:
    curr = len(targets) * len(methods) * args.max_prompts * args.max_steps
    succ = curr * args.top_k
    print("conditions:", ", ".join(t.condition for t in targets))
    print("methods:", ", ".join(methods))
    print("mode:", args.mode)
    print("top_k:", args.top_k)
    print("max_steps:", args.max_steps)
    print("max_prompts:", args.max_prompts)
    print("candidate_batch_size:", args.candidate_batch_size)
    print("estimated_current_forward_passes:", curr)
    print("estimated_successor_forward_passes:", succ)
    for t in targets:
        print(f"target[{t.condition}]: model={t.model}")
        print(f"  generations_dir={t.generations_dir}")
        print(f"  prompt_path={t.prompt_path}")


def maybe_gate(args: argparse.Namespace, targets: list[RunTarget]) -> None:
    if any(t.condition.startswith("gemma") for t in targets) and not args.allow_gemma:
        raise SystemExit("refusing Gemma conditions without --allow-gemma")
    if args.top_k > 100:
        raise SystemExit("refusing --top-k > 100")
    if args.max_steps > 32:
        raise SystemExit("refusing --max-steps > 32")
    if args.max_prompts > 20:
        raise SystemExit("refusing --max-prompts > 20")


def safe_decode(tokenizer, token_id: int) -> str:
    text = tokenizer.decode([int(token_id)], skip_special_tokens=True)
    return text.replace("\n", "\\n")


def softmax_stats(logits):
    import torch

    probs = torch.softmax(logits.float(), dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, stable=True)
    sorted_logits = logits[sorted_idx]
    logit_std = float(torch.std(logits.float(), unbiased=False).item())
    q25, q75 = torch.quantile(logits.float(), torch.tensor([0.25, 0.75], device=logits.device)).tolist()
    logit_iqr = float(q75 - q25)
    top1_prob = float(sorted_probs[0].item())
    top5_mass = float(sorted_probs[:5].sum().item())
    top10_mass = float(sorted_probs[:10].sum().item())
    entropy = float((-(probs * torch.log(probs.clamp_min(1e-12))).sum()).item())
    csum = torch.cumsum(sorted_probs, dim=0)
    nucleus_size_p90 = int((csum < 0.9).sum().item()) + 1
    tail_mass_after_top50 = float((1.0 - sorted_probs[:50].sum().item()) if sorted_probs.numel() > 50 else 0.0)
    gap_1_2 = float((sorted_logits[0] - sorted_logits[1]).item()) if sorted_logits.numel() > 1 else 0.0
    gap_1_5 = float((sorted_logits[0] - sorted_logits[4]).item()) if sorted_logits.numel() > 4 else 0.0
    gaps = (sorted_logits[:-1] - sorted_logits[1:])[:50]
    max_gap_top50 = float(gaps.max().item()) if gaps.numel() else 0.0
    large_gap_count_top50 = int((gaps > 1.0).sum().item()) if gaps.numel() else 0
    return {
        "logit_std": logit_std,
        "logit_iqr": logit_iqr,
        "top1_prob": top1_prob,
        "top5_mass": top5_mass,
        "top10_mass": top10_mass,
        "softmax_entropy": entropy,
        "nucleus_size_p90": nucleus_size_p90,
        "tail_mass_after_top50": tail_mass_after_top50,
        "gap_1_2": gap_1_2,
        "gap_1_5": gap_1_5,
        "max_gap_top50": max_gap_top50,
        "large_gap_count_top50": large_gap_count_top50,
        "sorted_idx": sorted_idx,
        "sorted_probs": sorted_probs,
    }


def token_type_masses(tokenizer, top_ids, top_probs, prompt_text: str) -> dict[str, float]:
    import re

    prompt_words = set(re.findall(r"[A-Za-z]{4,}", prompt_text.lower()))
    markup = 0.0
    newline_bullet = 0.0
    prompt_overlap = 0.0
    subword = 0.0
    total = float(top_probs.sum().item()) if top_probs.numel() else 0.0
    for token_id, prob in zip(top_ids.tolist(), top_probs.tolist()):
        tok = tokenizer.decode([int(token_id)], skip_special_tokens=True)
        low = tok.lower()
        if any(x in low for x in ["<", ">", "</", "<br", "<p", "<div", "```", "###"]):
            markup += prob
        if low.startswith(("\n", "-", "*", "•", "1.", "2.", "3.", "[", "•")) or "\n" in tok:
            newline_bullet += prob
        words = set(re.findall(r"[A-Za-z]{4,}", low))
        if words and prompt_words.intersection(words):
            prompt_overlap += prob
        if any(sym in tok for sym in ["##", "▁", "@@", "Ġ", "Ċ"]) or len(tok.strip()) <= 2:
            subword += prob
    if total > 0:
        return {
            "markup_mass_top20": markup / total,
            "newline_or_bullet_mass_top20": newline_bullet / total,
            "prompt_overlap_mass_top20": prompt_overlap / total,
            "subword_fragment_mass_top20": subword / total,
        }
    return {
        "markup_mass_top20": 0.0,
        "newline_or_bullet_mass_top20": 0.0,
        "prompt_overlap_mass_top20": 0.0,
        "subword_fragment_mass_top20": 0.0,
    }


def write_rows(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    write_header = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if not write_header else "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        return []
    import pandas as pd

    df = pd.DataFrame(rows)
    group_cols = ["condition", "model", "prompt_mode", "method", "mode"]
    summary = df.groupby(group_cols, dropna=False).agg(
        n_rows=("method", "size"),
        top_k=("top_k", "first"),
        mean_candidate_successor_logit_std=("candidate_successor_logit_std", "mean"),
        mean_candidate_successor_logit_iqr=("candidate_successor_logit_iqr", "mean"),
        mean_candidate_successor_top1_prob=("candidate_successor_top1_prob", "mean"),
        mean_candidate_successor_top5_mass=("candidate_successor_top5_mass", "mean"),
        mean_candidate_successor_top10_mass=("candidate_successor_top10_mass", "mean"),
        mean_candidate_successor_softmax_entropy=("candidate_successor_softmax_entropy", "mean"),
        mean_candidate_successor_nucleus_size_p90=("candidate_successor_nucleus_size_p90", "mean"),
        mean_candidate_successor_tail_mass_after_top50=("candidate_successor_tail_mass_after_top50", "mean"),
        mean_candidate_successor_gap_1_2=("candidate_successor_gap_1_2", "mean"),
        mean_candidate_successor_gap_1_5=("candidate_successor_gap_1_5", "mean"),
        mean_candidate_successor_max_gap_top50=("candidate_successor_max_gap_top50", "mean"),
        mean_candidate_successor_large_gap_count_top50=("candidate_successor_large_gap_count_top50", "mean"),
        mean_candidate_successor_markup_mass_top20=("candidate_successor_markup_mass_top20", "mean"),
        mean_candidate_successor_newline_or_bullet_mass_top20=("candidate_successor_newline_or_bullet_mass_top20", "mean"),
        mean_candidate_successor_prompt_overlap_mass_top20=("candidate_successor_prompt_overlap_mass_top20", "mean"),
        mean_candidate_successor_subword_fragment_mass_top20=("candidate_successor_subword_fragment_mass_top20", "mean"),
    ).reset_index()
    return summary.to_dict(orient="records")


def write_report(out_dir: Path, rows: list[dict[str, object]], summary_rows: list[dict[str, object]]) -> None:
    report = out_dir / "successor_distribution_morphology_report.md"
    lines = [
        "# Successor Distribution Morphology",
        "",
        "This report summarizes lightweight morphology proxies for successor distributions.",
        "These are diagnostic proxies, not quality metrics.",
        "",
        "## Morphology-to-mode map",
        "",
        "- direct narrative prose: moderate spread, moderate entropy, modest tail mass, low markup / prompt overlap / fragment mass",
        "- generic atmospheric prose: moderate-to-high spread with elevated tail mass and broad continuation support",
        "- quiz/classification/options: strong prompt overlap, instruction-like lexical recycling, and sometimes high top1 concentration",
        "- markup/HTML: elevated markup mass and punctuation-like concentration in the top tokens",
        "- prompt-marker recycling: elevated prompt overlap and prompt-marker-like token mass",
        "- token-fragment collapse: elevated subword-fragment mass, high concentration, or pathological top-1 dominance",
        "- repetition/loop: high concentration with low effective tail diversity or repeated token families",
        "- semantic drift: broad tail, weak prompt overlap, and unstable rank/gap structure",
        "",
        "## Status",
        "",
        f"- rows: {len(rows)}",
        f"- summary rows: {len(summary_rows)}",
        "- p50-style rank questions are out of scope here; this script focuses on morphology proxies.",
        "",
        "The recommended initial execute slice is `qwen_raw`, `top_p`, `top_k=50`, `max_prompts=2`, `max_steps=4`.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def clear_memory():
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def run_condition(target: RunTarget, methods: list[str], args: argparse.Namespace, out_dir: Path) -> list[dict[str, object]]:
    import torch
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
        rows = load_generation_rows(target.generations_dir, args.max_prompts)
        completed = load_completed_keys(out_dir / "successor_distribution_morphology_features.csv") if args.resume else set()
        output_rows: list[dict[str, object]] = []
        collected_rows: list[dict[str, object]] = []
        for gen in rows:
            method = str(gen.get("method", ""))
            if method not in methods:
                continue
            prompt_id = str(gen.get("prompt_id", ""))
            prompt_index = int(gen.get("prompt_index", 0))
            prompt = str(gen.get("prompt", ""))
            continuation = str(gen.get("continuation_text", ""))
            prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
            cont_ids = tokenizer(continuation, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
            max_steps = min(args.max_steps, len(cont_ids))
            for step in range(max_steps):
                prefix = torch.tensor([cont_ids[:step]], dtype=torch.long, device=device)
                input_ids = prompt_ids if prefix.numel() == 0 else torch.cat([prompt_ids, prefix], dim=1)
                with torch.inference_mode():
                    current_logits = model(input_ids=input_ids).logits[0, -1, :]
                    top_k = min(args.top_k, int(current_logits.shape[-1]))
                    top = torch.topk(current_logits, k=top_k)
                    candidate_ids = top.indices.detach().cpu()
                    current_logprobs = torch.log_softmax(current_logits.float(), dim=-1).detach().cpu()
                    current_probs = torch.softmax(current_logits.float(), dim=-1).detach().cpu()
                    if args.mode == "raw":
                        feat = softmax_stats(current_logits)
                        top_probs = feat["sorted_probs"][:20].detach().cpu()
                        top_ids = feat["sorted_idx"][:20].detach().cpu()
                        token_masses = token_type_masses(tokenizer, top_ids, top_probs, prompt)
                        row = {
                            "condition": target.condition,
                            "model": target.model,
                            "prompt_mode": target.prompt_mode,
                            "method": method,
                            "prompt_id": prompt_id,
                            "prompt_index": prompt_index,
                            "step": step,
                            "mode": "raw_current",
                            "top_k": top_k,
                            "candidate_batch_size": args.candidate_batch_size,
                            "context_input_tokens": int(input_ids.shape[-1]),
                            "candidate_rank": None,
                            "candidate_token_id": None,
                            "candidate_token": None,
                            "candidate_current_logprob": None,
                            "candidate_current_rank": None,
                            "candidate_current_prob": None,
                            "candidate_successor_logit_std": feat["logit_std"],
                            "candidate_successor_logit_iqr": feat["logit_iqr"],
                            "candidate_successor_top1_prob": feat["top1_prob"],
                            "candidate_successor_top5_mass": feat["top5_mass"],
                            "candidate_successor_top10_mass": feat["top10_mass"],
                            "candidate_successor_softmax_entropy": feat["softmax_entropy"],
                            "candidate_successor_nucleus_size_p90": feat["nucleus_size_p90"],
                            "candidate_successor_tail_mass_after_top50": feat["tail_mass_after_top50"],
                            "candidate_successor_gap_1_2": feat["gap_1_2"],
                            "candidate_successor_gap_1_5": feat["gap_1_5"],
                            "candidate_successor_max_gap_top50": feat["max_gap_top50"],
                            "candidate_successor_large_gap_count_top50": feat["large_gap_count_top50"],
                            "candidate_successor_markup_mass_top20": token_masses["markup_mass_top20"],
                            "candidate_successor_newline_or_bullet_mass_top20": token_masses["newline_or_bullet_mass_top20"],
                            "candidate_successor_prompt_overlap_mass_top20": token_masses["prompt_overlap_mass_top20"],
                            "candidate_successor_subword_fragment_mass_top20": token_masses["subword_fragment_mass_top20"],
                            "runtime_seconds": 0.0,
                        }
                        output_rows.append(row)
                        collected_rows.append(row)
                        continue
                    candidate_flatness = []
                    for cid in candidate_ids.tolist():
                        cand = torch.tensor([[cid]], device=device)
                        successor_ids = torch.cat([input_ids, cand], dim=1)
                        succ_logits = model(input_ids=successor_ids).logits[0, -1, :]
                        candidate_flatness.append(float(torch.std(succ_logits.float(), unbiased=False).item()))
                    candidate_flatness_t = torch.tensor(candidate_flatness)
                    best_idx = int(torch.argmin(candidate_flatness_t).item())
                    candidate_probs = current_probs[candidate_ids]
                    candidate_logprobs = current_logprobs[candidate_ids]
                    for rank, cid in enumerate(candidate_ids.tolist(), start=1):
                        if (target.condition, method, prompt_id, step, "successor") in completed:
                            break
                        started = time.perf_counter()
                        cand = torch.tensor([[cid]], device=device)
                        successor_ids = torch.cat([input_ids, cand], dim=1)
                        succ_logits = model(input_ids=successor_ids).logits[0, -1, :]
                        feat = softmax_stats(succ_logits)
                        top_ids = feat["sorted_idx"][:20].detach().cpu()
                        top_probs = feat["sorted_probs"][:20].detach().cpu()
                        token_masses = token_type_masses(tokenizer, top_ids, top_probs, prompt)
                        row = {
                            "condition": target.condition,
                            "model": target.model,
                            "prompt_mode": target.prompt_mode,
                            "method": method,
                            "prompt_id": prompt_id,
                            "prompt_index": prompt_index,
                            "step": step,
                            "mode": "successor",
                            "top_k": top_k,
                            "candidate_batch_size": args.candidate_batch_size,
                            "context_input_tokens": int(input_ids.shape[-1]),
                            "candidate_rank": rank,
                            "candidate_token_id": cid,
                            "candidate_token": safe_decode(tokenizer, cid),
                            "candidate_current_logprob": float(candidate_logprobs[rank - 1].item()),
                            "candidate_current_rank": int((current_logits > current_logits[int(cid)]).sum().item()) + 1,
                            "candidate_current_prob": float(candidate_probs[rank - 1].item()),
                            "candidate_successor_logit_std": feat["logit_std"],
                            "candidate_successor_logit_iqr": feat["logit_iqr"],
                            "candidate_successor_top1_prob": feat["top1_prob"],
                            "candidate_successor_top5_mass": feat["top5_mass"],
                            "candidate_successor_top10_mass": feat["top10_mass"],
                            "candidate_successor_softmax_entropy": feat["softmax_entropy"],
                            "candidate_successor_nucleus_size_p90": feat["nucleus_size_p90"],
                            "candidate_successor_tail_mass_after_top50": feat["tail_mass_after_top50"],
                            "candidate_successor_gap_1_2": feat["gap_1_2"],
                            "candidate_successor_gap_1_5": feat["gap_1_5"],
                            "candidate_successor_max_gap_top50": feat["max_gap_top50"],
                            "candidate_successor_large_gap_count_top50": feat["large_gap_count_top50"],
                            "candidate_successor_markup_mass_top20": token_masses["markup_mass_top20"],
                            "candidate_successor_newline_or_bullet_mass_top20": token_masses["newline_or_bullet_mass_top20"],
                            "candidate_successor_prompt_overlap_mass_top20": token_masses["prompt_overlap_mass_top20"],
                            "candidate_successor_subword_fragment_mass_top20": token_masses["subword_fragment_mass_top20"],
                            "runtime_seconds": time.perf_counter() - started,
                        }
                        output_rows.append(row)
                        collected_rows.append(row)
                        if len(output_rows) >= args.flush_every:
                            write_rows(out_dir / "successor_distribution_morphology_features.csv", output_rows, REPORT_COLUMNS)
                            output_rows.clear()
                            clear_memory()
                    break
        if output_rows:
            write_rows(out_dir / "successor_distribution_morphology_features.csv", output_rows, REPORT_COLUMNS)
            output_rows.clear()
        return collected_rows
    finally:
        del model
        clear_memory()


def main() -> None:
    args = parse_args()
    conditions = normalize(args.conditions or ["qwen_raw"])
    methods = normalize(args.methods or ["top_p"])
    targets = resolve_targets(conditions)
    maybe_gate(args, targets)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print_plan(args, targets, methods)
    if not args.execute:
        return
    all_rows: list[dict[str, object]] = []
    for target in targets:
        all_rows.extend(run_condition(target, methods, args, out_dir))
    summary_rows = summarize_rows(all_rows)
    if summary_rows:
        write_rows(out_dir / "successor_distribution_morphology_by_condition.csv", summary_rows, SUMMARY_COLUMNS)
    write_report(out_dir, all_rows, summary_rows)


if __name__ == "__main__":
    main()
