from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flatness_decoding.io_utils import read_jsonl


OUTPUT_DIR_DEFAULT = ROOT / "results" / "analysis"

CONDITION_SPECS: dict[str, dict[str, Any]] = {
    "gpt2": {
        "model_name": "gpt2",
        "prompt_mode": "raw_writingprompts",
        "prompt_path": ROOT / "prompts" / "writingprompts_fixed_100.jsonl",
        "generation_dir": None,
        "token_log_dir": None,
        "generation_candidates": [
            ROOT / "results" / "gpt2_writingprompts" / "generations",
            ROOT / "results" / "comparison_gpt2_writingprompts" / "generations",
            ROOT / "results" / "gpt2_medium_micro" / "generations",
            ROOT / "results" / "gpt2_lite" / "generations",
            ROOT / "results" / "distilgpt2_writingprompts" / "generations",
        ],
        "notes": "GPT-2 path is ambiguous; set a local path explicitly before execution.",
    },
    "distilgpt2_raw": {
        "model_name": "distilgpt2",
        "prompt_mode": "raw_writingprompts",
        "prompt_path": ROOT / "prompts" / "writingprompts_fixed_100.jsonl",
        "generation_dir": ROOT / "outputs" / "generations" / "comparison_distilgpt2_writingprompts",
        "token_log_dir": ROOT / "outputs" / "token_logs" / "comparison_distilgpt2_writingprompts",
        "notes": "GPT small baseline using the existing distilgpt2 writingprompts outputs.",
    },
    "gemma_raw": {
        "model_name": "google/gemma-3-4b-it",
        "prompt_mode": "raw_writingprompts",
        "prompt_path": ROOT / "prompts" / "writingprompts_fixed_100.jsonl",
        "generation_dir": ROOT / "results" / "gemma3_4b_it_smoke" / "generations",
        "token_log_dir": None,
        "notes": "Raw writingprompts stress test.",
        "requires_allow_gemma": True,
    },
    "gemma2_raw": {
        "model_name": "google/gemma-2-2b",
        "prompt_mode": "raw_writingprompts",
        "prompt_path": ROOT / "prompts" / "writingprompts_fixed_100.jsonl",
        "generation_dir": ROOT / "results" / "gemma2_2b_raw" / "10" / "generations",
        "token_log_dir": ROOT / "results" / "gemma2_2b_raw" / "10" / "token_logs",
        "notes": "Gemma 2 2B raw-writingprompts probe.",
        "requires_allow_gemma": True,
    },
    "gemma_wrapped": {
        "model_name": "google/gemma-3-4b-it",
        "prompt_mode": "explicit_story_continuation_wrapper",
        "prompt_path": ROOT / "prompts" / "writingprompts_fixed_100_gemma_wrapped.jsonl",
        "generation_dir": ROOT / "results" / "gemma3_4b_it_wrapped" / "10" / "generations",
        "token_log_dir": ROOT / "results" / "gemma3_4b_it_wrapped" / "10" / "token_logs",
        "notes": "Default to wrapped10; 20x20 must be provided explicitly later.",
        "requires_allow_gemma": True,
    },
    "gemma2_wrapped": {
        "model_name": "google/gemma-2-2b",
        "prompt_mode": "explicit_story_continuation_wrapper",
        "prompt_path": ROOT / "prompts" / "writingprompts_fixed_100_gemma_wrapped.jsonl",
        "generation_dir": ROOT / "modern_model_probes" / "gemma2_2b_wrapped_10" / "generations",
        "token_log_dir": ROOT / "modern_model_probes" / "gemma2_2b_wrapped_10" / "token_logs",
        "notes": "Gemma 2 2B wrapped-writingprompts probe.",
        "requires_allow_gemma": True,
    },
    "qwen_raw": {
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "prompt_mode": "raw_writingprompts",
        "prompt_path": ROOT / "prompts" / "writingprompts_fixed_100.jsonl",
        "generation_dir": ROOT / "results" / "qwen2_5_1_5b_instruct_raw" / "10" / "generations",
        "token_log_dir": ROOT / "results" / "qwen2_5_1_5b_instruct_raw" / "10" / "token_logs",
        "notes": "Plain causal prompting stress test.",
    },
    "qwen25_05b_raw": {
        "model_name": "Qwen/Qwen2.5-0.5B",
        "prompt_mode": "raw_writingprompts",
        "prompt_path": ROOT / "prompts" / "writingprompts_fixed_100.jsonl",
        "generation_dir": ROOT / "outputs" / "generations" / "comparison_qwen25_05b_micro",
        "token_log_dir": ROOT / "outputs" / "token_logs" / "comparison_qwen25_05b_micro",
        "notes": "Qwen2.5 0.5B raw-writingprompts micro probe.",
    },
    "qwen_wrapped": {
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "prompt_mode": "explicit_story_continuation_wrapper",
        "prompt_path": ROOT / "prompts" / "writingprompts_fixed_100_gemma_wrapped.jsonl",
        "generation_dir": ROOT / "results" / "qwen2_5_1_5b_instruct_wrapped" / "10" / "generations",
        "token_log_dir": ROOT / "results" / "qwen2_5_1_5b_instruct_wrapped" / "10" / "token_logs",
        "notes": "Plain causal prompting stress test; not best-faith chat behavior.",
    },
    "qwen25_05b_wrapped": {
        "model_name": "Qwen/Qwen2.5-0.5B",
        "prompt_mode": "explicit_story_continuation_wrapper",
        "prompt_path": ROOT / "prompts" / "writingprompts_fixed_100_gemma_wrapped.jsonl",
        "generation_dir": ROOT / "outputs" / "generations" / "comparison_qwen25_05b_micro",
        "token_log_dir": ROOT / "outputs" / "token_logs" / "comparison_qwen25_05b_micro",
        "notes": "Qwen2.5 0.5B wrapped-writingprompts micro probe.",
    },
    "qwen_chat_wrapped": {
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "prompt_mode": "chat_template_story_continuation",
        "prompt_path": ROOT / "prompts" / "writingprompts_fixed_100_gemma_wrapped.jsonl",
        "generation_dir": None,
        "token_log_dir": None,
        "notes": "Best-faith chat-template condition when data becomes available.",
    },
}

PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "conditions": ["qwen_raw"],
        "methods": ["top_p"],
        "top_k": 20,
        "max_prompts": 2,
        "max_steps": 4,
        "repeat_count": 1,
        "seed_start": 0,
        "seed_count": 1,
        "bootstrap_samples": 1000,
    },
    "qwen_plain": {
        "conditions": ["qwen_raw", "qwen_wrapped"],
        "methods": ["top_p", "top_k", "minflat", "maxflat", "centerflat_q50", "centerflat_q75", "highprob_centerflat"],
        "top_k": 50,
        "max_prompts": 10,
        "max_steps": 16,
        "repeat_count": 1,
        "seed_start": 0,
        "seed_count": 1,
        "bootstrap_samples": 1000,
    },
    "gpt2_main": {
        "conditions": ["gpt2"],
        "methods": ["top_p", "top_k", "random", "minflat", "maxflat", "centerflat_q50", "centerflat_q75", "highprob_centerflat"],
        "top_k": 50,
        "max_prompts": 100,
        "max_steps": 32,
        "repeat_count": 1,
        "seed_start": 0,
        "seed_count": 1,
        "bootstrap_samples": 1000,
    },
    "gemma_wrapped10": {
        "conditions": ["gemma_wrapped"],
        "methods": ["top_p", "top_k", "random", "minflat", "maxflat", "centerflat_q50", "centerflat_q75", "highprob_centerflat"],
        "top_k": 50,
        "max_prompts": 10,
        "max_steps": 32,
        "repeat_count": 1,
        "seed_start": 0,
        "seed_count": 1,
        "bootstrap_samples": 1000,
    },
    "modern_compare": {
        "conditions": ["gemma_wrapped", "qwen_wrapped", "qwen_chat_wrapped"],
        "methods": ["top_p", "top_k", "minflat", "maxflat", "centerflat_q50", "centerflat_q75", "highprob_centerflat"],
        "top_k": 50,
        "max_prompts": 10,
        "max_steps": 32,
        "repeat_count": 1,
        "seed_start": 0,
        "seed_count": 1,
        "bootstrap_samples": 1000,
    },
    "morphology_20x20_top20": {
        "conditions": ["qwen_raw"],
        "methods": ["top_p"],
        "top_k": 20,
        "max_prompts": 20,
        "max_steps": 20,
        "repeat_count": 20,
        "seed_start": 0,
        "seed_count": 20,
        "bootstrap_samples": 1000,
    },
}

MODE_LABELS = [
    "direct_story_prose",
    "generic_atmospheric_prose",
    "title_or_character_sheet",
    "commentary_or_analysis",
    "quiz_or_options",
    "markup_or_html",
    "prompt_marker_recycling",
    "token_fragment_collapse",
    "repetition_or_loop",
    "semantic_drift_possible",
]

FEATURE_COLUMNS = [
    "condition",
    "model_name",
    "prompt_mode",
    "method",
    "prompt_id",
    "prompt_index",
    "seed",
    "repeat_index",
    "requested_repeat_count",
    "step",
    "top_k",
    "candidate_batch_size",
    "chosen_token_id",
    "chosen_token",
    "context_input_tokens",
    "output_mode_label",
    "candidate_rank",
    "candidate_token_id",
    "candidate_token",
    "candidate_current_logprob",
    "candidate_current_rank",
    "candidate_current_prob",
    "logit_std",
    "logit_iqr",
    "logit_mad",
    "logit_range",
    "top1_prob",
    "top5_mass",
    "top10_mass",
    "softmax_entropy",
    "effective_vocab_size",
    "tail_mass_after_top50",
    "tail_mass_after_top100",
    "nucleus_size_p90",
    "nucleus_size_p95",
    "rank_prob_slope_top50",
    "gap_1_2",
    "gap_1_5",
    "gap_5_20",
    "max_gap_top50",
    "large_gap_count_top50",
    "gap_std_top50",
    "markup_mass_top20",
    "newline_or_bullet_mass_top20",
    "quote_mass_top20",
    "digit_mass_top20",
    "subword_fragment_mass_top20",
    "html_marker_mass_top20",
    "title_marker_mass_top20",
    "choice_marker_mass_top20",
    "prompt_overlap_mass_top20",
    "prompt_marker_mass_top20",
    "prompt_entity_overlap_mass_top20",
    "recent_overlap_mass_top20",
    "rank_at_flat_p50",
    "rank_at_flat_p75",
    "prob_mass_p25_p75",
    "spearman_flatness_probability_rank",
    "runtime_seconds",
]

SUMMARY_GROUPS = {
    "by_condition": ["condition", "model_name", "prompt_mode", "method"],
    "by_method": ["model_name", "method"],
    "by_prompt": ["condition", "model_name", "prompt_mode", "method", "prompt_id"],
    "by_mode": ["condition", "model_name", "prompt_mode", "method", "output_mode_label"],
}


@dataclass(frozen=True)
class ConditionMeta:
    condition: str
    model_name: str
    prompt_mode: str
    prompt_path: Path
    generation_dir: Path | None
    token_log_dir: Path | None
    notes: str
    requires_allow_gemma: bool = False


def parse_csv_like(values: list[str]) -> list[str]:
    parts: list[str] = []
    for value in values:
        parts.extend(x.strip() for x in value.split(",") if x.strip())
    return parts


def stable_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze successor distribution morphology proxies.")
    p.add_argument("--conditions", action="extend", nargs="+", default=[], help="Repeatable or comma-separated.")
    p.add_argument("--methods", action="extend", nargs="+", default=[], help="Repeatable or comma-separated.")
    p.add_argument("--preset", default=None)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--dry-run", action="store_true", default=True)
    p.add_argument("--estimate-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--allow-gemma", action="store_true")
    p.add_argument("--confirm-large-run", action="store_true")
    p.add_argument("--allow-top-k-gt-100", action="store_true")
    p.add_argument("--allow-max-prompts-gt-100", action="store_true")
    p.add_argument("--allow-max-steps-gt-64", action="store_true")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--max-prompts", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--candidate-batch-size", type=int, default=1)
    p.add_argument("--flush-every", type=int, default=1)
    p.add_argument("--output-dir", default=str(OUTPUT_DIR_DEFAULT))
    p.add_argument("--model-cache-dir", default=None)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--generation-dir", default=None, help="Override the generation directory for a single-condition run.")
    p.add_argument("--token-log-dir", default=None, help="Override the token-log directory for a single-condition run.")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--repeat-count", type=int, default=None)
    p.add_argument("--seed-start", type=int, default=None)
    p.add_argument("--seed-count", type=int, default=None)
    p.add_argument("--bootstrap-samples", type=int, default=None)
    return p.parse_args()


def resolve_preset(args: argparse.Namespace) -> dict[str, Any]:
    preset = PRESETS.get(args.preset) if args.preset else {}
    if args.preset and preset is None:
        raise SystemExit(f"unknown preset: {args.preset}")
    return dict(preset or {})


def merge_setting(args_value: Any, preset_value: Any, fallback: Any) -> Any:
    if args_value is not None:
        return args_value
    if preset_value is not None:
        return preset_value
    return fallback


def condition_meta(condition: str, generation_dir_override: str | None = None, token_log_dir_override: str | None = None) -> ConditionMeta:
    spec = CONDITION_SPECS.get(condition)
    if spec is None:
        raise SystemExit(f"unknown condition: {condition}")
    gen_dir = spec.get("generation_dir")
    if isinstance(gen_dir, str):
        gen_dir = ROOT / gen_dir
    tok_dir = spec.get("token_log_dir")
    if isinstance(tok_dir, str):
        tok_dir = ROOT / tok_dir
    if generation_dir_override is not None:
        gen_dir = Path(generation_dir_override)
    if token_log_dir_override is not None:
        tok_dir = Path(token_log_dir_override)
    return ConditionMeta(
        condition=condition,
        model_name=spec["model_name"],
        prompt_mode=spec["prompt_mode"],
        prompt_path=Path(spec["prompt_path"]),
        generation_dir=gen_dir,
        token_log_dir=tok_dir,
        notes=spec.get("notes", ""),
        requires_allow_gemma=bool(spec.get("requires_allow_gemma", False)),
    )


def group_by_model(conditions: list[str], generation_dir_override: str | None = None, token_log_dir_override: str | None = None) -> dict[str, list[ConditionMeta]]:
    grouped: dict[str, list[ConditionMeta]] = {}
    for cond in conditions:
        meta = condition_meta(cond, generation_dir_override=generation_dir_override, token_log_dir_override=token_log_dir_override)
        grouped.setdefault(meta.model_name, []).append(meta)
    return grouped


def resolve_device(name: str):
    import torch

    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def dtype_from_name(name: str, device) -> Any:
    import torch

    if name != "auto":
        return getattr(torch, name)
    return torch.float32 if str(device) == "cpu" else torch.float16


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


def safe_decode(tokenizer, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], skip_special_tokens=True).replace("\n", "\\n")


def load_generations(generation_dir: Path, max_prompts: int) -> list[dict[str, Any]]:
    if not generation_dir.exists():
        raise SystemExit(f"generation dir not found: {generation_dir}")
    rows: list[dict[str, Any]] = []
    for path in sorted(generation_dir.glob("*.jsonl")):
        rows.extend(read_jsonl(path))
    rows.sort(key=lambda r: (str(r.get("prompt_id", "")), int(r.get("prompt_index", 0)), str(r.get("method", "")), int(r.get("seed", 0)), int(r.get("repeat_index", 0))))
    prompt_ids = []
    seen = set()
    for row in rows:
        pid = str(row.get("prompt_id", ""))
        if pid not in seen:
            seen.add(pid)
            prompt_ids.append(pid)
        if len(prompt_ids) >= max_prompts:
            break
    return [row for row in rows if str(row.get("prompt_id", "")) in set(prompt_ids)]


def load_completed_keys(path: Path) -> set[tuple[str, str, str, int, int, int]]:
    if not path.exists():
        return set()
    out: set[tuple[str, str, str, int, int, int]] = set()
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                out.add(
                    (
                        row["condition"],
                        row["method"],
                        row["prompt_id"],
                        int(row.get("seed", 0)),
                        int(row.get("repeat_index", 0)),
                        int(row["step"]),
                    )
                )
            except Exception:
                continue
    return out


def write_rows(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a" if not write_header else "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _quantile(values, q: float):
    import torch

    if values.numel() == 0:
        return 0.0
    return float(torch.quantile(values.float(), torch.tensor(q, device=values.device)).item())


def _median_scalar(values) -> float:
    import torch

    if values.numel() == 0:
        return 0.0
    return float(torch.median(values.float()).item())


def morphology_stats(logits):
    import torch

    logits_f = logits.float()
    probs = torch.softmax(logits_f, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, stable=True)
    sorted_logits = logits_f[sorted_idx]
    n = int(sorted_logits.numel())
    logit_std = float(torch.std(logits_f, unbiased=False).item())
    logit_iqr = _quantile(logits_f, 0.75) - _quantile(logits_f, 0.25)
    med = torch.median(logits_f)
    abs_dev = torch.abs(logits_f - med)
    logit_mad = _median_scalar(abs_dev)
    logit_range = float((logits_f.max() - logits_f.min()).item()) if n else 0.0
    top1_prob = float(sorted_probs[0].item()) if n else 0.0
    top5_mass = float(sorted_probs[:5].sum().item()) if n >= 5 else float(sorted_probs.sum().item())
    top10_mass = float(sorted_probs[:10].sum().item()) if n >= 10 else float(sorted_probs.sum().item())
    entropy = float((-(probs * torch.log(probs.clamp_min(1e-12))).sum()).item())
    effective_vocab_size = float(math.exp(entropy))
    csum = torch.cumsum(sorted_probs, dim=0)
    nucleus_size_p90 = int((csum < 0.9).sum().item()) + 1 if n else 0
    nucleus_size_p95 = int((csum < 0.95).sum().item()) + 1 if n else 0
    tail_mass_after_top50 = float(max(0.0, 1.0 - sorted_probs[:50].sum().item())) if n else 0.0
    tail_mass_after_top100 = float(max(0.0, 1.0 - sorted_probs[:100].sum().item())) if n else 0.0
    gap_1_2 = float((sorted_logits[0] - sorted_logits[1]).item()) if n > 1 else 0.0
    gap_1_5 = float((sorted_logits[0] - sorted_logits[4]).item()) if n > 4 else 0.0
    gap_5_20 = float((sorted_logits[4] - sorted_logits[19]).item()) if n > 19 else 0.0
    gaps = (sorted_logits[:-1] - sorted_logits[1:])[:50] if n > 1 else torch.tensor([], device=logits.device)
    max_gap_top50 = float(gaps.max().item()) if gaps.numel() else 0.0
    large_gap_count_top50 = int((gaps > 1.0).sum().item()) if gaps.numel() else 0
    gap_std_top50 = float(torch.std(gaps, unbiased=False).item()) if gaps.numel() > 1 else 0.0
    rank_arr = torch.arange(1, n + 1, device=logits.device, dtype=torch.float32)
    if n > 1:
        slope = torch.cov(torch.stack([rank_arr[: min(n, 50)], torch.log(sorted_probs[: min(n, 50)].clamp_min(1e-12))]))[0, 1] / torch.var(rank_arr[: min(n, 50)], unbiased=False).clamp_min(1e-12)
        rank_prob_slope_top50 = float(slope.item())
    else:
        rank_prob_slope_top50 = 0.0
    return {
        "logit_std": logit_std,
        "logit_iqr": logit_iqr,
        "logit_mad": logit_mad,
        "logit_range": logit_range,
        "top1_prob": top1_prob,
        "top5_mass": top5_mass,
        "top10_mass": top10_mass,
        "softmax_entropy": entropy,
        "effective_vocab_size": effective_vocab_size,
        "tail_mass_after_top50": tail_mass_after_top50,
        "tail_mass_after_top100": tail_mass_after_top100,
        "nucleus_size_p90": nucleus_size_p90,
        "nucleus_size_p95": nucleus_size_p95,
        "rank_prob_slope_top50": rank_prob_slope_top50,
        "gap_1_2": gap_1_2,
        "gap_1_5": gap_1_5,
        "gap_5_20": gap_5_20,
        "max_gap_top50": max_gap_top50,
        "large_gap_count_top50": large_gap_count_top50,
        "gap_std_top50": gap_std_top50,
        "sorted_idx": sorted_idx,
        "sorted_probs": sorted_probs,
    }


def token_type_masses(tokenizer, top_ids, top_probs, prompt_text: str, recent_text: str) -> dict[str, float]:
    import re

    total = float(top_probs.sum().item()) if top_probs.numel() else 0.0
    prompt_words = set(re.findall(r"[A-Za-z]{4,}", prompt_text.lower()))
    prompt_markers = {"[ wp ]", "[wp]", "wp", "writingprompts", "prompt", "story", "continue", "narrative", "prose"}
    recent_words = set(re.findall(r"[A-Za-z]{4,}", recent_text.lower()))
    vals = {k: 0.0 for k in [
        "markup_mass_top20",
        "newline_or_bullet_mass_top20",
        "quote_mass_top20",
        "digit_mass_top20",
        "subword_fragment_mass_top20",
        "html_marker_mass_top20",
        "title_marker_mass_top20",
        "choice_marker_mass_top20",
        "prompt_overlap_mass_top20",
        "prompt_marker_mass_top20",
        "prompt_entity_overlap_mass_top20",
        "recent_overlap_mass_top20",
    ]}
    for token_id, prob in zip(top_ids.tolist(), top_probs.tolist()):
        tok = tokenizer.decode([int(token_id)], skip_special_tokens=True)
        low = tok.lower()
        if any(x in low for x in ["<", ">", "</", "<br", "<p", "<div", "```", "###"]):
            vals["markup_mass_top20"] += prob
        if low.startswith(("\n", "-", "*", "•", "1.", "2.", "3.", "[", "(", "a.", "b.", "c.")) or "\n" in tok:
            vals["newline_or_bullet_mass_top20"] += prob
        if '"' in tok or "'" in tok or "“" in tok or "”" in tok:
            vals["quote_mass_top20"] += prob
        if any(ch.isdigit() for ch in tok):
            vals["digit_mass_top20"] += prob
        if any(sym in tok for sym in ["##", "▁", "@@", "Ġ", "Ċ"]) or len(tok.strip()) <= 2:
            vals["subword_fragment_mass_top20"] += prob
        if any(x in low for x in ["html", "<p", "<div", "<br", "</p>", "</div>"]):
            vals["html_marker_mass_top20"] += prob
        if any(x in low for x in ["title", "chapter", "part", "section"]):
            vals["title_marker_mass_top20"] += prob
        if any(x in low for x in ["option", "choice", "a)", "b)", "c)", "multiple choice", "quiz"]):
            vals["choice_marker_mass_top20"] += prob
        words = set(re.findall(r"[A-Za-z]{4,}", low))
        if words and prompt_words.intersection(words):
            vals["prompt_overlap_mass_top20"] += prob
        if any(marker in low for marker in prompt_markers):
            vals["prompt_marker_mass_top20"] += prob
        if words and words.intersection(recent_words):
            vals["recent_overlap_mass_top20"] += prob
        if words and ("wp" in words or "writingprompts" in words):
            vals["prompt_entity_overlap_mass_top20"] += prob
    if total > 0:
        for key in vals:
            vals[key] /= total
    return vals


def reconstruct_prefix(gen: dict[str, Any], step: int) -> str:
    prefix = gen.get("generated_prefix")
    if isinstance(prefix, str) and prefix:
        return prefix
    cont = str(gen.get("continuation_text", ""))
    return cont[:0] if step <= 0 else cont


def output_mode_label(tokens: list[str], prompt_text: str) -> str:
    text = " ".join(tokens).lower()
    if any(x in text for x in ["<html", "<div", "<p", "```"]):
        return "markup_or_html"
    if any(x in text for x in ["chapter", "title", "character sheet", "cast"]):
        return "title_or_character_sheet"
    if any(x in text for x in ["quiz", "options", "multiple choice", "a)", "b)"]):
        return "quiz_or_options"
    if any(x in text for x in ["analysis", "commentary", "note"]):
        return "commentary_or_analysis"
    if any(x in text for x in ["wp", "writingprompts", "continue the following"]):
        return "prompt_marker_recycling"
    if len(text.strip()) <= 2:
        return "token_fragment_collapse"
    if prompt_text and any(w in text for w in prompt_text.lower().split()[:5]):
        return "direct_story_prose"
    return "semantic_drift_possible"


def spearman_rank_corr(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return 0.0
    xs = {v: i for i, v in enumerate(sorted(set(x)))}
    ys = {v: i for i, v in enumerate(sorted(set(y)))}
    rx = [xs[v] for v in x]
    ry = [ys[v] for v in y]
    import statistics

    mx = statistics.mean(rx)
    my = statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    deny = math.sqrt(sum((b - my) ** 2 for b in ry))
    return float(num / (denx * deny)) if denx and deny else 0.0


def least_squares_slope(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(ys) < 2 or len(xs) != len(ys):
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    varx = sum((x - mx) ** 2 for x in xs)
    if varx == 0:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return float(cov / varx)


def summarize_rows(rows: list[dict[str, Any]], bootstrap_samples: int) -> dict[str, list[dict[str, Any]]]:
    if not rows:
        return {k: [] for k in SUMMARY_GROUPS}
    import pandas as pd
    import numpy as np

    df = pd.DataFrame(rows)
    numeric_cols = [c for c in FEATURE_COLUMNS if c not in {"condition", "model_name", "prompt_mode", "method", "prompt_id", "prompt_index", "seed", "repeat_index", "step", "chosen_token_id", "chosen_token", "output_mode_label"}]

    def prompt_level_agg(g: pd.DataFrame) -> pd.DataFrame:
        return g.groupby("prompt_id", dropna=False)[numeric_cols].mean(numeric_only=True).reset_index()

    outputs: dict[str, list[dict[str, Any]]] = {}
    for name, group_cols in SUMMARY_GROUPS.items():
        group_rows: list[dict[str, Any]] = []
        for keys, g in df.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            item = dict(zip(group_cols, keys))
            item["n_rows"] = int(len(g))
            item["n_prompts"] = int(g["prompt_id"].nunique())
            item["n_repeats"] = int(g.get("repeat_index", pd.Series([0])).nunique())
            item["n_seeds"] = int(g.get("seed", pd.Series([0])).nunique())
            for col in numeric_cols:
                item[f"mean_{col}"] = float(g[col].mean())
                item[f"std_{col}"] = float(g[col].std(ddof=0)) if len(g) > 1 else 0.0
                item[f"standard_error_{col}"] = float(g[col].std(ddof=1) / math.sqrt(len(g))) if len(g) > 1 else 0.0
            if len(g) > 0:
                prompt_avg = prompt_level_agg(g)
                rng = np.random.default_rng(0)
                bs = []
                for _ in range(max(1, bootstrap_samples)):
                    sample_ids = rng.choice(prompt_avg["prompt_id"].tolist(), size=len(prompt_avg), replace=True)
                    sample = prompt_avg.set_index("prompt_id").loc[sample_ids]
                    bs.append(sample[numeric_cols].mean(numeric_only=True))
                bsdf = pd.DataFrame(bs)
                for col in numeric_cols:
                    item[f"bootstrap_95ci_low_{col}"] = float(bsdf[col].quantile(0.025))
                    item[f"bootstrap_95ci_high_{col}"] = float(bsdf[col].quantile(0.975))
            group_rows.append(item)
        outputs[name] = group_rows
    return outputs


def write_report(
    out_dir: Path,
    feature_rows: list[dict[str, Any]],
    summaries: dict[str, list[dict[str, Any]]],
    requested_repeat_count: int,
) -> None:
    report = out_dir / "successor_distribution_morphology_report.md"
    lines = [
        "# Successor Distribution Morphology",
        "",
        "These are diagnostic morphology proxies, not quality metrics.",
        "",
        "Main morphology columns are computed from the successor distribution after the actual generated token.",
        "Current-logit candidate alignment diagnostics are kept separate in the `candidate_*` fields.",
        "",
        "## What the proxies cover",
        "",
        "- spread: `logit_std`, `logit_iqr`, `logit_mad`, `logit_range`",
        "- concentration: `top1_prob`, `top5_mass`, `top10_mass`, `softmax_entropy`, `effective_vocab_size`",
        "- tail: `tail_mass_after_top50`, `tail_mass_after_top100`, `nucleus_size_p90`, `nucleus_size_p95`, `rank_prob_slope_top50`",
        "- gap / cluster: `gap_1_2`, `gap_1_5`, `gap_5_20`, `max_gap_top50`, `large_gap_count_top50`, `gap_std_top50`",
        "- token type and recycling: markup, bullet, quote, digit, HTML, title, choice, prompt/context overlap, recent overlap",
        "- flatness alignment: `rank_at_flat_p50`, `rank_at_flat_p75`, `prob_mass_p25_p75`, `spearman_flatness_probability_rank`",
        "",
        "## Mode mapping",
        "",
        "- direct narrative prose",
        "- generic atmospheric prose",
        "- title or character sheet",
        "- commentary or analysis",
        "- quiz or options",
        "- markup or HTML",
        "- prompt marker recycling",
        "- token fragment collapse",
        "- repetition or loop",
        "- semantic drift possible",
        "",
        "## Execution model",
        "",
        "This script loads one model at a time, processes the requested conditions for that model, streams rows to CSV, and clears memory before moving on.",
        "",
        f"- feature rows collected: {len(feature_rows)}",
        f"- requested_repeat_count: {requested_repeat_count}",
        f"- observed_seed_count: {len({int(row.get('seed', 0)) for row in feature_rows}) if feature_rows else 0}",
        f"- observed_repeat_count: {len({int(row.get('repeat_index', 0)) for row in feature_rows}) if feature_rows else 0}",
        f"- by-condition summary rows: {len(summaries.get('by_condition', []))}",
        f"- by-method summary rows: {len(summaries.get('by_method', []))}",
        f"- by-prompt summary rows: {len(summaries.get('by_prompt', []))}",
        f"- by-mode summary rows: {len(summaries.get('by_mode', []))}",
        "",
        "## Presets",
        "",
        "- `smoke`: qwen_raw, top_p, 20 top-K, 2 prompts, 4 steps",
        "- `qwen_plain`: qwen_raw and qwen_wrapped, 10 prompts, 16 steps",
        "- `gpt2_main`: GPT-2 main probe; GPT-2 path must be resolved explicitly",
        "- `gemma_wrapped10`: Gemma wrapped10; requires `--allow-gemma`",
        "- `modern_compare`: Gemma wrapped plus Qwen wrapped/chat when available",
        "- `morphology_20x20_top20`: 20 prompts × 20 repeats × top20 morphology slice",
        "",
        "## Notes",
        "",
        "Prompt_count, repeat_count, and top_k are distinct. `prompt_count` is the number of prompts; `repeat_count`/`seed_count` are stochastic replicates; `top_k` is the candidate set used for morphology.",
        "For the 20x20 preset, actual repeated generation outputs are required to observe repeat_count > 1; otherwise the report will show the available observed repeat count only.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_generation_dir(meta: ConditionMeta) -> Path:
    if meta.generation_dir is not None:
        if meta.generation_dir.exists():
            return meta.generation_dir
        raise SystemExit(f"generation dir not found: {meta.generation_dir}")
    candidates = list(CONDITION_SPECS.get(meta.condition, {}).get("generation_candidates", []))
    existing = [Path(path) for path in candidates if Path(path).exists()]
    if len(existing) == 1:
        return existing[0]
    candidate_lines = "\n".join(f"  - {Path(path)}" for path in candidates) or "  - <none>"
    if existing:
        existing_lines = "\n".join(f"  - {path}" for path in existing)
        raise SystemExit(
            f"ambiguous generation dir for {meta.condition}; multiple candidates exist:\n{existing_lines}\n"
            f"candidate list considered:\n{candidate_lines}"
        )
    raise SystemExit(
        f"generation dir for {meta.condition} is ambiguous or missing.\n"
        f"candidate paths considered:\n{candidate_lines}"
    )


def estimate_totals(targets: list[ConditionMeta], methods: list[str], max_prompts: int, max_steps: int, repeat_count: int, top_k: int) -> tuple[int, int, int, int]:
    context_passes = len(targets) * len(methods) * max_prompts * max_steps * repeat_count
    successor_passes = context_passes
    candidate_alignment_passes = context_passes * top_k
    total_passes = context_passes + successor_passes + candidate_alignment_passes
    return context_passes, successor_passes, candidate_alignment_passes, total_passes


def maybe_gate(args: argparse.Namespace, targets: list[ConditionMeta]) -> None:
    if any(t.requires_allow_gemma for t in targets) and not args.allow_gemma:
        raise SystemExit("refusing Gemma conditions without --allow-gemma")
    if args.top_k > 100 and not args.allow_top_k_gt_100:
        raise SystemExit("refusing --top-k > 100 without --allow-top-k-gt-100")
    if args.max_prompts > 100 and not args.allow_max_prompts_gt_100:
        raise SystemExit("refusing --max-prompts > 100 without --allow-max-prompts-gt-100")
    if args.max_steps > 64 and not args.allow_max_steps_gt_64:
        raise SystemExit("refusing --max-steps > 64 without --allow-max-steps-gt-64")
    if args.confirm_large_run is False and (args.max_prompts * args.max_steps) >= 200:
        print("warning: large run requested without --confirm-large-run")


def preset_overrides(args: argparse.Namespace) -> dict[str, Any]:
    preset = PRESETS.get(args.preset or "", {})
    if args.preset and not preset:
        raise SystemExit(f"unknown preset: {args.preset}")
    merged = dict(preset)
    merged["conditions"] = stable_unique(parse_csv_like(args.conditions) or merged.get("conditions", []))
    merged["methods"] = stable_unique(parse_csv_like(args.methods) or merged.get("methods", []))
    merged["top_k"] = merge_setting(args.top_k, merged.get("top_k"), 20)
    merged["max_prompts"] = merge_setting(args.max_prompts, merged.get("max_prompts"), 2)
    merged["max_steps"] = merge_setting(args.max_steps, merged.get("max_steps"), 4)
    merged["repeat_count"] = merge_setting(args.repeat_count, merged.get("repeat_count"), 1)
    merged["seed_start"] = merge_setting(args.seed_start, merged.get("seed_start"), 0)
    merged["seed_count"] = merge_setting(args.seed_count, merged.get("seed_count"), merged["repeat_count"])
    merged["bootstrap_samples"] = merge_setting(args.bootstrap_samples, merged.get("bootstrap_samples"), 1000)
    return merged


def print_plan(targets: list[ConditionMeta], methods: list[str], args: argparse.Namespace) -> None:
    context_passes, successor_passes, candidate_alignment_passes, total_passes = estimate_totals(
        targets, methods, args.max_prompts, args.max_steps, args.repeat_count, args.top_k
    )
    print("conditions:", ", ".join(t.condition for t in targets))
    print("models:", ", ".join(sorted({t.model_name for t in targets})))
    print("methods:", ", ".join(methods))
    print("top_k:", args.top_k)
    print("max_prompts:", args.max_prompts)
    print("max_steps:", args.max_steps)
    print("repeat_count:", args.repeat_count)
    print("seed_start:", args.seed_start)
    print("seed_count:", args.seed_count)
    print("candidate_batch_size:", args.candidate_batch_size)
    print("estimated_context_forward_passes:", context_passes)
    print("estimated_successor_forward_passes:", successor_passes)
    print("estimated_candidate_alignment_forward_passes:", candidate_alignment_passes)
    print("estimated_total_forward_passes:", total_passes)
    for t in targets:
        print(f"target[{t.condition}]: model={t.model_name}")
        print(f"  prompt_mode={t.prompt_mode}")
        print(f"  prompt_path={t.prompt_path}")
        print(f"  generation_dir={t.generation_dir}")
        if t.token_log_dir is not None:
            print(f"  token_log_dir={t.token_log_dir}")
        print(f"  notes={t.notes}")


def build_prompt_text(gen: dict[str, Any], condition: ConditionMeta) -> str:
    prompt = str(gen.get("prompt", ""))
    if prompt:
        return prompt
    if condition.prompt_path.exists():
        rows = read_jsonl(condition.prompt_path)
        pid = str(gen.get("prompt_id", ""))
        for row in rows:
            if str(row.get("prompt_id", "")) == pid:
                return str(row.get("prompt", ""))
    return ""


def summarize_candidate_metrics(current_logits, candidate_ids, candidate_flatness, tokenizer, prompt_text: str, recent_text: str):
    import torch

    flatness = torch.tensor(candidate_flatness, dtype=torch.float32)
    probs = torch.softmax(current_logits.float(), dim=-1)
    cand_probs = probs[candidate_ids]
    cand_logprobs = torch.log_softmax(current_logits.float(), dim=-1)[candidate_ids]
    order = torch.argsort(flatness, stable=True)
    flat_sorted = flatness[order]
    prob_sorted = cand_probs[order]
    cum_prob = torch.cumsum(prob_sorted, dim=0)
    total_prob = float(prob_sorted.sum().item()) if prob_sorted.numel() else 1.0
    if total_prob <= 0:
        total_prob = 1.0
    p25_val = float(torch.quantile(flat_sorted, torch.tensor(0.25)).item()) if flat_sorted.numel() else 0.0
    p50_val = float(torch.quantile(flat_sorted, torch.tensor(0.5)).item()) if flat_sorted.numel() else 0.0
    p75_val = float(torch.quantile(flat_sorted, torch.tensor(0.75)).item()) if flat_sorted.numel() else 0.0
    p50_idx = int(torch.argmin(torch.abs(flat_sorted - p50_val)).item()) if flat_sorted.numel() else 0
    p75_idx = int(torch.argmin(torch.abs(flat_sorted - p75_val)).item()) if flat_sorted.numel() else 0
    current_rank_order = torch.argsort(cand_probs, descending=True, stable=True)
    current_rank = torch.empty_like(current_rank_order)
    current_rank[current_rank_order] = torch.arange(1, len(current_rank_order) + 1, device=current_rank_order.device)
    rank_at_p50 = float(current_rank[order[p50_idx]].item()) if current_rank.numel() else 0.0
    rank_at_p75 = float(current_rank[order[p75_idx]].item()) if current_rank.numel() else 0.0
    p25_mass = float(prob_sorted[flat_sorted <= p25_val].sum().item() / total_prob)
    mid_mass = float(prob_sorted[(flat_sorted > p25_val) & (flat_sorted <= p75_val)].sum().item() / total_prob)
    high_mass = float(prob_sorted[flat_sorted > p75_val].sum().item() / total_prob)
    if flat_sorted.numel() > 1:
        x = flat_sorted.tolist()
        y = prob_sorted.tolist()
        spearman = spearman_rank_corr(x, y)
    else:
        spearman = 0.0
    top_ids = candidate_ids[:20] if candidate_ids.numel() else candidate_ids[:0]
    top_probs = cand_probs[:20] if candidate_ids.numel() else cand_probs[:0]
    token_masses = token_type_masses(tokenizer, top_ids.cpu(), top_probs.cpu(), prompt_text, recent_text)
    return {
        "flat_min": float(flat_sorted.min().item()) if flat_sorted.numel() else 0.0,
        "flat_p10": float(torch.quantile(flat_sorted, torch.tensor(0.10)).item()) if flat_sorted.numel() else 0.0,
        "flat_p25": p25_val,
        "flat_p50": p50_val,
        "flat_p75": p75_val,
        "flat_p90": float(torch.quantile(flat_sorted, torch.tensor(0.90)).item()) if flat_sorted.numel() else 0.0,
        "flat_max": float(flat_sorted.max().item()) if flat_sorted.numel() else 0.0,
        "rank_at_flat_p50": rank_at_p50,
        "rank_at_flat_p75": rank_at_p75,
        "prob_mass_p25_p75": mid_mass,
        "spearman_flatness_probability_rank": spearman,
        "token_masses": token_masses,
        "candidate_current_prob": cand_probs.tolist(),
        "candidate_current_logprob": cand_logprobs.tolist(),
        "candidate_ranks": list(range(1, len(candidate_ids) + 1)),
        "flatness_sorted_idx": order.tolist(),
        "candidate_token_ids_sorted": candidate_ids[order].tolist(),
        "candidate_probs_sorted": prob_sorted.tolist(),
        "candidate_flat_sorted": flat_sorted.tolist(),
        "candidate_ranks_sorted": [int(current_rank[i].item()) for i in range(len(current_rank))],
        "candidate_logprobs_sorted": cand_logprobs[order].tolist(),
        "candidate_total_prob": total_prob,
        "prob_mass_below_p25": p25_mass,
        "prob_mass_above_p75": high_mass,
    }


def actual_successor_features(successor_logits, tokenizer, prompt_text: str, recent_text: str) -> dict[str, Any]:
    stats = morphology_stats(successor_logits)
    token_masses = token_type_masses(
        tokenizer,
        stats["sorted_idx"][:20].detach().cpu(),
        stats["sorted_probs"][:20].detach().cpu(),
        prompt_text,
        recent_text,
    )
    return {**stats, **token_masses}


def run_model_group(model_name: str, metas: list[ConditionMeta], args: argparse.Namespace, out_dir: Path) -> list[dict[str, Any]]:
    from flatness_decoding.model_io import load_causal_lm
    import torch

    model, tokenizer, device = load_causal_lm(
        model_name,
        device_name=args.device,
        dtype_name=args.dtype,
        cache_dir=args.model_cache_dir,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )
    all_rows: list[dict[str, Any]] = []
    try:
        for meta in metas:
            generation_dir = resolve_generation_dir(meta)
            generations = load_generations(generation_dir, args.max_prompts)
            completed = load_completed_keys(out_dir / "successor_distribution_morphology_features.csv") if args.resume else set()
            pending_rows: list[dict[str, Any]] = []
            for gen in generations:
                method = str(gen.get("method", ""))
                if method not in args.methods:
                    continue
                prompt_id = str(gen.get("prompt_id", ""))
                prompt_index = int(gen.get("prompt_index", 0))
                seed = int(gen.get("seed", 0))
                repeat_index = int(gen.get("repeat_index", 0))
                prompt_text = build_prompt_text(gen, meta)
                continuation = str(gen.get("continuation_text", ""))
                cont_ids = tokenizer(continuation, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
                prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
                max_steps = min(args.max_steps, len(cont_ids))
                for step in range(max_steps):
                    key = (meta.condition, method, prompt_id, seed, repeat_index, step)
                    if key in completed:
                        continue
                    prefix_ids = torch.tensor([cont_ids[:step]], dtype=torch.long, device=device)
                    input_ids = prompt_ids if prefix_ids.numel() == 0 else torch.cat([prompt_ids, prefix_ids], dim=1)
                    chosen_token_id = int(cont_ids[step])
                    chosen_token = safe_decode(tokenizer, chosen_token_id) if chosen_token_id >= 0 else ""
                    started = time.perf_counter()
                    with torch.inference_mode():
                        current_logits = model(input_ids=input_ids).logits[0, -1, :]
                        current_probs = torch.softmax(current_logits.float(), dim=-1)
                        current_logprobs = torch.log_softmax(current_logits.float(), dim=-1)
                        current_current_prob = float(current_probs[chosen_token_id].item()) if chosen_token_id >= 0 else 0.0
                        current_current_logprob = float(current_logprobs[chosen_token_id].item()) if chosen_token_id >= 0 else 0.0
                        current_current_rank = int((current_logits > current_logits[chosen_token_id]).sum().item()) + 1 if chosen_token_id >= 0 else 0
                        top_k = min(args.top_k, int(current_logits.shape[-1]))
                        top = torch.topk(current_logits, k=top_k)
                        candidate_ids = top.indices.detach().cpu()
                        candidate_flatness = []
                        for cid in candidate_ids.tolist():
                            successor_ids = torch.cat([input_ids, torch.tensor([[cid]], device=device)], dim=1)
                            succ_logits = model(input_ids=successor_ids).logits[0, -1, :]
                            candidate_flatness.append(float(torch.std(succ_logits.float(), unbiased=False).item()))
                        candidate_flatness_t = torch.tensor(candidate_flatness)
                        summary = summarize_candidate_metrics(
                            current_logits.detach().cpu(),
                            candidate_ids,
                            candidate_flatness_t,
                            tokenizer,
                            prompt_text,
                            continuation[: max(0, step * 4)],
                        )
                        successor_ids = torch.cat([input_ids, torch.tensor([[chosen_token_id]], device=device)], dim=1)
                        successor_logits = model(input_ids=successor_ids).logits[0, -1, :]
                        successor_features = actual_successor_features(
                            successor_logits,
                            tokenizer,
                            prompt_text,
                            continuation[: max(0, step * 4)],
                        )
                        chosen_token = safe_decode(tokenizer, chosen_token_id) if chosen_token_id >= 0 else ""
                        mode_label = output_mode_label([chosen_token], prompt_text)
                        row = {
                            "condition": meta.condition,
                            "model_name": meta.model_name,
                            "prompt_mode": meta.prompt_mode,
                            "method": method,
                            "prompt_id": prompt_id,
                            "prompt_index": prompt_index,
                            "seed": seed,
                            "repeat_index": repeat_index,
                            "requested_repeat_count": args.repeat_count,
                            "step": step,
                            "top_k": top_k,
                            "candidate_batch_size": args.candidate_batch_size,
                            "chosen_token_id": chosen_token_id,
                            "chosen_token": chosen_token,
                            "context_input_tokens": int(input_ids.shape[-1]),
                            "output_mode_label": mode_label,
                            "candidate_rank": current_current_rank,
                            "candidate_token_id": chosen_token_id,
                            "candidate_token": chosen_token,
                            "candidate_current_logprob": current_current_logprob,
                            "candidate_current_rank": current_current_rank,
                            "candidate_current_prob": current_current_prob,
                            "logit_std": float(successor_features["logit_std"]),
                            "logit_iqr": float(successor_features["logit_iqr"]),
                            "logit_mad": float(successor_features["logit_mad"]),
                            "logit_range": float(successor_features["logit_range"]),
                            "top1_prob": float(successor_features["top1_prob"]),
                            "top5_mass": float(successor_features["top5_mass"]),
                            "top10_mass": float(successor_features["top10_mass"]),
                            "softmax_entropy": float(successor_features["softmax_entropy"]),
                            "effective_vocab_size": float(successor_features["effective_vocab_size"]),
                            "tail_mass_after_top50": float(successor_features["tail_mass_after_top50"]),
                            "tail_mass_after_top100": float(successor_features["tail_mass_after_top100"]),
                            "nucleus_size_p90": int(successor_features["nucleus_size_p90"]),
                            "nucleus_size_p95": int(successor_features["nucleus_size_p95"]),
                            "rank_prob_slope_top50": float(successor_features["rank_prob_slope_top50"]),
                            "gap_1_2": float(successor_features["gap_1_2"]),
                            "gap_1_5": float(successor_features["gap_1_5"]),
                            "gap_5_20": float(successor_features["gap_5_20"]),
                            "max_gap_top50": float(successor_features["max_gap_top50"]),
                            "large_gap_count_top50": int(successor_features["large_gap_count_top50"]),
                            "gap_std_top50": float(successor_features["gap_std_top50"]),
                            "markup_mass_top20": float(successor_features["markup_mass_top20"]),
                            "newline_or_bullet_mass_top20": float(successor_features["newline_or_bullet_mass_top20"]),
                            "quote_mass_top20": float(successor_features["quote_mass_top20"]),
                            "digit_mass_top20": float(successor_features["digit_mass_top20"]),
                            "subword_fragment_mass_top20": float(successor_features["subword_fragment_mass_top20"]),
                            "html_marker_mass_top20": float(successor_features["html_marker_mass_top20"]),
                            "title_marker_mass_top20": float(successor_features["title_marker_mass_top20"]),
                            "choice_marker_mass_top20": float(successor_features["choice_marker_mass_top20"]),
                            "prompt_overlap_mass_top20": float(successor_features["prompt_overlap_mass_top20"]),
                            "prompt_marker_mass_top20": float(successor_features["prompt_marker_mass_top20"]),
                            "prompt_entity_overlap_mass_top20": float(successor_features["prompt_entity_overlap_mass_top20"]),
                            "recent_overlap_mass_top20": float(successor_features["recent_overlap_mass_top20"]),
                            "rank_at_flat_p50": summary["rank_at_flat_p50"],
                            "rank_at_flat_p75": summary["rank_at_flat_p75"],
                            "prob_mass_p25_p75": summary["prob_mass_p25_p75"],
                            "spearman_flatness_probability_rank": summary["spearman_flatness_probability_rank"],
                            "runtime_seconds": time.perf_counter() - started,
                        }
                        pending_rows.append(row)
                        all_rows.append(row)
                        if len(pending_rows) >= args.flush_every:
                            write_rows(out_dir / "successor_distribution_morphology_features.csv", pending_rows, FEATURE_COLUMNS)
                            pending_rows.clear()
                            clear_memory()
                if pending_rows:
                    write_rows(out_dir / "successor_distribution_morphology_features.csv", pending_rows, FEATURE_COLUMNS)
                    pending_rows.clear()
                    clear_memory()
    finally:
        del model, tokenizer
        clear_memory()
    return all_rows


def main() -> None:
    args = parse_args()
    preset = preset_overrides(args)
    conditions = stable_unique(preset.get("conditions", []))
    methods = stable_unique(preset.get("methods", [])) or ["top_p"]
    if not conditions:
        conditions = ["qwen_raw"]
    if not methods:
        methods = ["top_p"]
    args.top_k = int(preset["top_k"])
    args.max_prompts = int(preset["max_prompts"])
    args.max_steps = int(preset["max_steps"])
    args.repeat_count = int(preset["repeat_count"])
    args.seed_start = int(preset["seed_start"])
    args.seed_count = int(preset["seed_count"])
    args.bootstrap_samples = int(preset["bootstrap_samples"])
    args.methods = methods
    targets = [condition_meta(c, generation_dir_override=args.generation_dir, token_log_dir_override=args.token_log_dir) for c in conditions]
    maybe_gate(args, targets)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print_plan(targets, methods, args)
    if args.estimate_only or not args.execute:
        return
    grouped = group_by_model(conditions, generation_dir_override=args.generation_dir, token_log_dir_override=args.token_log_dir)
    all_rows: list[dict[str, Any]] = []
    for model_name, metas in grouped.items():
        all_rows.extend(run_model_group(model_name, metas, args, out_dir))
    summaries = summarize_rows(all_rows, args.bootstrap_samples)
    write_rows(out_dir / "successor_distribution_morphology_by_condition.csv", summaries["by_condition"], list(summaries["by_condition"][0].keys()) if summaries["by_condition"] else [])
    write_rows(out_dir / "successor_distribution_morphology_by_method.csv", summaries["by_method"], list(summaries["by_method"][0].keys()) if summaries["by_method"] else [])
    write_rows(out_dir / "successor_distribution_morphology_by_prompt.csv", summaries["by_prompt"], list(summaries["by_prompt"][0].keys()) if summaries["by_prompt"] else [])
    write_rows(out_dir / "successor_distribution_morphology_by_mode.csv", summaries["by_mode"], list(summaries["by_mode"][0].keys()) if summaries["by_mode"] else [])
    confidence_rows = []
    for name, rows in summaries.items():
        for row in rows:
            confidence_rows.append({"summary_type": name, **row})
    if confidence_rows:
        write_rows(out_dir / "successor_distribution_morphology_confidence_intervals.csv", confidence_rows, list(confidence_rows[0].keys()))
    write_report(out_dir, all_rows, summaries, args.repeat_count)


if __name__ == "__main__":
    main()
