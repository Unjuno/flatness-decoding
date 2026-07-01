from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from flatness_decoding.io_utils import read_jsonl

from analyze_successor_distribution_morphology import (
    CONDITION_SPECS,
    clear_memory,
    morphology_stats,
    output_mode_label,
    safe_decode,
    token_type_masses,
)


OUTPUT_DIR_DEFAULT = ROOT / "results" / "analysis"

REGIME_FAMILIES = {
    "spread": {
        "feature": "logit_std",
        "regimes": ("low_spread", "mid_spread", "high_spread"),
    },
    "concentration": {
        "feature": "top1_prob",
        "regimes": ("low_concentration", "mid_concentration", "high_concentration"),
    },
    "tail": {
        "feature": "tail_mass_after_top50",
        "regimes": ("thin_tail", "mid_tail", "heavy_tail"),
    },
    "gap": {
        "feature": "gap_1_2",
        "regimes": ("small_gap", "mid_gap", "large_gap"),
    },
    "token_type": {
        "feature": "markup_mass_top20",
        "regimes": (
            "markup_heavy",
            "list_heavy",
            "fragment_heavy",
            "prompt_overlap_heavy",
            "recent_overlap_heavy",
        ),
    },
}

TOKEN_TYPE_REGIMES = {
    "markup_heavy": "successor_markup_mass_top20",
    "list_heavy": "successor_newline_or_bullet_mass_top20",
    "fragment_heavy": "successor_subword_fragment_mass_top20",
    "prompt_overlap_heavy": "successor_prompt_overlap_mass_top20",
    "recent_overlap_heavy": "successor_recent_overlap_mass_top20",
}

SCALAR_FEATURES = [
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
]

MORPHOLOGY_COLUMNS = [
    "condition",
    "model_name",
    "prompt_mode",
    "prompt_id",
    "prompt_index",
    "step",
    "prompt",
    "candidate_pool_mode",
    "candidate_pool_size",
    "candidate_rank",
    "candidate_token_id",
    "candidate_token",
    "candidate_current_logprob",
    "candidate_current_prob",
    "candidate_current_rank",
    "assigned_regimes",
    "current_logit_std",
    "current_logit_iqr",
    "current_logit_mad",
    "current_logit_range",
    "current_top1_prob",
    "current_top5_mass",
    "current_top10_mass",
    "current_softmax_entropy",
    "current_effective_vocab_size",
    "current_tail_mass_after_top50",
    "current_tail_mass_after_top100",
    "current_nucleus_size_p90",
    "current_nucleus_size_p95",
    "current_rank_prob_slope_top50",
    "successor_logit_std",
    "successor_logit_iqr",
    "successor_logit_mad",
    "successor_logit_range",
    "successor_top1_prob",
    "successor_top5_mass",
    "successor_top10_mass",
    "successor_softmax_entropy",
    "successor_effective_vocab_size",
    "successor_tail_mass_after_top50",
    "successor_tail_mass_after_top100",
    "successor_nucleus_size_p90",
    "successor_nucleus_size_p95",
    "successor_rank_prob_slope_top50",
    "successor_gap_1_2",
    "successor_gap_1_5",
    "successor_gap_5_20",
    "successor_max_gap_top50",
    "successor_large_gap_count_top50",
    "successor_gap_std_top50",
    "successor_markup_mass_top20",
    "successor_newline_or_bullet_mass_top20",
    "successor_quote_mass_top20",
    "successor_digit_mass_top20",
    "successor_subword_fragment_mass_top20",
    "successor_html_marker_mass_top20",
    "successor_title_marker_mass_top20",
    "successor_choice_marker_mass_top20",
    "successor_prompt_overlap_mass_top20",
    "successor_prompt_marker_mass_top20",
    "successor_prompt_entity_overlap_mass_top20",
    "successor_recent_overlap_mass_top20",
    "runtime_seconds",
]

SELECTED_COLUMNS = [
    "condition",
    "model_name",
    "prompt_id",
    "prompt_index",
    "step",
    "prompt",
    "regime_family",
    "regime",
    "selected_rank_within_regime",
    "candidate_rank",
    "candidate_token_id",
    "candidate_token",
    "candidate_current_rank",
    "candidate_current_prob",
    "assigned_regimes",
    "successor_logit_std",
    "successor_logit_iqr",
    "successor_logit_mad",
    "successor_logit_range",
    "successor_top1_prob",
    "successor_top5_mass",
    "successor_top10_mass",
    "successor_softmax_entropy",
    "successor_effective_vocab_size",
    "successor_tail_mass_after_top50",
    "successor_tail_mass_after_top100",
    "successor_nucleus_size_p90",
    "successor_nucleus_size_p95",
    "successor_rank_prob_slope_top50",
    "successor_gap_1_2",
    "successor_gap_1_5",
    "successor_gap_5_20",
    "successor_max_gap_top50",
    "successor_large_gap_count_top50",
    "successor_gap_std_top50",
    "successor_markup_mass_top20",
    "successor_newline_or_bullet_mass_top20",
    "successor_quote_mass_top20",
    "successor_digit_mass_top20",
    "successor_subword_fragment_mass_top20",
    "successor_html_marker_mass_top20",
    "successor_title_marker_mass_top20",
    "successor_choice_marker_mass_top20",
    "successor_prompt_overlap_mass_top20",
    "successor_prompt_marker_mass_top20",
    "successor_prompt_entity_overlap_mass_top20",
    "successor_recent_overlap_mass_top20",
]


@dataclass(frozen=True)
class ProbeConditionMeta:
    condition: str
    model_name: str
    prompt_mode: str
    prompt_path: Path
    notes: str
    requires_allow_gemma: bool = False


def parse_csv_like(values: list[str]) -> list[str]:
    parts: list[str] = []
    for value in values:
        parts.extend(item.strip() for item in value.split(",") if item.strip())
    return parts


def stable_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Morphology-stratified successor probing.")
    p.add_argument("--conditions", action="append", default=[], help="Repeatable or comma-separated.")
    p.add_argument("--preset", default=None)
    p.add_argument("--candidate-pool-mode", choices=["raw_topk", "sampled"], default=None)
    p.add_argument("--candidate-pool-size", type=int, default=None)
    p.add_argument("--samples-per-regime", type=int, default=None)
    p.add_argument("--max-prompts", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--generate-continuations", action="store_true")
    p.add_argument("--continuation-max-new-tokens", type=int, default=48)
    p.add_argument("--continuation-decoding", choices=["greedy", "top_p"], default="greedy")
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--dry-run", action="store_true", default=True)
    p.add_argument("--estimate-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--allow-gemma", action="store_true")
    p.add_argument("--confirm-large-run", action="store_true")
    p.add_argument("--output-dir", default=str(OUTPUT_DIR_DEFAULT))
    p.add_argument("--model-cache-dir", default=None)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    return p.parse_args()


PRESETS = {
    "morphology_probe_smoke": {
        "conditions": ["qwen_raw"],
        "candidate_pool_mode": "raw_topk",
        "candidate_pool_size": 50,
        "samples_per_regime": 2,
        "max_prompts": 2,
        "max_steps": 1,
        "generate_continuations": False,
    },
    "morphology_probe_20": {
        "conditions": ["qwen_raw"],
        "candidate_pool_mode": "raw_topk",
        "candidate_pool_size": 100,
        "samples_per_regime": 20,
        "max_prompts": 20,
        "max_steps": 1,
        "generate_continuations": False,
        "continuation_max_new_tokens": 48,
    },
}


def load_condition_meta(condition: str) -> ProbeConditionMeta:
    spec = CONDITION_SPECS.get(condition)
    if spec is None:
        raise SystemExit(f"unknown condition: {condition}")
    return ProbeConditionMeta(
        condition=condition,
        model_name=spec["model_name"],
        prompt_mode=spec["prompt_mode"],
        prompt_path=Path(spec["prompt_path"]),
        notes=spec.get("notes", ""),
        requires_allow_gemma=bool(spec.get("requires_allow_gemma", False)),
    )


def group_by_model(conditions: list[str]) -> dict[str, list[ProbeConditionMeta]]:
    grouped: dict[str, list[ProbeConditionMeta]] = {}
    for condition in conditions:
        meta = load_condition_meta(condition)
        grouped.setdefault(meta.model_name, []).append(meta)
    return grouped


def read_prompts(prompt_path: Path, max_prompts: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not prompt_path.exists():
        raise SystemExit(f"prompt file not found: {prompt_path}")
    for row in read_jsonl(prompt_path):
        rows.append(row)
        if len(rows) >= max_prompts:
            break
    return rows


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clear_probe_memory() -> None:
    clear_memory()


def maybe_gate(args: argparse.Namespace, targets: list[ProbeConditionMeta]) -> None:
    pool_size = int(getattr(args, "candidate_pool_size", 100) or 100)
    if any(t.requires_allow_gemma for t in targets) and not args.allow_gemma:
        raise SystemExit("refusing Gemma conditions without --allow-gemma")
    if pool_size > 200 and not args.confirm_large_run:
        raise SystemExit("refusing candidate_pool_size > 200 without --confirm-large-run")


def resolve_args(args: argparse.Namespace) -> dict[str, Any]:
    preset = PRESETS.get(args.preset or "", {})
    if args.preset and not preset:
        raise SystemExit(f"unknown preset: {args.preset}")
    merged = dict(preset)
    merged["conditions"] = stable_unique(parse_csv_like(args.conditions) or merged.get("conditions", []))
    merged["candidate_pool_mode"] = args.candidate_pool_mode or merged.get("candidate_pool_mode", "raw_topk")
    merged["candidate_pool_size"] = int(args.candidate_pool_size or merged.get("candidate_pool_size", 100))
    merged["samples_per_regime"] = int(args.samples_per_regime or merged.get("samples_per_regime", 20))
    merged["max_prompts"] = int(args.max_prompts or merged.get("max_prompts", 20))
    merged["max_steps"] = int(args.max_steps or merged.get("max_steps", 1))
    merged["generate_continuations"] = bool(args.generate_continuations or merged.get("generate_continuations", False))
    merged["continuation_max_new_tokens"] = int(
        getattr(args, "continuation_max_new_tokens", merged.get("continuation_max_new_tokens", 48))
    )
    merged["continuation_decoding"] = args.continuation_decoding
    merged["top_p"] = float(args.top_p)
    merged["temperature"] = float(args.temperature)
    return merged


def namespace_from_resolved(resolved: dict[str, Any]) -> argparse.Namespace:
    required = [
        "candidate_pool_mode",
        "candidate_pool_size",
        "samples_per_regime",
        "max_prompts",
        "max_steps",
        "generate_continuations",
        "continuation_max_new_tokens",
        "continuation_decoding",
        "top_p",
        "temperature",
    ]
    missing = [key for key in required if key not in resolved]
    if missing:
        raise SystemExit(f"internal error: missing resolved args: {missing}")
    return argparse.Namespace(**resolved)


def split_prompts_by_model(targets: list[ProbeConditionMeta]) -> dict[str, list[ProbeConditionMeta]]:
    grouped: dict[str, list[ProbeConditionMeta]] = defaultdict(list)
    for target in targets:
        grouped[target.model_name].append(target)
    return grouped


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = max(0, min(len(values) - 1, int(round((len(values) - 1) * q))))
    return float(values[idx])


def candidate_regimes_for_family(values: list[float], family: str) -> dict[int, str]:
    if not values:
        return {}
    q25 = quantile(values, 0.25)
    q40 = quantile(values, 0.40)
    q50 = quantile(values, 0.50)
    q60 = quantile(values, 0.60)
    q75 = quantile(values, 0.75)
    mapping = {}
    for idx, value in enumerate(values):
        if family == "token_type":
            continue
        if value <= q25:
            regime = REGIME_FAMILIES[family]["regimes"][0]
        elif value >= q75:
            regime = REGIME_FAMILIES[family]["regimes"][2]
        elif q40 <= value <= q60:
            regime = REGIME_FAMILIES[family]["regimes"][1]
        else:
            regime = REGIME_FAMILIES[family]["regimes"][0] if abs(value - q25) <= abs(value - q75) else REGIME_FAMILIES[family]["regimes"][2]
        mapping[idx] = regime
    return mapping


def assign_regimes(candidate_rows: list[dict[str, Any]]) -> None:
    families = {
        "spread": [float(r["successor_logit_std"]) for r in candidate_rows],
        "concentration": [float(r["successor_top1_prob"]) for r in candidate_rows],
        "tail": [float(r["successor_tail_mass_after_top50"]) for r in candidate_rows],
        "gap": [float(r["successor_gap_1_2"]) for r in candidate_rows],
    }
    family_labels = {family: candidate_regimes_for_family(values, family) for family, values in families.items()}
    token_type_thresholds = {
        regime: quantile([float(row[feature]) for row in candidate_rows], 0.75)
        for regime, feature in TOKEN_TYPE_REGIMES.items()
    }
    for idx, row in enumerate(candidate_rows):
        assigned = []
        for family in ("spread", "concentration", "tail", "gap"):
            regime = family_labels[family].get(idx)
            if regime:
                assigned.append(f"{family}:{regime}")
        for regime, feature in TOKEN_TYPE_REGIMES.items():
            value = float(row[feature])
            threshold = float(token_type_thresholds[regime])
            if value > 0.0 and value >= threshold:
                assigned.append(f"token_type:{regime}")
        row["assigned_regimes"] = ";".join(assigned)


def candidate_pool_from_logits(current_logits, pool_mode: str, pool_size: int, temperature: float) -> list[int]:
    import torch

    logits = current_logits.float()
    if pool_mode == "raw_topk":
        top = torch.topk(logits, k=min(pool_size, logits.numel()))
        return top.indices.tolist()
    probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
    sample_count = min(pool_size, probs.numel())
    sampled = torch.multinomial(probs, num_samples=sample_count, replacement=False)
    return sampled.tolist()


def pool_prompt_rows(prompt_rows: list[dict[str, Any]], max_prompts: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen = set()
    for row in prompt_rows:
        prompt_id = str(row.get("prompt_id", ""))
        if prompt_id in seen:
            continue
        seen.add(prompt_id)
        out.append(row)
        if len(out) >= max_prompts:
            break
    return out


def continuation_text_from_ids(tokenizer, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def continuation_generation(model, tokenizer, device, input_ids, decoding: str, top_p: float, max_new_tokens: int) -> tuple[list[int], str]:
    import torch

    if decoding == "greedy":
        generated = input_ids
        token_ids: list[int] = []
        for _ in range(max_new_tokens):
            logits = model(input_ids=generated).logits[0, -1, :]
            next_id = int(torch.argmax(logits).item())
            token_ids.append(next_id)
            generated = torch.cat([generated, torch.tensor([[next_id]], device=device)], dim=1)
        return token_ids, continuation_text_from_ids(tokenizer, token_ids)
    token_ids: list[int] = []
    generated = input_ids
    for _ in range(max_new_tokens):
        logits = model(input_ids=generated).logits[0, -1, :]
        probs = torch.softmax(logits.float(), dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        csum = torch.cumsum(sorted_probs, dim=0)
        keep = csum <= top_p
        if not torch.any(keep):
            keep = torch.ones_like(sorted_probs, dtype=torch.bool)
        keep[0] = True
        filt_idx = sorted_idx[keep]
        filt_probs = sorted_probs[keep]
        filt_probs = filt_probs / filt_probs.sum().clamp_min(1e-12)
        next_id = int(filt_idx[torch.multinomial(filt_probs, 1).item()].item())
        token_ids.append(next_id)
        generated = torch.cat([generated, torch.tensor([[next_id]], device=device)], dim=1)
    return token_ids, continuation_text_from_ids(tokenizer, token_ids)


def write_rows(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    if not rows:
        return
    safe_mkdir(path.parent)
    write_header = not path.exists()
    with path.open("a" if not write_header else "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def load_completed_candidate_keys(path: Path) -> set[tuple[str, str, int, int, int]]:
    if not path.exists():
        return set()
    completed: set[tuple[str, str, int, int, int]] = set()
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                completed.add(
                    (
                        row["condition"],
                        row["prompt_id"],
                        int(row["step"]),
                        int(row["candidate_rank"]),
                        int(row["candidate_token_id"]),
                    )
                )
            except Exception:
                continue
    return completed


def load_completed_selected_keys(path: Path) -> set[tuple[str, str, str, int, int, int]]:
    if not path.exists():
        return set()
    completed: set[tuple[str, str, str, int, int, int]] = set()
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                completed.add(
                    (
                        row["condition"],
                        row["prompt_id"],
                        row["regime_family"],
                        int(row["step"]),
                        int(row["candidate_rank"]),
                        int(row["candidate_token_id"]),
                    )
                )
            except Exception:
                continue
    return completed


def read_selected_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def load_completed_generation_keys(path: Path) -> set[tuple[str, str, str, int, int, int]]:
    if not path.exists():
        return set()
    completed: set[tuple[str, str, str, int, int, int]] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
                completed.add(
                    (
                        str(row.get("condition", "")),
                        str(row.get("prompt_id", "")),
                        str(row.get("regime", "")),
                        int(row.get("step", 0)),
                        int(row.get("candidate_rank", 0)),
                        int(row.get("candidate_token_id", -1)),
                    )
                )
            except Exception:
                continue
    return completed


def current_and_successor_features(
    model,
    tokenizer,
    input_ids,
    candidate_id: int,
    prompt_text: str,
    recent_text: str,
    device,
    current_logits=None,
) -> dict[str, Any]:
    import torch

    if current_logits is None:
        current_logits = model(input_ids=input_ids).logits[0, -1, :]
    current_probs = torch.softmax(current_logits.float(), dim=-1)
    current_logprobs = torch.log_softmax(current_logits.float(), dim=-1)
    successor_ids = torch.cat([input_ids, torch.tensor([[candidate_id]], device=device)], dim=1)
    successor_logits = model(input_ids=successor_ids).logits[0, -1, :]
    current_stats = morphology_stats(current_logits)
    successor_stats = morphology_stats(successor_logits)
    current_token_masses = token_type_masses(
        tokenizer,
        current_stats["sorted_idx"][:20].detach().cpu(),
        current_stats["sorted_probs"][:20].detach().cpu(),
        prompt_text,
        recent_text,
    )
    successor_token_masses = token_type_masses(
        tokenizer,
        successor_stats["sorted_idx"][:20].detach().cpu(),
        successor_stats["sorted_probs"][:20].detach().cpu(),
        prompt_text,
        recent_text,
    )
    current_rank = int((current_logits > current_logits[candidate_id]).sum().item()) + 1
    return {
        "current_logit_std": float(current_stats["logit_std"]),
        "current_logit_iqr": float(current_stats["logit_iqr"]),
        "current_logit_mad": float(current_stats["logit_mad"]),
        "current_logit_range": float(current_stats["logit_range"]),
        "current_top1_prob": float(current_stats["top1_prob"]),
        "current_top5_mass": float(current_stats["top5_mass"]),
        "current_top10_mass": float(current_stats["top10_mass"]),
        "current_softmax_entropy": float(current_stats["softmax_entropy"]),
        "current_effective_vocab_size": float(current_stats["effective_vocab_size"]),
        "current_tail_mass_after_top50": float(current_stats["tail_mass_after_top50"]),
        "current_tail_mass_after_top100": float(current_stats["tail_mass_after_top100"]),
        "current_nucleus_size_p90": int(current_stats["nucleus_size_p90"]),
        "current_nucleus_size_p95": int(current_stats["nucleus_size_p95"]),
        "current_rank_prob_slope_top50": float(current_stats["rank_prob_slope_top50"]),
        "current_candidate_rank": current_rank,
        "current_candidate_prob": float(current_probs[candidate_id].item()),
        "current_candidate_logprob": float(current_logprobs[candidate_id].item()),
        "successor_logit_std": float(successor_stats["logit_std"]),
        "successor_logit_iqr": float(successor_stats["logit_iqr"]),
        "successor_logit_mad": float(successor_stats["logit_mad"]),
        "successor_logit_range": float(successor_stats["logit_range"]),
        "successor_top1_prob": float(successor_stats["top1_prob"]),
        "successor_top5_mass": float(successor_stats["top5_mass"]),
        "successor_top10_mass": float(successor_stats["top10_mass"]),
        "successor_softmax_entropy": float(successor_stats["softmax_entropy"]),
        "successor_effective_vocab_size": float(successor_stats["effective_vocab_size"]),
        "successor_tail_mass_after_top50": float(successor_stats["tail_mass_after_top50"]),
        "successor_tail_mass_after_top100": float(successor_stats["tail_mass_after_top100"]),
        "successor_nucleus_size_p90": int(successor_stats["nucleus_size_p90"]),
        "successor_nucleus_size_p95": int(successor_stats["nucleus_size_p95"]),
        "successor_rank_prob_slope_top50": float(successor_stats["rank_prob_slope_top50"]),
        "successor_gap_1_2": float(successor_stats["gap_1_2"]),
        "successor_gap_1_5": float(successor_stats["gap_1_5"]),
        "successor_gap_5_20": float(successor_stats["gap_5_20"]),
        "successor_max_gap_top50": float(successor_stats["max_gap_top50"]),
        "successor_large_gap_count_top50": int(successor_stats["large_gap_count_top50"]),
        "successor_gap_std_top50": float(successor_stats["gap_std_top50"]),
        "successor_markup_mass_top20": float(successor_token_masses["markup_mass_top20"]),
        "successor_newline_or_bullet_mass_top20": float(successor_token_masses["newline_or_bullet_mass_top20"]),
        "successor_quote_mass_top20": float(successor_token_masses["quote_mass_top20"]),
        "successor_digit_mass_top20": float(successor_token_masses["digit_mass_top20"]),
        "successor_subword_fragment_mass_top20": float(successor_token_masses["subword_fragment_mass_top20"]),
        "successor_html_marker_mass_top20": float(successor_token_masses["html_marker_mass_top20"]),
        "successor_title_marker_mass_top20": float(successor_token_masses["title_marker_mass_top20"]),
        "successor_choice_marker_mass_top20": float(successor_token_masses["choice_marker_mass_top20"]),
        "successor_prompt_overlap_mass_top20": float(successor_token_masses["prompt_overlap_mass_top20"]),
        "successor_prompt_marker_mass_top20": float(successor_token_masses["prompt_marker_mass_top20"]),
        "successor_prompt_entity_overlap_mass_top20": float(successor_token_masses["prompt_entity_overlap_mass_top20"]),
        "successor_recent_overlap_mass_top20": float(successor_token_masses["recent_overlap_mass_top20"]),
        "current_markup_mass_top20": float(current_token_masses["markup_mass_top20"]),
        "current_newline_or_bullet_mass_top20": float(current_token_masses["newline_or_bullet_mass_top20"]),
        "current_quote_mass_top20": float(current_token_masses["quote_mass_top20"]),
        "current_digit_mass_top20": float(current_token_masses["digit_mass_top20"]),
        "current_subword_fragment_mass_top20": float(current_token_masses["subword_fragment_mass_top20"]),
        "current_html_marker_mass_top20": float(current_token_masses["html_marker_mass_top20"]),
        "current_title_marker_mass_top20": float(current_token_masses["title_marker_mass_top20"]),
        "current_choice_marker_mass_top20": float(current_token_masses["choice_marker_mass_top20"]),
        "current_prompt_overlap_mass_top20": float(current_token_masses["prompt_overlap_mass_top20"]),
        "current_prompt_marker_mass_top20": float(current_token_masses["prompt_marker_mass_top20"]),
        "current_prompt_entity_overlap_mass_top20": float(current_token_masses["prompt_entity_overlap_mass_top20"]),
        "current_recent_overlap_mass_top20": float(current_token_masses["recent_overlap_mass_top20"]),
    }


def regime_buckets(feature_rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in feature_rows:
        assigned = str(row["assigned_regimes"]).split(";")
        for item in assigned:
            family, regime = item.split(":", 1)
            buckets[(family, regime)].append(row)
    return buckets


def select_candidates(candidate_rows: list[dict[str, Any]], samples_per_regime: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    buckets = regime_buckets(candidate_rows)
    for (family, regime), rows in buckets.items():
        by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_prompt[str(row["prompt_id"])].append(row)
        for prompt_rows in by_prompt.values():
            prompt_rows.sort(key=lambda r: int(r["candidate_rank"]))
        chosen: list[dict[str, Any]] = []
        while len(chosen) < samples_per_regime:
            progress = False
            for prompt_id in sorted(by_prompt):
                if len(chosen) >= samples_per_regime:
                    break
                prompt_rows = by_prompt[prompt_id]
                if not prompt_rows:
                    continue
                chosen.append(prompt_rows.pop(0))
                progress = True
            if not progress:
                break
        for rank_within_regime, row in enumerate(chosen, start=1):
            selected.append(
                {
                    "condition": row["condition"],
                    "model_name": row["model_name"],
                    "prompt_id": row["prompt_id"],
                    "prompt_index": row["prompt_index"],
                    "step": row["step"],
                    "prompt": row["prompt"],
                    "regime_family": family,
                    "regime": regime,
                    "selected_rank_within_regime": rank_within_regime,
                    "candidate_rank": row["candidate_rank"],
                    "candidate_token_id": row["candidate_token_id"],
                    "candidate_token": row["candidate_token"],
                    "candidate_current_rank": row["current_candidate_rank"],
                    "candidate_current_prob": row["current_candidate_prob"],
                    "assigned_regimes": row["assigned_regimes"],
                    "successor_logit_std": row["successor_logit_std"],
                    "successor_logit_iqr": row["successor_logit_iqr"],
                    "successor_logit_mad": row["successor_logit_mad"],
                    "successor_logit_range": row["successor_logit_range"],
                    "successor_top1_prob": row["successor_top1_prob"],
                    "successor_top5_mass": row["successor_top5_mass"],
                    "successor_top10_mass": row["successor_top10_mass"],
                    "successor_softmax_entropy": row["successor_softmax_entropy"],
                    "successor_effective_vocab_size": row["successor_effective_vocab_size"],
                    "successor_tail_mass_after_top50": row["successor_tail_mass_after_top50"],
                    "successor_tail_mass_after_top100": row["successor_tail_mass_after_top100"],
                    "successor_nucleus_size_p90": row["successor_nucleus_size_p90"],
                    "successor_nucleus_size_p95": row["successor_nucleus_size_p95"],
                    "successor_rank_prob_slope_top50": row["successor_rank_prob_slope_top50"],
                    "successor_gap_1_2": row["successor_gap_1_2"],
                    "successor_gap_1_5": row["successor_gap_1_5"],
                    "successor_gap_5_20": row["successor_gap_5_20"],
                    "successor_max_gap_top50": row["successor_max_gap_top50"],
                    "successor_large_gap_count_top50": row["successor_large_gap_count_top50"],
                    "successor_gap_std_top50": row["successor_gap_std_top50"],
                    "successor_markup_mass_top20": row["successor_markup_mass_top20"],
                    "successor_newline_or_bullet_mass_top20": row["successor_newline_or_bullet_mass_top20"],
                    "successor_quote_mass_top20": row["successor_quote_mass_top20"],
                    "successor_digit_mass_top20": row["successor_digit_mass_top20"],
                    "successor_subword_fragment_mass_top20": row["successor_subword_fragment_mass_top20"],
                    "successor_html_marker_mass_top20": row["successor_html_marker_mass_top20"],
                    "successor_title_marker_mass_top20": row["successor_title_marker_mass_top20"],
                    "successor_choice_marker_mass_top20": row["successor_choice_marker_mass_top20"],
                    "successor_prompt_overlap_mass_top20": row["successor_prompt_overlap_mass_top20"],
                    "successor_prompt_marker_mass_top20": row["successor_prompt_marker_mass_top20"],
                    "successor_prompt_entity_overlap_mass_top20": row["successor_prompt_entity_overlap_mass_top20"],
                    "successor_recent_overlap_mass_top20": row["successor_recent_overlap_mass_top20"],
                }
            )
    return selected


def probe_prompt(model, tokenizer, device, meta: ProbeConditionMeta, prompt_row: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    import torch

    prompt = str(prompt_row.get("prompt") or prompt_row.get("text") or "")
    prompt_id = str(prompt_row.get("prompt_id") or prompt_row.get("id") or "")
    prompt_index = int(prompt_row.get("prompt_index", 0))
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    current_logits = model(input_ids=input_ids).logits[0, -1, :]
    candidate_ids = candidate_pool_from_logits(
        current_logits,
        args.candidate_pool_mode,
        args.candidate_pool_size,
        args.temperature,
    )
    current_probs = torch.softmax(current_logits.float(), dim=-1)
    current_logprobs = torch.log_softmax(current_logits.float(), dim=-1)
    candidate_rows: list[dict[str, Any]] = []
    for rank, candidate_id in enumerate(candidate_ids, start=1):
        started = time.perf_counter()
        features = current_and_successor_features(
            model,
            tokenizer,
            input_ids,
            candidate_id,
            prompt,
            prompt,
            device,
            current_logits=current_logits,
        )
        candidate_token = safe_decode(tokenizer, candidate_id)
        candidate_rows.append(
            {
                "condition": meta.condition,
                "model_name": meta.model_name,
                "prompt_mode": meta.prompt_mode,
                "prompt_id": prompt_id,
                "prompt_index": prompt_index,
                "step": 0,
                "prompt": prompt,
                "candidate_pool_mode": args.candidate_pool_mode,
                "candidate_pool_size": args.candidate_pool_size,
                "candidate_rank": rank,
                "candidate_token_id": candidate_id,
                "candidate_token": candidate_token,
                "candidate_current_logprob": float(current_logprobs[candidate_id].item()),
                "candidate_current_prob": float(current_probs[candidate_id].item()),
                "candidate_current_rank": int((current_logits > current_logits[candidate_id]).sum().item()) + 1,
                "runtime_seconds": time.perf_counter() - started,
                **features,
            }
        )
    return candidate_rows


def summarize(selected_rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    family_counts = defaultdict(int)
    for row in selected_rows:
        family_counts[(row["regime_family"], row["regime"])] += 1
    prompt_count = len({row["prompt_id"] for row in selected_rows})
    summary_rows: list[dict[str, Any]] = []
    all_regimes: list[tuple[str, str]] = []
    for family, spec in REGIME_FAMILIES.items():
        if family == "token_type":
            continue
        for regime in spec["regimes"]:
            all_regimes.append((family, regime))
    for regime in TOKEN_TYPE_REGIMES:
        all_regimes.append(("token_type", regime))
    for family, regime in all_regimes:
        count = family_counts.get((family, regime), 0)
        selected = [row for row in selected_rows if row["regime_family"] == family and row["regime"] == regime]
        summary_rows.append(
            {
                "regime_family": family,
                "regime": regime,
                "n_selected": count,
                "n_prompts": prompt_count,
                "requested_samples_per_regime": args.samples_per_regime,
                "observed_samples_per_regime": count,
                "shortage": max(0, args.samples_per_regime - count),
                "mean_successor_logit_std": sum(float(r["successor_logit_std"]) for r in selected) / len(selected) if selected else 0.0,
                "mean_successor_top1_prob": sum(float(r["successor_top1_prob"]) for r in selected) / len(selected) if selected else 0.0,
                "mean_successor_tail_mass_after_top50": sum(float(r["successor_tail_mass_after_top50"]) for r in selected) / len(selected) if selected else 0.0,
                "mean_successor_gap_1_2": sum(float(r["successor_gap_1_2"]) for r in selected) / len(selected) if selected else 0.0,
            }
        )
    return summary_rows


def write_report(out_dir: Path, candidate_rows: list[dict[str, Any]], selected_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    report = out_dir / "morphology_stratified_probe_report.md"
    shortages = defaultdict(int)
    for row in summary_rows:
        shortages[(row["regime_family"], row["regime"])] = int(row["shortage"])
    lines = [
        "# Morphology-Stratified Successor Probing",
        "",
        "This is a diagnostic probe, not a quality metric.",
        "",
        "The script builds a candidate pool for each prompt/context, computes candidate-induced successor distribution morphology, stratifies candidates into morphology regimes, and optionally generates short continuations from selected candidates.",
        "",
        f"- conditions: {', '.join(sorted({row['condition'] for row in candidate_rows}) or args.conditions)}",
        f"- candidate_pool_mode: {args.candidate_pool_mode}",
        f"- candidate_pool_size: {args.candidate_pool_size}",
        f"- samples_per_regime_requested: {args.samples_per_regime}",
        f"- samples_per_regime_observed: {len(selected_rows)}",
        f"- total_regime_count: {sum(len(v['regimes']) for family, v in REGIME_FAMILIES.items() if family != 'token_type') + len(TOKEN_TYPE_REGIMES)}",
        f"- global_selection: true",
        f"- max_steps: {args.max_steps}",
        f"- generate_continuations: {args.generate_continuations}",
        "",
        "## Regime shortages",
    ]
    if shortages:
        for (family, regime), shortage in sorted(shortages.items()):
            lines.append(f"- {family}:{regime} shortage={shortage}")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "Selection is global across prompts within each condition, and it prefers one candidate per prompt per regime before reusing the same prompt.",
            "This active probe currently uses step 0 / the initial prompt context only.",
            "Morphology-conditioned candidate selection is diagnostic, not a claim of quality improvement.",
            "Token-type regimes are independent and may overlap.",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def estimate_counts(n_conditions: int, n_prompts: int, candidate_pool_size: int, samples_per_regime: int, generate_continuations: bool) -> dict[str, int]:
    candidate_rows = n_conditions * n_prompts * candidate_pool_size
    total_regime_count = sum(len(v["regimes"]) for family, v in REGIME_FAMILIES.items() if family != "token_type") + len(TOKEN_TYPE_REGIMES)
    selected_rows = n_conditions * samples_per_regime * total_regime_count
    continuation_rows = selected_rows if generate_continuations else 0
    return {
        "total_regime_count": total_regime_count,
        "candidate_rows": candidate_rows,
        "selected_rows": selected_rows,
        "continuation_rows": continuation_rows,
        "total_rows": candidate_rows + selected_rows + continuation_rows,
    }


def main() -> None:
    args = parse_args()
    preset = PRESETS.get(args.preset or "", {})
    if args.preset and not preset:
        raise SystemExit(f"unknown preset: {args.preset}")
    resolved = dict(preset)
    resolved["conditions"] = stable_unique(parse_csv_like(args.conditions) or resolved.get("conditions", []))
    resolved["candidate_pool_mode"] = args.candidate_pool_mode or resolved.get("candidate_pool_mode", "raw_topk")
    resolved["candidate_pool_size"] = int(args.candidate_pool_size or resolved.get("candidate_pool_size", 100))
    resolved["samples_per_regime"] = int(args.samples_per_regime or resolved.get("samples_per_regime", 20))
    resolved["max_prompts"] = int(args.max_prompts or resolved.get("max_prompts", 20))
    resolved["max_steps"] = int(args.max_steps or resolved.get("max_steps", 1))
    if resolved["max_steps"] != 1:
        raise SystemExit("this probe currently supports max_steps == 1 only")
    resolved["generate_continuations"] = bool(args.generate_continuations or resolved.get("generate_continuations", False))
    resolved["continuation_max_new_tokens"] = int(
        getattr(args, "continuation_max_new_tokens", resolved.get("continuation_max_new_tokens", 48))
    )
    resolved["continuation_decoding"] = args.continuation_decoding
    resolved["top_p"] = float(args.top_p)
    resolved["temperature"] = float(args.temperature)
    targets = [load_condition_meta(cond) for cond in resolved["conditions"]]
    maybe_gate(args, targets)
    out_dir = Path(args.output_dir)
    safe_mkdir(out_dir)
    counts = estimate_counts(
        len(resolved["conditions"]),
        resolved["max_prompts"],
        resolved["candidate_pool_size"],
        resolved["samples_per_regime"],
        resolved["generate_continuations"],
    )
    print("conditions:", ", ".join(resolved["conditions"]))
    print("candidate_pool_mode:", resolved["candidate_pool_mode"])
    print("candidate_pool_size:", resolved["candidate_pool_size"])
    print("samples_per_regime:", resolved["samples_per_regime"])
    print("max_prompts:", resolved["max_prompts"])
    print("max_steps:", resolved["max_steps"])
    print("total_regime_count:", counts["total_regime_count"])
    print("generate_continuations:", resolved["generate_continuations"])
    print("continuation_max_new_tokens:", resolved["continuation_max_new_tokens"])
    print("estimated_candidate_rows:", counts["candidate_rows"])
    print("estimated_selected_rows:", counts["selected_rows"])
    print("estimated_continuation_rows:", counts["continuation_rows"])
    print("estimated_total_rows:", counts["total_rows"])
    if args.estimate_only or not args.execute:
        return
    if resolved["candidate_pool_size"] > 200 and not args.confirm_large_run:
        raise SystemExit("refusing candidate_pool_size > 200 without --confirm-large-run")

    try:
        from flatness_decoding.model_io import load_causal_lm
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "execute mode requires torch and the model stack to be installed; "
            "install the project runtime environment before using --execute"
        ) from exc

    grouped = split_prompts_by_model(targets)
    all_candidate_rows: list[dict[str, Any]] = []
    all_selected_rows: list[dict[str, Any]] = []
    all_generation_rows: list[dict[str, Any]] = []
    candidate_path = out_dir / "morphology_stratified_candidate_morphology.csv"
    selected_path = out_dir / "morphology_stratified_selected_candidates.csv"
    generation_path = out_dir / "morphology_stratified_probe_generations.jsonl"
    completed_candidates = load_completed_candidate_keys(candidate_path) if args.resume else set()
    completed_selected = load_completed_selected_keys(selected_path) if args.resume else set()
    completed_generation = load_completed_generation_keys(generation_path) if args.resume else set()

    for model_name, metas in grouped.items():
        model, tokenizer, device = load_causal_lm(
            model_name,
            device_name=args.device,
            dtype_name=args.dtype,
            cache_dir=args.model_cache_dir,
            local_files_only=args.local_files_only,
            trust_remote_code=True,
        )
        try:
            for meta in metas:
                condition_candidate_rows: list[dict[str, Any]] = []
                prompt_rows = pool_prompt_rows(read_prompts(meta.prompt_path, resolved["max_prompts"]), resolved["max_prompts"])
                for prompt_row in prompt_rows:
                    probe_args = namespace_from_resolved(resolved)
                    candidate_rows = probe_prompt(model, tokenizer, device, meta, prompt_row, probe_args)
                    fresh_candidate_rows = []
                    for row in candidate_rows:
                        key = (
                            row["condition"],
                            row["prompt_id"],
                            row["step"],
                            row["candidate_rank"],
                            row["candidate_token_id"],
                        )
                        if key in completed_candidates:
                            continue
                        fresh_candidate_rows.append(row)
                    condition_candidate_rows.extend(fresh_candidate_rows)
                    clear_probe_memory()
                assign_regimes(condition_candidate_rows)
                selected_rows = select_candidates(condition_candidate_rows, resolved["samples_per_regime"])
                fresh_selected_rows = []
                for row in selected_rows:
                    key = (
                        row["condition"],
                        row["prompt_id"],
                        row["regime_family"],
                        int(row["step"]),
                        int(row["candidate_rank"]),
                        int(row["candidate_token_id"]),
                    )
                    if key in completed_selected:
                        continue
                    fresh_selected_rows.append(row)
                fresh_generation_rows = []
                if resolved["generate_continuations"]:
                    for row in fresh_selected_rows:
                        prompt_ids = tokenizer(str(row.get("prompt", "")), return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
                        selected_prefix_ids = torch.cat(
                            [prompt_ids, torch.tensor([[int(row["candidate_token_id"])]], device=device)],
                            dim=1,
                        )
                        generated_ids, continuation_text = continuation_generation(
                            model,
                            tokenizer,
                            device,
                            selected_prefix_ids,
                            probe_args.continuation_decoding,
                            probe_args.top_p,
                            probe_args.continuation_max_new_tokens,
                        )
                        row_prompt = str(row.get("prompt", ""))
                        fresh_generation_rows.append(
                            {
                                "condition": row["condition"],
                                "model_name": row["model_name"],
                                "prompt_id": row["prompt_id"],
                                "step": row["step"],
                                "regime": row["regime"],
                                "candidate_rank": row["candidate_rank"],
                                "candidate_token_id": int(row["candidate_token_id"]),
                                "candidate_token": row["candidate_token"],
                                "prompt": row_prompt,
                                "selected_prefix": f"{row_prompt}{row['candidate_token']}",
                                "continuation_text": continuation_text,
                                "generation_settings": json.dumps(
                                    {
                                        "generate_continuations": True,
                                        "continuation_max_new_tokens": probe_args.continuation_max_new_tokens,
                                        "continuation_decoding": probe_args.continuation_decoding,
                                        "top_p": probe_args.top_p,
                                        "candidate_pool_mode": probe_args.candidate_pool_mode,
                                        "candidate_pool_size": probe_args.candidate_pool_size,
                                    },
                                    ensure_ascii=False,
                                ),
                                "output_mode_label": output_mode_label([continuation_text], row_prompt),
                            }
                        )
                fresh_generation_rows = [row for row in fresh_generation_rows if (
                    str(row.get("condition", "")),
                    str(row.get("prompt_id", "")),
                    str(row.get("regime", "")),
                    int(row.get("step", 0)),
                    int(row.get("candidate_rank", 0)),
                    int(row.get("candidate_token_id", -1)),
                ) not in completed_generation]
                write_rows(candidate_path, condition_candidate_rows, MORPHOLOGY_COLUMNS)
                write_rows(selected_path, fresh_selected_rows, SELECTED_COLUMNS)
                if fresh_generation_rows:
                    with generation_path.open("a" if generation_path.exists() else "w", encoding="utf-8") as fh:
                        for row in fresh_generation_rows:
                            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                all_candidate_rows.extend(condition_candidate_rows)
                all_selected_rows.extend(fresh_selected_rows)
                all_generation_rows.extend(fresh_generation_rows)
        finally:
            del model, tokenizer
            clear_probe_memory()

    full_selected_rows = read_selected_rows(selected_path) if selected_path.exists() else all_selected_rows
    summary_rows = summarize(full_selected_rows, namespace_from_resolved(resolved))
    write_rows(out_dir / "morphology_stratified_probe_summary.csv", summary_rows, list(summary_rows[0].keys()) if summary_rows else ["regime_family", "regime"])
    write_report(out_dir, all_candidate_rows, full_selected_rows, summary_rows, namespace_from_resolved(resolved))


if __name__ == "__main__":
    main()
