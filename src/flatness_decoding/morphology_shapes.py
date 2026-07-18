from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .flatness import flatness_std_from_logits


@dataclass(frozen=True)
class ShapeStats:
    logit_std: float
    top1_prob: float
    top5_mass: float
    top10_mass: float
    softmax_entropy: float
    effective_vocab_size: float
    tail_mass_after_top20: float
    tail_mass_after_top50: float
    tail_mass_after_top100: float
    nucleus_size_p90: int
    nucleus_size_p95: int
    gap_1_2: float
    gap_1_5: float
    gap_5_20: float
    max_gap_top50: float
    large_gap_count_top50: int
    gap_std_top50: float
    rank_prob_slope_top20: float
    rank_prob_slope_top50: float
    rank_prob_slope_top100: float
    top20_mass: float
    hhi_top50: float
    markup_mass_top20: float = 0.0
    newline_or_bullet_mass_top20: float = 0.0
    quote_mass_top20: float = 0.0
    digit_mass_top20: float = 0.0
    subword_fragment_mass_top20: float = 0.0
    html_marker_mass_top20: float = 0.0
    title_marker_mass_top20: float = 0.0
    choice_marker_mass_top20: float = 0.0
    prompt_overlap_mass_top20: float = 0.0
    prompt_marker_mass_top20: float = 0.0
    prompt_entity_overlap_mass_top20: float = 0.0
    recent_overlap_mass_top20: float = 0.0


def _quantile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return 0.0
    return float(torch.quantile(values.float(), torch.tensor(q, device=values.device)).item())


def _slope_topk(sorted_probs: torch.Tensor, k: int) -> float:
    if sorted_probs.numel() <= 1:
        return 0.0
    n = min(int(sorted_probs.numel()), k)
    ranks = torch.arange(1, n + 1, device=sorted_probs.device, dtype=torch.float32)
    y = torch.log(sorted_probs[:n].clamp_min(1e-12))
    x = ranks
    vx = torch.var(x, unbiased=False).clamp_min(1e-12)
    cov = torch.cov(torch.stack([x, y]))[0, 1]
    return float((cov / vx).item())


def morphology_stats_extended(logits: torch.Tensor) -> ShapeStats:
    logits_f = logits.float()
    probs = torch.softmax(logits_f, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, stable=True)
    sorted_logits = logits_f[sorted_idx]
    n = int(sorted_logits.numel())

    entropy = float((-(probs * torch.log(probs.clamp_min(1e-12))).sum()).item())
    csum = torch.cumsum(sorted_probs, dim=0)
    tail = lambda k: float(max(0.0, 1.0 - sorted_probs[:k].sum().item())) if n else 0.0
    nucleus = lambda p: int((csum < p).sum().item()) + 1 if n else 0

    gaps = (sorted_logits[:-1] - sorted_logits[1:])[:50] if n > 1 else torch.tensor([], device=logits.device)
    top20_mass = float(sorted_probs[:20].sum().item()) if n >= 20 else float(sorted_probs.sum().item())
    hhi_top50 = float((sorted_probs[:50] ** 2).sum().item()) if n else 0.0

    return ShapeStats(
        logit_std=float(flatness_std_from_logits(logits_f).item()),
        top1_prob=float(sorted_probs[0].item()) if n else 0.0,
        top5_mass=float(sorted_probs[:5].sum().item()) if n >= 5 else float(sorted_probs.sum().item()),
        top10_mass=float(sorted_probs[:10].sum().item()) if n >= 10 else float(sorted_probs.sum().item()),
        softmax_entropy=entropy,
        effective_vocab_size=float(torch.exp(torch.tensor(entropy)).item()),
        tail_mass_after_top20=tail(20),
        tail_mass_after_top50=tail(50),
        tail_mass_after_top100=tail(100),
        nucleus_size_p90=nucleus(0.9),
        nucleus_size_p95=nucleus(0.95),
        gap_1_2=float((sorted_logits[0] - sorted_logits[1]).item()) if n > 1 else 0.0,
        gap_1_5=float((sorted_logits[0] - sorted_logits[4]).item()) if n > 4 else 0.0,
        gap_5_20=float((sorted_logits[4] - sorted_logits[19]).item()) if n > 19 else 0.0,
        max_gap_top50=float(gaps.max().item()) if gaps.numel() else 0.0,
        large_gap_count_top50=int((gaps > 1.0).sum().item()) if gaps.numel() else 0,
        gap_std_top50=float(torch.std(gaps, unbiased=False).item()) if gaps.numel() > 1 else 0.0,
        rank_prob_slope_top20=_slope_topk(sorted_probs, 20),
        rank_prob_slope_top50=_slope_topk(sorted_probs, 50),
        rank_prob_slope_top100=_slope_topk(sorted_probs, 100),
        top20_mass=top20_mass,
        hhi_top50=hhi_top50,
    )


def detect_spread(stats: ShapeStats) -> dict[str, bool]:
    return {
        "low_spread": stats.logit_std <= 2.5 or stats.softmax_entropy >= 5.0,
        "mid_spread": 2.5 < stats.logit_std < 4.0 and 3.0 <= stats.softmax_entropy < 5.0,
        "high_spread": stats.logit_std >= 4.0 and stats.softmax_entropy < 5.0,
    }


def detect_concentration(stats: ShapeStats) -> dict[str, bool]:
    return {
        "low_concentration": stats.top1_prob <= 0.15 and stats.gap_1_2 <= 0.2 and stats.softmax_entropy >= 2.0,
        "mid_concentration": (
            0.15 < stats.top1_prob < 0.35
            or 0.2 < stats.gap_1_2 < 1.0
            or 1.5 < stats.softmax_entropy < 3.5
        ),
        "high_concentration": stats.top1_prob >= 0.35 or stats.top5_mass >= 0.70 or stats.softmax_entropy < 1.5,
    }


def detect_tail(stats: ShapeStats) -> dict[str, bool]:
    return {
        "thin_tail": stats.tail_mass_after_top20 <= 0.01 or stats.nucleus_size_p90 <= 10,
        "mid_tail": 0.01 < stats.tail_mass_after_top20 < 0.02 or 10 < stats.nucleus_size_p90 < 30,
        "heavy_tail": stats.tail_mass_after_top20 >= 0.02 or stats.tail_mass_after_top50 >= 0.05 or stats.nucleus_size_p90 >= 30,
    }


def detect_gap(stats: ShapeStats) -> dict[str, bool]:
    return {
        "small_gap": stats.gap_1_2 <= 0.5 and stats.large_gap_count_top50 <= 1,
        "mid_gap": 0.5 < stats.gap_1_2 < 1.5,
        "large_gap": stats.gap_1_2 >= 1.5 or stats.large_gap_count_top50 >= 3,
    }


def detect_rank_shape(stats: ShapeStats) -> dict[str, bool]:
    return {
        "rank_slope_flat": stats.rank_prob_slope_top50 >= -0.05,
        "rank_slope_mid": -0.20 < stats.rank_prob_slope_top50 < -0.05,
        "rank_slope_steep": stats.rank_prob_slope_top50 <= -0.20,
    }


def detect_peak_structure(stats: ShapeStats) -> dict[str, bool]:
    return {
        "spiky_top": stats.top1_prob >= 0.60 and stats.gap_1_2 >= 1.0,
        "smooth_top": stats.top1_prob <= 0.20 and stats.gap_1_2 <= 0.15,
        "multi_peak": stats.large_gap_count_top50 >= 2 and stats.gap_std_top50 >= 0.5,
    }


def detect_shape_family(logits: torch.Tensor) -> dict[str, bool]:
    stats = morphology_stats_extended(logits)
    out: dict[str, bool] = {}
    out.update(detect_spread(stats))
    out.update(detect_concentration(stats))
    out.update(detect_tail(stats))
    out.update(detect_gap(stats))
    out.update(detect_rank_shape(stats))
    out.update(detect_peak_structure(stats))
    return out


def explain_shape(stats: ShapeStats) -> dict[str, Any]:
    return {
        "spread": "broad" if stats.softmax_entropy > 4.5 else "narrow" if stats.softmax_entropy < 3.5 else "mid",
        "concentration": "peaky" if stats.top1_prob >= 0.35 else "flat" if stats.top1_prob <= 0.12 else "mid",
        "tail": "heavy" if stats.tail_mass_after_top50 >= 0.30 else "thin" if stats.tail_mass_after_top50 <= 0.15 else "mid",
        "gap": "large" if stats.gap_1_2 >= 1.5 else "small" if stats.gap_1_2 <= 0.5 else "mid",
        "rank_slope": "steep" if stats.rank_prob_slope_top50 <= -0.20 else "flat" if stats.rank_prob_slope_top50 >= -0.05 else "mid",
        "fragment_proxy": "high" if stats.hhi_top50 >= 0.08 else "mid" if stats.hhi_top50 >= 0.04 else "low",
        "peak_structure": "spiky" if stats.top1_prob >= 0.60 else "smooth" if stats.top1_prob <= 0.20 else "multi_or_mid",
    }
