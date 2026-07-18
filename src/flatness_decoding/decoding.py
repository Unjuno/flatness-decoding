from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch

from .flatness import flatness_std_from_logits, selection_gain
from .morphology_shapes import (
    detect_concentration,
    detect_gap,
    detect_peak_structure,
    detect_rank_shape,
    detect_spread,
    detect_tail,
    morphology_stats_extended,
)
from .sampling import sample_candidates_from_logits

DecodeMode = Literal[
    "minflat",
    "maxflat",
    "random",
    "midflat",
    "centerflat_q25",
    "centerflat_q50",
    "centerflat_q75",
    "highprob_centerflat",
    "low_spread",
    "mid_spread",
    "high_spread",
    "low_concentration",
    "mid_concentration",
    "high_concentration",
    "thin_tail",
    "mid_tail",
    "heavy_tail",
    "small_gap",
    "mid_gap",
    "large_gap",
    "rank_slope_flat",
    "rank_slope_mid",
    "rank_slope_steep",
    "spiky_top",
    "smooth_top",
    "multi_peak",
]


@dataclass
class SelectionInfo:
    chosen_index: int
    chosen_id: int
    selector_target_quantile: float | None
    allowed_candidate_indices: list[int]
    selection_fallback: bool

    def __iter__(self):
        """Preserve the historical ``index, token = choose_candidate(...)`` API."""
        yield self.chosen_index
        yield self.chosen_id


@dataclass
class StepRecord:
    step: int
    method: str
    n_trials: int
    proposal_temperature: float
    candidate_ids: list[int]
    candidate_unique_count: int
    candidate_flatness: list[float]
    candidate_current_logprobs: list[float]
    candidate_current_probs: list[float]
    candidate_current_ranks: list[int]
    chosen_index: int
    chosen_id: int
    chosen_flatness: float
    chosen_current_logprob: float
    chosen_current_prob: float
    chosen_current_rank: int
    chosen_flatness_quantile: float
    selector_target_quantile: float | None
    allowed_candidate_indices: list[int]
    allowed_candidate_count: int
    selection_fallback: bool
    mean_candidate_flatness: float
    min_candidate_flatness: float
    max_candidate_flatness: float
    selection_gain: float
    mean_candidate_current_logprob: float
    min_candidate_current_rank: int
    max_candidate_current_rank: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@torch.no_grad()
def next_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Return next-token logits for a single sequence.

    Args:
        input_ids: shape `[1, seq_len]`.
    """
    outputs = model(input_ids=input_ids)
    return outputs.logits[0, -1, :]


def to_legacy_cache(past_key_values: Any) -> Any:
    """Return a stable legacy tuple cache when Transformers exposes one."""
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()
    return past_key_values


def expand_batch_tensor(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Materialize a batch-1 cache tensor as `batch_size`.

    This is intentionally `repeat` rather than `expand`: some model/cache paths
    are sensitive to zero-stride expanded views.
    """
    if tensor.dim() == 0 or tensor.shape[0] != 1 or batch_size == 1:
        return tensor
    return tensor.repeat(batch_size, *([1] * (tensor.dim() - 1)))


def expand_past_key_values(past_key_values: Any, batch_size: int) -> Any:
    """Expand legacy tuple/list KV caches from batch size 1 to `batch_size`.

    This helper is kept for experimental cached-lookahead tests. Official runs
    default to full-prefix micro-batched candidate evaluation because cached
    lookahead did not pass full-prefix equivalence in the current GPT-2 stack.
    """
    past_key_values = to_legacy_cache(past_key_values)
    if past_key_values is None or batch_size == 1:
        return past_key_values
    if torch.is_tensor(past_key_values):
        return expand_batch_tensor(past_key_values, batch_size)
    if isinstance(past_key_values, tuple):
        return tuple(expand_past_key_values(item, batch_size) for item in past_key_values)
    if isinstance(past_key_values, list):
        return [expand_past_key_values(item, batch_size) for item in past_key_values]
    if hasattr(past_key_values, "batch_repeat_interleave"):
        return past_key_values.batch_repeat_interleave(batch_size)
    return past_key_values


def cache_sequence_length(past_key_values: Any) -> int | None:
    """Infer cached sequence length from a legacy GPT-style KV cache."""
    past_key_values = to_legacy_cache(past_key_values)
    if past_key_values is None:
        return None
    try:
        first_layer = past_key_values[0]
        key = first_layer[0]
        if torch.is_tensor(key) and key.dim() >= 3:
            return int(key.shape[-2])
    except Exception:
        return None
    return None


@torch.no_grad()
def evaluate_candidate_batch_next_flatness(
    model,
    input_ids: torch.Tensor,
    candidate_ids: torch.Tensor,
    past_key_values: Any | None = None,
) -> torch.Tensor:
    """Evaluate next-state logit flatness for one candidate batch."""
    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids must have shape [1, seq_len]")
    if candidate_ids.dim() != 1:
        raise ValueError("candidate_ids must have shape [n]")

    device = input_ids.device
    n = int(candidate_ids.shape[0])

    if past_key_values is not None:
        stable_past = to_legacy_cache(past_key_values)
        expanded_past = expand_past_key_values(stable_past, n)
        past_len = cache_sequence_length(stable_past)
        model_kwargs: dict[str, torch.Tensor | Any] = {
            "input_ids": candidate_ids.to(device).view(n, 1),
            "past_key_values": expanded_past,
            "use_cache": False,
        }
        if past_len is not None:
            model_kwargs["attention_mask"] = torch.ones(n, past_len + 1, dtype=torch.long, device=device)
            model_kwargs["position_ids"] = torch.full((n, 1), past_len, dtype=torch.long, device=device)
        outputs = model(**model_kwargs)
    else:
        repeated = input_ids.repeat(n, 1)
        appended = torch.cat([repeated, candidate_ids.to(device).view(n, 1)], dim=1)
        outputs = model(input_ids=appended)

    logits_next = outputs.logits[:, -1, :]
    return flatness_std_from_logits(logits_next)


@torch.no_grad()
def evaluate_candidates_next_flatness(
    model,
    input_ids: torch.Tensor,
    candidate_ids: torch.Tensor,
    candidate_eval_batch_size: int | None = None,
    past_key_values: Any | None = None,
) -> torch.Tensor:
    """Evaluate next-state logit flatness for each candidate.

    Official experiments pass `past_key_values=None`, using exact full-prefix
    evaluation while still reducing peak memory through candidate micro-batches.
    The cached path is retained for tests/experimentation but is not enabled by
    default because it did not pass equivalence in the current GPT-2 stack.
    """
    if candidate_eval_batch_size is None or candidate_eval_batch_size <= 0:
        candidate_eval_batch_size = int(candidate_ids.shape[0])

    stable_past = to_legacy_cache(past_key_values)
    chunks: list[torch.Tensor] = []
    for start in range(0, int(candidate_ids.shape[0]), candidate_eval_batch_size):
        end = start + candidate_eval_batch_size
        chunk = candidate_ids[start:end]
        chunks.append(
            evaluate_candidate_batch_next_flatness(
                model,
                input_ids,
                chunk,
                past_key_values=stable_past,
            )
        )
    return torch.cat(chunks, dim=0)


def current_candidate_distribution_stats(
    logits_now: torch.Tensor,
    candidate_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return current-distribution logprobs, probs, and strict-greater ranks."""
    logits_cpu = logits_now.detach().float().cpu()
    candidate_ids_cpu = candidate_ids.detach().cpu().long()
    logprobs_cpu = torch.log_softmax(logits_cpu, dim=-1)
    candidate_logits = logits_cpu[candidate_ids_cpu]
    candidate_logprobs = logprobs_cpu[candidate_ids_cpu]
    candidate_ranks = (logits_cpu.view(1, -1) > candidate_logits.view(-1, 1)).sum(dim=1).long() + 1
    return candidate_logprobs, candidate_logprobs.exp(), candidate_ranks


def selector_quantile(mode: DecodeMode) -> float | None:
    if mode in {"midflat", "centerflat_q50", "highprob_centerflat"}:
        return 0.5
    if mode == "centerflat_q25":
        return 0.25
    if mode == "centerflat_q75":
        return 0.75
    return None


def _shape_label_set(stats) -> set[str]:
    labels: set[str] = set()
    for fn in (detect_spread, detect_concentration, detect_tail, detect_gap, detect_rank_shape, detect_peak_structure):
        labels.update(k for k, v in fn(stats).items() if v)
    return labels


def _shape_target_label(mode: DecodeMode) -> str | None:
    shape_modes = {
        "low_spread", "mid_spread", "high_spread",
        "low_concentration", "mid_concentration", "high_concentration",
        "thin_tail", "mid_tail", "heavy_tail",
        "small_gap", "mid_gap", "large_gap",
        "rank_slope_flat", "rank_slope_mid", "rank_slope_steep",
        "spiky_top", "smooth_top", "multi_peak",
    }
    return mode if mode in shape_modes else None


def choose_quantile_index(flatness_cpu: torch.Tensor, candidate_indices: torch.Tensor, q: float) -> int:
    subset = flatness_cpu[candidate_indices]
    if subset.numel() == 1:
        return int(candidate_indices[0].item())
    target = torch.quantile(subset.float(), q)
    local_idx = int(torch.argmin((subset.float() - target).abs()).item())
    return int(candidate_indices[local_idx].item())


def flatness_quantile_position(flatness_cpu: torch.Tensor, chosen_index: int) -> float:
    n = int(flatness_cpu.shape[0])
    if n <= 1:
        return 0.0
    order = torch.argsort(flatness_cpu.float(), stable=True)
    matches = (order == chosen_index).nonzero(as_tuple=False)
    if matches.numel() == 0:
        return 0.0
    rank = int(matches[0].item())
    return rank / float(n - 1)


def allowed_highprob_indices(
    candidate_logprobs: torch.Tensor,
    candidate_ranks: torch.Tensor,
    highprob_logprob_margin: float | None,
    highprob_rank_limit: int | None,
) -> tuple[list[int], bool]:
    n = int(candidate_logprobs.shape[0])
    mask = torch.ones(n, dtype=torch.bool)
    if highprob_logprob_margin is not None:
        threshold = candidate_logprobs.max() - float(highprob_logprob_margin)
        mask &= candidate_logprobs >= threshold
    if highprob_rank_limit is not None:
        mask &= candidate_ranks <= int(highprob_rank_limit)
    indices = [int(x) for x in mask.nonzero(as_tuple=False).flatten().tolist()]
    if indices:
        return indices, False
    return list(range(n)), True


def choose_candidate(
    candidate_ids: torch.Tensor,
    flatness_values: torch.Tensor,
    mode: DecodeMode,
    candidate_current_logprobs: torch.Tensor | None = None,
    candidate_current_ranks: torch.Tensor | None = None,
    highprob_logprob_margin: float | None = None,
    highprob_rank_limit: int | None = None,
) -> SelectionInfo:
    if candidate_ids.shape[0] < 1:
        raise ValueError("candidate_ids must contain at least one candidate")
    flatness_cpu = flatness_values.detach().cpu()
    candidate_cpu = candidate_ids.detach().cpu()
    all_indices = list(range(int(candidate_cpu.shape[0])))
    allowed_indices = all_indices
    fallback = False

    if mode == "minflat":
        idx = int(torch.argmin(flatness_cpu).item())
    elif mode == "maxflat":
        idx = int(torch.argmax(flatness_cpu).item())
    elif mode == "random":
        idx = 0 if candidate_cpu.shape[0] == 1 else int(torch.randint(0, candidate_cpu.shape[0], (1,)).item())
    elif mode in {"midflat", "centerflat_q25", "centerflat_q50", "centerflat_q75"}:
        q = selector_quantile(mode)
        if q is None:
            raise ValueError(f"Missing selector quantile for mode: {mode}")
        idx = choose_quantile_index(flatness_cpu, torch.tensor(all_indices, dtype=torch.long), q)
    elif mode == "highprob_centerflat":
        if candidate_current_logprobs is None or candidate_current_ranks is None:
            raise ValueError("highprob_centerflat requires current logprobs and ranks")
        allowed_indices, fallback = allowed_highprob_indices(
            candidate_current_logprobs.detach().cpu(),
            candidate_current_ranks.detach().cpu(),
            highprob_logprob_margin,
            highprob_rank_limit,
        )
        idx = choose_quantile_index(flatness_cpu, torch.tensor(allowed_indices, dtype=torch.long), 0.5)
    elif _shape_target_label(mode) is not None:
        # Shape modes are selected by candidate-induced successor logits.
        # The caller passes flatness_values as a side channel if desired, but
        # the actual shape selection is handled higher up in generate_flatness_controlled.
        raise ValueError("shape-based modes must be resolved by generate_flatness_controlled")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return SelectionInfo(
        chosen_index=idx,
        chosen_id=int(candidate_cpu[idx].item()),
        selector_target_quantile=selector_quantile(mode),
        allowed_candidate_indices=allowed_indices,
        selection_fallback=fallback,
    )


def banned_candidate_tokens(tokenizer) -> set[int]:
    """Return special token IDs to ban from proposal candidates."""
    banned: set[int] = set()
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    if pad_id is not None and pad_id != eos_id:
        banned.add(int(pad_id))
    return banned


@torch.no_grad()
def generate_flatness_controlled(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    n_trials: int = 8,
    proposal_temperature: float = 1.0,
    method: DecodeMode = "minflat",
    device: torch.device | None = None,
    seed: int | None = None,
    candidate_eval_batch_size: int | None = None,
    use_kv_cache_lookahead: bool = False,
    highprob_logprob_margin: float | None = None,
    highprob_rank_limit: int | None = None,
) -> tuple[str, list[StepRecord]]:
    """Generate text with Random-N and flatness-controlled decoding.

    `method="minflat"` is Flatness-N. Shape-based modes select by candidate
    successor distribution morphology instead of successor flatness. Official
    runs default to exact full-prefix candidate evaluation with micro-batching.
    """
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if device is None:
        device = next(model.parameters()).device

    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    records: list[StepRecord] = []

    logits_now = next_logits(model, input_ids)
    past_key_values = None
    if use_kv_cache_lookahead:
        prefix_outputs = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=True)
        logits_now = prefix_outputs.logits[0, -1, :]
        past_key_values = to_legacy_cache(getattr(prefix_outputs, "past_key_values", None))

    banned = banned_candidate_tokens(tokenizer)

    for step in range(max_new_tokens):
        candidate_ids = sample_candidates_from_logits(
            logits=logits_now,
            n=n_trials,
            temperature=proposal_temperature,
            replacement=True,
            banned_token_ids=banned,
        )
        candidate_logprobs, candidate_probs, candidate_ranks = current_candidate_distribution_stats(logits_now, candidate_ids)

        flatness_values = evaluate_candidates_next_flatness(
            model=model,
            input_ids=input_ids,
            candidate_ids=candidate_ids,
            candidate_eval_batch_size=candidate_eval_batch_size,
            past_key_values=past_key_values if use_kv_cache_lookahead else None,
        )

        # Shape-based selection evaluates each candidate's successor logits directly.
        if _shape_target_label(method) is not None:
            candidate_shape_stats = []
            for cand_id in candidate_ids.tolist():
                successor_ids = torch.cat([input_ids, torch.tensor([[int(cand_id)]], device=device)], dim=1)
                successor_logits = model(input_ids=successor_ids).logits[0, -1, :]
                candidate_shape_stats.append(morphology_stats_extended(successor_logits))
            target = _shape_target_label(method)
            shape_scores = []
            for stats in candidate_shape_stats:
                labels = _shape_label_set(stats)
                shape_scores.append(1.0 if target in labels else 0.0)
            if any(shape_scores):
                idx = int(torch.tensor(shape_scores).argmax().item())
            else:
                # Fallback: choose the candidate closest to the target's primary metric.
                if method == "low_spread":
                    key = lambda s: s.logit_std
                    idx = int(torch.tensor([key(s) for s in candidate_shape_stats]).argmin().item())
                elif method == "mid_spread":
                    key = lambda s: abs(s.logit_std - 3.0)
                    idx = int(torch.tensor([key(s) for s in candidate_shape_stats]).argmin().item())
                elif method == "high_spread":
                    key = lambda s: -s.logit_std
                    idx = int(torch.tensor([key(s) for s in candidate_shape_stats]).argmin().item())
                elif method == "low_concentration":
                    key = lambda s: s.top1_prob
                    idx = int(torch.tensor([key(s) for s in candidate_shape_stats]).argmin().item())
                elif method == "mid_concentration":
                    key = lambda s: abs(s.top1_prob - 0.25)
                    idx = int(torch.tensor([key(s) for s in candidate_shape_stats]).argmin().item())
                elif method == "high_concentration":
                    key = lambda s: -s.top1_prob
                    idx = int(torch.tensor([key(s) for s in candidate_shape_stats]).argmin().item())
                elif method == "thin_tail":
                    key = lambda s: s.tail_mass_after_top50
                    idx = int(torch.tensor([key(s) for s in candidate_shape_stats]).argmin().item())
                elif method == "mid_tail":
                    key = lambda s: abs(s.tail_mass_after_top50 - 0.15)
                    idx = int(torch.tensor([key(s) for s in candidate_shape_stats]).argmin().item())
                elif method == "heavy_tail":
                    key = lambda s: -s.tail_mass_after_top50
                    idx = int(torch.tensor([key(s) for s in candidate_shape_stats]).argmin().item())
                elif method == "small_gap":
                    key = lambda s: s.gap_1_2
                    idx = int(torch.tensor([key(s) for s in candidate_shape_stats]).argmin().item())
                elif method == "mid_gap":
                    key = lambda s: abs(s.gap_1_2 - 0.75)
                    idx = int(torch.tensor([key(s) for s in candidate_shape_stats]).argmin().item())
                elif method == "large_gap":
                    key = lambda s: -s.gap_1_2
                    idx = int(torch.tensor([key(s) for s in candidate_shape_stats]).argmin().item())
                elif method == "rank_slope_flat":
                    key = lambda s: -s.rank_prob_slope_top50
                    idx = int(torch.tensor([key(s) for s in candidate_shape_stats]).argmin().item())
                elif method == "rank_slope_mid":
                    key = lambda s: abs(s.rank_prob_slope_top50 + 0.1)
                    idx = int(torch.tensor([key(s) for s in candidate_shape_stats]).argmin().item())
                elif method == "rank_slope_steep":
                    key = lambda s: s.rank_prob_slope_top50
                    idx = int(torch.tensor([key(s) for s in candidate_shape_stats]).argmin().item())
                elif method == "spiky_top":
                    key = lambda s: -s.top1_prob
                    idx = int(torch.tensor([key(s) for s in candidate_shape_stats]).argmin().item())
                elif method == "smooth_top":
                    key = lambda s: s.top1_prob
                    idx = int(torch.tensor([key(s) for s in candidate_shape_stats]).argmin().item())
                elif method == "multi_peak":
                    key = lambda s: -s.large_gap_count_top50
                    idx = int(torch.tensor([key(s) for s in candidate_shape_stats]).argmin().item())
                else:
                    raise ValueError(f"Unknown shape mode: {method}")
            selection = SelectionInfo(
                chosen_index=idx,
                chosen_id=int(candidate_ids[idx].item()),
                selector_target_quantile=None,
                allowed_candidate_indices=list(range(int(candidate_ids.shape[0]))),
                selection_fallback=not bool(shape_scores and any(shape_scores)),
            )
        else:
            selection = choose_candidate(
                candidate_ids=candidate_ids,
                flatness_values=flatness_values,
                mode=method,
                candidate_current_logprobs=candidate_logprobs,
                candidate_current_ranks=candidate_ranks,
                highprob_logprob_margin=highprob_logprob_margin,
                highprob_rank_limit=highprob_rank_limit,
            )
        chosen_index = selection.chosen_index
        chosen_id = selection.chosen_id

        chosen_tensor = torch.tensor([[chosen_id]], device=device)
        if use_kv_cache_lookahead and past_key_values is not None:
            past_len = cache_sequence_length(past_key_values)
            model_kwargs: dict[str, torch.Tensor | Any] = {
                "input_ids": chosen_tensor,
                "past_key_values": past_key_values,
                "use_cache": True,
            }
            if past_len is not None:
                model_kwargs["attention_mask"] = torch.ones(1, past_len + 1, dtype=torch.long, device=device)
                model_kwargs["position_ids"] = torch.full((1, 1), past_len, dtype=torch.long, device=device)
            advance_outputs = model(**model_kwargs)
            logits_now = advance_outputs.logits[0, -1, :]
            past_key_values = to_legacy_cache(getattr(advance_outputs, "past_key_values", None))
            input_ids = torch.cat([input_ids, chosen_tensor], dim=1)
        else:
            input_ids = torch.cat([input_ids, chosen_tensor], dim=1)
            logits_now = next_logits(model, input_ids)

        gain = selection_gain(flatness_values)
        flatness_cpu = flatness_values.detach().cpu()
        records.append(
            StepRecord(
                step=step,
                method=method,
                n_trials=n_trials,
                proposal_temperature=proposal_temperature,
                candidate_ids=[int(x) for x in candidate_ids.detach().cpu().tolist()],
                candidate_unique_count=len(set(int(x) for x in candidate_ids.detach().cpu().tolist())),
                candidate_flatness=[float(x) for x in flatness_cpu.tolist()],
                candidate_current_logprobs=[float(x) for x in candidate_logprobs.tolist()],
                candidate_current_probs=[float(x) for x in candidate_probs.tolist()],
                candidate_current_ranks=[int(x) for x in candidate_ranks.tolist()],
                chosen_index=chosen_index,
                chosen_id=chosen_id,
                chosen_flatness=float(flatness_cpu[chosen_index].item()),
                chosen_current_logprob=float(candidate_logprobs[chosen_index].item()),
                chosen_current_prob=float(candidate_probs[chosen_index].item()),
                chosen_current_rank=int(candidate_ranks[chosen_index].item()),
                chosen_flatness_quantile=flatness_quantile_position(flatness_cpu, chosen_index),
                selector_target_quantile=selection.selector_target_quantile,
                allowed_candidate_indices=selection.allowed_candidate_indices,
                allowed_candidate_count=len(selection.allowed_candidate_indices),
                selection_fallback=selection.selection_fallback,
                mean_candidate_flatness=float(flatness_values.float().mean().detach().cpu().item()),
                min_candidate_flatness=float(flatness_values.float().min().detach().cpu().item()),
                max_candidate_flatness=float(flatness_values.float().max().detach().cpu().item()),
                selection_gain=float(gain.detach().cpu().item()),
                mean_candidate_current_logprob=float(candidate_logprobs.float().mean().item()),
                min_candidate_current_rank=int(candidate_ranks.min().item()),
                max_candidate_current_rank=int(candidate_ranks.max().item()),
            )
        )

        if tokenizer.eos_token_id is not None and chosen_id == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(input_ids[0].detach().cpu().tolist(), skip_special_tokens=True)
    return text, records
