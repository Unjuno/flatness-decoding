from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch

from .flatness import flatness_std_from_logits, selection_gain
from .sampling import sample_candidates_from_logits

DecodeMode = Literal["minflat", "maxflat", "random"]


@dataclass
class StepRecord:
    step: int
    method: str
    n_trials: int
    proposal_temperature: float
    candidate_ids: list[int]
    candidate_unique_count: int
    candidate_flatness: list[float]
    chosen_id: int
    chosen_flatness: float
    mean_candidate_flatness: float
    min_candidate_flatness: float
    max_candidate_flatness: float
    selection_gain: float

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


def choose_candidate(
    candidate_ids: torch.Tensor,
    flatness_values: torch.Tensor,
    mode: DecodeMode,
) -> tuple[int, int]:
    if candidate_ids.shape[0] < 1:
        raise ValueError("candidate_ids must contain at least one candidate")
    flatness_cpu = flatness_values.detach().cpu()
    candidate_cpu = candidate_ids.detach().cpu()
    if mode == "minflat":
        idx = int(torch.argmin(flatness_cpu).item())
    elif mode == "maxflat":
        idx = int(torch.argmax(flatness_cpu).item())
    elif mode == "random":
        idx = 0 if candidate_cpu.shape[0] == 1 else int(torch.randint(0, candidate_cpu.shape[0], (1,)).item())
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return idx, int(candidate_cpu[idx].item())


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
) -> tuple[str, list[StepRecord]]:
    """Generate text with Random-N, MinFlat-N, or MaxFlat-N decoding.

    `method="minflat"` is Flatness-N. Official runs default to exact full-prefix
    candidate evaluation with micro-batching. Cached candidate lookahead remains
    optional and disabled by default because it did not pass equivalence tests in
    the current GPT-2/Transformers stack.
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

        flatness_values = evaluate_candidates_next_flatness(
            model=model,
            input_ids=input_ids,
            candidate_ids=candidate_ids,
            candidate_eval_batch_size=candidate_eval_batch_size,
            past_key_values=past_key_values if use_kv_cache_lookahead else None,
        )

        chosen_index, chosen_id = choose_candidate(
            candidate_ids=candidate_ids,
            flatness_values=flatness_values,
            mode=method,
        )

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
        records.append(
            StepRecord(
                step=step,
                method=method,
                n_trials=n_trials,
                proposal_temperature=proposal_temperature,
                candidate_ids=[int(x) for x in candidate_ids.detach().cpu().tolist()],
                candidate_unique_count=len(set(int(x) for x in candidate_ids.detach().cpu().tolist())),
                candidate_flatness=[float(x) for x in flatness_values.detach().cpu().tolist()],
                chosen_id=chosen_id,
                chosen_flatness=float(flatness_values[chosen_index].detach().cpu().item()),
                mean_candidate_flatness=float(flatness_values.float().mean().detach().cpu().item()),
                min_candidate_flatness=float(flatness_values.float().min().detach().cpu().item()),
                max_candidate_flatness=float(flatness_values.float().max().detach().cpu().item()),
                selection_gain=float(gain.detach().cpu().item()),
            )
        )

        if tokenizer.eos_token_id is not None and chosen_id == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(input_ids[0].detach().cpu().tolist(), skip_special_tokens=True)
    return text, records
