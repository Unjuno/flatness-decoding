from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Any

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


def expand_batch_tensor(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    """View a batch-1 cache tensor as `batch_size` without copying when possible."""
    if tensor.dim() == 0 or tensor.shape[0] != 1 or batch_size == 1:
        return tensor
    return tensor.expand(batch_size, *tensor.shape[1:])


def expand_past_key_values(past_key_values: Any, batch_size: int) -> Any:
    """Expand legacy tuple/list KV caches from batch size 1 to `batch_size`.

    GPT-2-like models in Transformers return a nested tuple of key/value tensors
    with batch dimension first. This helper preserves non-tensor objects and
    avoids materializing repeated prefix caches when possible.
    """
    if past_key_values is None or batch_size == 1:
        return past_key_values
    if torch.is_tensor(past_key_values):
        return expand_batch_tensor(past_key_values, batch_size)
    if isinstance(past_key_values, tuple):
        return tuple(expand_past_key_values(item, batch_size) for item in past_key_values)
    if isinstance(past_key_values, list):
        return [expand_past_key_values(item, batch_size) for item in past_key_values]
    if hasattr(past_key_values, "batch_repeat_interleave"):
        # Newer Transformers cache classes expose this method. It may copy, but
        # is the safest generic fallback for non-legacy cache objects.
        return past_key_values.batch_repeat_interleave(batch_size)
    return past_key_values


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
        expanded_past = expand_past_key_values(past_key_values, n)
        outputs = model(
            input_ids=candidate_ids.to(device).view(n, 1),
            past_key_values=expanded_past,
            use_cache=False,
        )
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

    The default evaluates all candidates in one batch. Passing
    `candidate_eval_batch_size` micro-batches the lookahead forward passes to
    lower peak memory, which is important on Apple MPS and for large N.

    When `past_key_values` is provided, candidates are evaluated as one-step
    continuations of the cached prefix instead of re-forwarding the full prefix.
    """
    if candidate_eval_batch_size is None or candidate_eval_batch_size <= 0:
        candidate_eval_batch_size = int(candidate_ids.shape[0])

    chunks: list[torch.Tensor] = []
    for start in range(0, int(candidate_ids.shape[0]), candidate_eval_batch_size):
        end = start + candidate_eval_batch_size
        chunk = candidate_ids[start:end]
        chunks.append(
            evaluate_candidate_batch_next_flatness(
                model,
                input_ids,
                chunk,
                past_key_values=past_key_values,
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
        # With one candidate there is no choice to make. Avoid consuming RNG so
        # Random-1 remains an exact control for MinFlat-1/MaxFlat-1 under the
        # same proposal seed. Use CPU RNG to avoid backend-specific MPS issues.
        idx = 0 if candidate_cpu.shape[0] == 1 else int(torch.randint(0, candidate_cpu.shape[0], (1,)).item())
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return idx, int(candidate_cpu[idx].item())


def banned_candidate_tokens(tokenizer) -> set[int]:
    """Return special token IDs to ban from proposal candidates.

    GPT-2-like models often have no native pad token, so `model_io` sets
    `pad_token=eos_token`. In that case, banning `pad_token_id` would also ban
    EOS and prevent Flatness-N from stopping early. Therefore, ban pad only when
    it is distinct from EOS.
    """
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
) -> tuple[str, list[StepRecord]]:
    """Generate text with Random-N, MinFlat-N, or MaxFlat-N decoding.

    `method="minflat"` is Flatness-N.
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

    prefix_outputs = model(input_ids=input_ids, use_cache=True)
    logits_now = prefix_outputs.logits[0, -1, :]
    past_key_values = getattr(prefix_outputs, "past_key_values", None)

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
            past_key_values=past_key_values,
        )

        chosen_index, chosen_id = choose_candidate(
            candidate_ids=candidate_ids,
            flatness_values=flatness_values,
            mode=method,
        )

        chosen_tensor = torch.tensor([[chosen_id]], device=device)
        if past_key_values is not None:
            advance_outputs = model(
                input_ids=chosen_tensor,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits_now = advance_outputs.logits[0, -1, :]
            past_key_values = getattr(advance_outputs, "past_key_values", None)
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
