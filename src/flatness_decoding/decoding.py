from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

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


@torch.no_grad()
def evaluate_candidates_next_flatness(
    model,
    input_ids: torch.Tensor,
    candidate_ids: torch.Tensor,
) -> torch.Tensor:
    """Evaluate next-state logit flatness for each candidate.

    The candidates are evaluated in one batch for speed.
    """
    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids must have shape [1, seq_len]")
    if candidate_ids.dim() != 1:
        raise ValueError("candidate_ids must have shape [n]")

    device = input_ids.device
    n = candidate_ids.shape[0]
    repeated = input_ids.repeat(n, 1)
    appended = torch.cat([repeated, candidate_ids.to(device).view(n, 1)], dim=1)

    outputs = model(input_ids=appended)
    logits_next = outputs.logits[:, -1, :]
    return flatness_std_from_logits(logits_next)


def choose_candidate(
    candidate_ids: torch.Tensor,
    flatness_values: torch.Tensor,
    mode: DecodeMode,
) -> tuple[int, int]:
    if mode == "minflat":
        idx = int(torch.argmin(flatness_values).item())
    elif mode == "maxflat":
        idx = int(torch.argmax(flatness_values).item())
    elif mode == "random":
        idx = int(torch.randint(0, candidate_ids.shape[0], (1,), device=candidate_ids.device).item())
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return idx, int(candidate_ids[idx].item())


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

    banned = {tokenizer.pad_token_id} if tokenizer.pad_token_id is not None else set()

    for step in range(max_new_tokens):
        logits_now = next_logits(model, input_ids)
        candidate_ids = sample_candidates_from_logits(
            logits=logits_now,
            n=n_trials,
            temperature=proposal_temperature,
            replacement=True,
            banned_token_ids=banned,
        ).to(device)

        flatness_values = evaluate_candidates_next_flatness(
            model=model,
            input_ids=input_ids,
            candidate_ids=candidate_ids,
        )

        chosen_index, chosen_id = choose_candidate(
            candidate_ids=candidate_ids,
            flatness_values=flatness_values,
            mode=method,
        )

        input_ids = torch.cat([input_ids, torch.tensor([[chosen_id]], device=device)], dim=1)

        gain = selection_gain(flatness_values)
        records.append(
            StepRecord(
                step=step,
                method=method,
                n_trials=n_trials,
                proposal_temperature=proposal_temperature,
                candidate_ids=[int(x) for x in candidate_ids.detach().cpu().tolist()],
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

    text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return text, records
