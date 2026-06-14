from __future__ import annotations

import torch


def flatness_std_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Compute standard deviation of raw logits.

    Args:
        logits: shape `[vocab]` or `[batch, vocab]`.

    Returns:
        Scalar tensor for 1D input, or `[batch]` tensor for 2D input.
    """
    return torch.std(logits.float(), dim=-1, unbiased=False)


def flatness_std_from_logprobs(logprobs: torch.Tensor) -> torch.Tensor:
    """Compute logit flatness from log probabilities.

    Since `log p_i = z_i - logsumexp(z)`, the standard deviation is invariant to
    the constant shift and equals `Std(z)`.
    """
    return torch.std(logprobs.float(), dim=-1, unbiased=False)


def selection_gain(candidate_flatness: torch.Tensor) -> torch.Tensor:
    """Relative gain from selecting the minimum-flatness candidate.

    Defined as `(mean(F_j) - min(F_j)) / mean(F_j)`.
    """
    values = candidate_flatness.float()
    mean_value = values.mean()
    if torch.isclose(mean_value, torch.tensor(0.0, device=values.device)):
        return torch.tensor(0.0, device=values.device)
    return (mean_value - values.min()) / mean_value
