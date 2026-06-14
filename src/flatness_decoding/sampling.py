from __future__ import annotations

import torch


def sample_candidates_from_logits(
    logits: torch.Tensor,
    n: int,
    temperature: float = 1.0,
    replacement: bool = True,
    banned_token_ids: set[int] | None = None,
) -> torch.Tensor:
    """Sample `n` proposal tokens from a logit vector.

    This proposal distribution is only used to produce candidates. The final
    selection rule is handled by the decoder.
    """
    if logits.dim() != 1:
        raise ValueError(f"Expected 1D logits, got shape {tuple(logits.shape)}")
    if n <= 0:
        raise ValueError("n must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    scaled = logits.float().clone() / temperature

    if banned_token_ids:
        for token_id in banned_token_ids:
            if 0 <= token_id < scaled.numel():
                scaled[token_id] = -torch.inf

    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=n, replacement=replacement)
