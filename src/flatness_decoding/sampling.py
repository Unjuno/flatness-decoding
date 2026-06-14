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

    Sampling is intentionally performed on CPU. This avoids backend-specific
    multinomial issues on Apple MPS while keeping model forward passes on the
    selected accelerator.
    """
    if logits.dim() != 1:
        raise ValueError(f"Expected 1D logits, got shape {tuple(logits.shape)}")
    if n <= 0:
        raise ValueError("n must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    scaled = logits.detach().float().cpu().clone() / temperature

    if banned_token_ids:
        for token_id in banned_token_ids:
            if 0 <= token_id < scaled.numel():
                scaled[token_id] = -torch.inf

    probs = torch.softmax(scaled, dim=-1)
    if not torch.isfinite(probs).all() or float(probs.sum().item()) <= 0.0:
        raise ValueError("Invalid proposal probabilities; check logits and banned tokens.")
    return torch.multinomial(probs, num_samples=n, replacement=replacement)
