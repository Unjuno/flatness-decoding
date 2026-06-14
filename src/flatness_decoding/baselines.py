from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch

BaselineMethod = Literal["greedy", "sampling", "temperature", "top_p", "top_k"]


@dataclass
class BaselineGeneration:
    text: str
    num_input_tokens: int
    num_output_tokens: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@torch.no_grad()
def generate_baseline(
    model,
    tokenizer,
    prompt: str,
    method: BaselineMethod,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50,
    device: torch.device | None = None,
    seed: int | None = None,
) -> BaselineGeneration:
    """Generate text with standard Hugging Face decoding baselines.

    These baselines use the same prompt set as Flatness-N but do not use N-trial
    lookahead. They are included for thesis-level comparison, not for exhaustive
    reproduction of every decoding paper baseline.
    """
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if device is None:
        device = next(model.parameters()).device

    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id

    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_token_id,
    }

    if method == "greedy":
        generation_kwargs.update({"do_sample": False})
    elif method == "sampling":
        generation_kwargs.update({"do_sample": True, "temperature": 1.0})
    elif method == "temperature":
        generation_kwargs.update({"do_sample": True, "temperature": temperature})
    elif method == "top_p":
        generation_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
    elif method == "top_k":
        generation_kwargs.update({"do_sample": True, "temperature": temperature, "top_k": top_k})
    else:
        raise ValueError(f"Unknown baseline method: {method}")

    # Remove None fields because some model.generate versions are strict.
    generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
    output_ids = model.generate(**generation_kwargs)

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return BaselineGeneration(
        text=text,
        num_input_tokens=int(input_ids.shape[1]),
        num_output_tokens=int(output_ids.shape[1] - input_ids.shape[1]),
    )
