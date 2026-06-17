from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .io_utils import read_jsonl


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if n <= 0:
        raise ValueError("n must be positive")
    return [tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1))]


def distinct_n(texts: list[str], n: int = 2) -> float:
    all_ngrams: list[tuple[str, ...]] = []
    for text in texts:
        all_ngrams.extend(ngrams(text.split(), n))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def repetition_rate(texts: list[str], n: int = 4) -> float:
    """Fraction of n-grams that repeat after first occurrence."""
    all_ngrams: list[tuple[str, ...]] = []
    for text in texts:
        all_ngrams.extend(ngrams(text.split(), n))
    if not all_ngrams:
        return 0.0
    counts = Counter(all_ngrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(all_ngrams)


def continuation_from_text(text: str, prompt: str) -> str:
    """Remove a prompt prefix from a decoded full text when possible."""
    if not prompt:
        return text
    if text.startswith(prompt):
        return text[len(prompt) :].lstrip()

    # Decoding can sometimes introduce leading whitespace. Handle that without
    # normalizing the internal content of the generated text.
    stripped_text = text.lstrip()
    stripped_prompt = prompt.lstrip()
    if stripped_prompt and stripped_text.startswith(stripped_prompt):
        return stripped_text[len(stripped_prompt) :].lstrip()

    return text


def analysis_text_from_row(row: dict[str, Any]) -> str:
    """Return the text span used for output-quality metrics.

    Generation scripts now write `continuation_text`. For older outputs, fall
    back to removing the prompt prefix from `text`. This avoids measuring prompt
    words in distinct/repetition/PPL style metrics.
    """
    continuation = row.get("continuation_text")
    if continuation is not None:
        return str(continuation)
    text = str(row.get("text", ""))
    prompt = str(row.get("prompt", ""))
    return continuation_from_text(text, prompt)


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def mean_text_length(texts: list[str]) -> float:
    if not texts:
        return 0.0
    return sum(len(text.split()) for text in texts) / len(texts)


def first_metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    first = rows[0]
    keys = [
        "experiment_name",
        "model",
        "method",
        "method_family",
        "n_trials",
        "proposal_temperature",
        "baseline_temperature",
        "top_p",
        "top_k",
        "num_beams",
        "highprob_logprob_margin",
        "highprob_rank_limit",
        "seed",
    ]
    return {key: first.get(key) for key in keys if key in first}


def summarize_generation_file(path: str | Path) -> dict[str, float | int | str | None]:
    rows = read_jsonl(path)
    texts = [analysis_text_from_row(row) for row in rows]
    output_tokens = [float(row["num_output_tokens"]) for row in rows if row.get("num_output_tokens") is not None]
    runtimes = [float(row["runtime_seconds"]) for row in rows if row.get("runtime_seconds") is not None]
    tps = [float(row["tokens_per_second"]) for row in rows if row.get("tokens_per_second") is not None]

    return {
        "file": str(path),
        **first_metadata(rows),
        "num_outputs": len(texts),
        "mean_continuation_words": mean_text_length(texts),
        "mean_output_tokens": mean_or_none(output_tokens),
        "mean_runtime_seconds": mean_or_none(runtimes),
        "mean_tokens_per_second": mean_or_none(tps),
        "distinct_1": distinct_n(texts, 1),
        "distinct_2": distinct_n(texts, 2),
        "distinct_3": distinct_n(texts, 3),
        "repetition_4": repetition_rate(texts, 4),
    }


def value_or_alias(row: dict[str, Any], key: str, alias: str | None = None, default: float = 0.0) -> float:
    value = row.get(key)
    if value is None and alias is not None:
        value = row.get(alias)
    if value is None:
        return default
    return float(value)


def summarize_token_log_file(path: str | Path) -> dict[str, float | int | str | None]:
    rows = read_jsonl(path)
    if not rows:
        return {
            "file": str(path),
            "num_steps": 0,
            "mean_chosen_flatness": 0.0,
            "mean_candidate_flatness": 0.0,
            "mean_min_candidate_flatness": 0.0,
            "mean_max_candidate_flatness": 0.0,
            "mean_selection_gain": 0.0,
            "mean_candidate_unique_count": 0.0,
            "mean_chosen_current_logprob": 0.0,
            "mean_chosen_current_prob": 0.0,
            "mean_chosen_current_rank": 0.0,
            "mean_chosen_flatness_quantile": 0.0,
            "mean_allowed_candidate_count": 0.0,
            "selection_fallback_rate": 0.0,
            "mean_candidate_current_logprob": 0.0,
            "mean_min_candidate_current_rank": 0.0,
            "mean_max_candidate_current_rank": 0.0,
        }

    chosen = [float(row.get("chosen_flatness", 0.0)) for row in rows]
    mean_candidate = [float(row.get("mean_candidate_flatness", 0.0)) for row in rows]
    min_candidate = [value_or_alias(row, "min_candidate_flatness", "mean_min_candidate_flatness") for row in rows]
    max_candidate = [value_or_alias(row, "max_candidate_flatness", "mean_max_candidate_flatness") for row in rows]
    gains = [float(row.get("selection_gain", 0.0)) for row in rows]
    unique_counts = [
        float(row.get("candidate_unique_count", len(set(row.get("candidate_ids", [])))))
        for row in rows
    ]
    chosen_current_logprob = [float(row.get("chosen_current_logprob", 0.0)) for row in rows]
    chosen_current_prob = [float(row.get("chosen_current_prob", 0.0)) for row in rows]
    chosen_current_rank = [float(row.get("chosen_current_rank", 0.0)) for row in rows]
    chosen_flatness_quantile = [float(row.get("chosen_flatness_quantile", 0.0)) for row in rows]
    allowed_candidate_count = [float(row.get("allowed_candidate_count", 0.0)) for row in rows]
    selection_fallback = [1.0 if bool(row.get("selection_fallback", False)) else 0.0 for row in rows]
    candidate_current_logprob = [float(row.get("mean_candidate_current_logprob", 0.0)) for row in rows]
    min_candidate_current_rank = [float(row.get("min_candidate_current_rank", 0.0)) for row in rows]
    max_candidate_current_rank = [float(row.get("max_candidate_current_rank", 0.0)) for row in rows]

    return {
        "file": str(path),
        **first_metadata(rows),
        "num_steps": len(rows),
        "mean_chosen_flatness": mean_or_none(chosen),
        "mean_candidate_flatness": mean_or_none(mean_candidate),
        "mean_min_candidate_flatness": mean_or_none(min_candidate),
        "mean_max_candidate_flatness": mean_or_none(max_candidate),
        "mean_selection_gain": mean_or_none(gains),
        "mean_candidate_unique_count": mean_or_none(unique_counts),
        "mean_chosen_current_logprob": mean_or_none(chosen_current_logprob),
        "mean_chosen_current_prob": mean_or_none(chosen_current_prob),
        "mean_chosen_current_rank": mean_or_none(chosen_current_rank),
        "mean_chosen_flatness_quantile": mean_or_none(chosen_flatness_quantile),
        "mean_allowed_candidate_count": mean_or_none(allowed_candidate_count),
        "selection_fallback_rate": mean_or_none(selection_fallback),
        "mean_candidate_current_logprob": mean_or_none(candidate_current_logprob),
        "mean_min_candidate_current_rank": mean_or_none(min_candidate_current_rank),
        "mean_max_candidate_current_rank": mean_or_none(max_candidate_current_rank),
    }
