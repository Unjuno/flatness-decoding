from __future__ import annotations

from collections import Counter
from pathlib import Path

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


def mean_text_length(texts: list[str]) -> float:
    if not texts:
        return 0.0
    return sum(len(text.split()) for text in texts) / len(texts)


def summarize_generation_file(path: str | Path) -> dict[str, float | int | str]:
    rows = read_jsonl(path)
    texts = [str(row.get("text", "")) for row in rows]
    return {
        "file": str(path),
        "num_outputs": len(texts),
        "mean_words": mean_text_length(texts),
        "distinct_1": distinct_n(texts, 1),
        "distinct_2": distinct_n(texts, 2),
        "distinct_3": distinct_n(texts, 3),
        "repetition_4": repetition_rate(texts, 4),
    }


def summarize_token_log_file(path: str | Path) -> dict[str, float | int | str]:
    rows = read_jsonl(path)
    if not rows:
        return {
            "file": str(path),
            "num_steps": 0,
            "mean_chosen_flatness": 0.0,
            "mean_selection_gain": 0.0,
        }

    chosen = [float(row.get("chosen_flatness", 0.0)) for row in rows]
    gains = [float(row.get("selection_gain", 0.0)) for row in rows]
    return {
        "file": str(path),
        "num_steps": len(rows),
        "mean_chosen_flatness": sum(chosen) / len(chosen),
        "mean_selection_gain": sum(gains) / len(gains),
    }
