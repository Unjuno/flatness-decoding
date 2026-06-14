from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

import pandas as pd

from flatness_decoding.io_utils import read_jsonl, write_csv
from flatness_decoding.metrics import analysis_text_from_row, distinct_n, repetition_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", default="results/summary/extended_metrics.csv")
    return parser.parse_args()


def tokenize(text: str) -> list[str]:
    return text.split()


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1))]


def entropy_n(texts: list[str], n: int) -> float:
    counts: Counter[tuple[str, ...]] = Counter()
    for text in texts:
        counts.update(ngrams(tokenize(text), n))
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log(count / total) for count in counts.values())


def type_token_ratio(texts: list[str]) -> float:
    tokens = [tok for text in texts for tok in tokenize(text)]
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def unique_output_ratio(texts: list[str]) -> float:
    if not texts:
        return 0.0
    return len(set(texts)) / len(texts)


def avg_sentence_like_length(texts: list[str]) -> float:
    # Lightweight approximation; avoids external NLP dependency.
    lengths: list[int] = []
    for text in texts:
        chunks = [chunk.strip() for chunk in text.replace("!", ".").replace("?", ".").split(".")]
        for chunk in chunks:
            if chunk:
                lengths.append(len(chunk.split()))
    if not lengths:
        return 0.0
    return sum(lengths) / len(lengths)


def metadata(rows: list[dict]) -> dict:
    first = rows[0] if rows else {}
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
        "seed",
    ]
    return {key: first.get(key) for key in keys}


def summarize_file(path: Path) -> dict:
    rows = read_jsonl(path)
    texts = [analysis_text_from_row(row) for row in rows]
    word_lengths = [len(tokenize(text)) for text in texts]
    return {
        "file": str(path),
        **metadata(rows),
        "num_outputs": len(texts),
        "mean_continuation_words": sum(word_lengths) / len(word_lengths) if word_lengths else 0.0,
        "median_continuation_words": float(pd.Series(word_lengths).median()) if word_lengths else 0.0,
        "type_token_ratio": type_token_ratio(texts),
        "unique_output_ratio": unique_output_ratio(texts),
        "distinct_1": distinct_n(texts, 1),
        "distinct_2": distinct_n(texts, 2),
        "distinct_3": distinct_n(texts, 3),
        "entropy_1": entropy_n(texts, 1),
        "entropy_2": entropy_n(texts, 2),
        "entropy_3": entropy_n(texts, 3),
        "repetition_2": repetition_rate(texts, 2),
        "repetition_3": repetition_rate(texts, 3),
        "repetition_4": repetition_rate(texts, 4),
        "avg_sentence_like_words": avg_sentence_like_length(texts),
    }


def main() -> None:
    args = parse_args()
    rows = [summarize_file(path) for path in sorted(Path(args.input_dir).glob("*.jsonl"))]
    write_csv(args.output, rows)
    print(f"wrote {args.output} with {len(rows)} rows")


if __name__ == "__main__":
    main()
