from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from datasets import load_dataset

from flatness_decoding.io_utils import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset name.")
    parser.add_argument("--config", default=None, help="Optional dataset config/subset name.")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--text-field", default=None, help="Dataset column for legacy long-text prefix extraction.")
    parser.add_argument("--prompt-field", default=None, help="Dataset column to use directly as the prompt.")
    parser.add_argument("--reference-field", default=None, help="Optional dataset column to save as reference/story text.")
    parser.add_argument("--output", default="prompts/writingprompts_fixed_100.jsonl")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--min-prompt-words", type=int, default=1)
    parser.add_argument("--min-words", type=int, default=40)
    parser.add_argument("--prompt-words", type=int, default=32)
    parser.add_argument("--prefix", default="wp")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle eligible examples before selecting prompts.")
    parser.add_argument("--seed", type=int, default=0, help="Seed used when --shuffle is set.")
    return parser.parse_args()


def clean_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").split())


def make_prompt(text: str, prompt_words: int) -> str:
    words = clean_text(text).split()
    return " ".join(words[:prompt_words])


def row_text(item: dict[str, Any], field: str) -> str:
    if field not in item:
        raise KeyError(f"Dataset row is missing requested field: {field}")
    return clean_text(str(item.get(field, "")))


def eligible_rows(
    dataset,
    *,
    dataset_name: str,
    source_config: str | None,
    split: str,
    prompt_field: str | None,
    reference_field: str | None,
    text_field: str | None,
    min_prompt_words: int,
    min_words: int,
    prompt_words: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    legacy_text_field = text_field or (None if prompt_field else "text")
    for source_row_index, item in enumerate(dataset):
        if prompt_field:
            prompt = row_text(item, prompt_field)
            if len(prompt.split()) < min_prompt_words:
                continue
        else:
            if legacy_text_field is None:
                raise ValueError("Specify --prompt-field or --text-field.")
            text = row_text(item, legacy_text_field)
            if len(text.split()) < min_words:
                continue
            prompt = make_prompt(text, prompt_words)
            if len(prompt.split()) < min_prompt_words:
                continue

        row: dict[str, Any] = {
            "prompt": prompt,
            "source_dataset": dataset_name,
            "source_config": source_config,
            "source_split": split,
            "source_row_index": source_row_index,
            "text_field": legacy_text_field,
        }
        if prompt_field is not None:
            row["prompt_field"] = prompt_field
        if reference_field is not None:
            row["reference_field"] = reference_field
            reference = row_text(item, reference_field)
            if reference:
                row["reference"] = reference
        rows.append(row)
    return rows


def build_prompt_rows(
    candidates: list[dict[str, Any]],
    *,
    limit: int,
    prefix: str,
    shuffle: bool = False,
    seed: int = 0,
) -> list[dict[str, Any]]:
    selected = list(candidates)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(selected)
    selected = selected[:limit]
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(selected):
        rows.append({"id": f"{prefix}_{idx:06d}", **row})
    return rows


def main() -> None:
    args = parse_args()
    if args.config:
        dataset = load_dataset(args.dataset, args.config, split=args.split)
    else:
        dataset = load_dataset(args.dataset, split=args.split)

    candidates = eligible_rows(
        dataset,
        dataset_name=args.dataset,
        source_config=args.config,
        split=args.split,
        prompt_field=args.prompt_field,
        reference_field=args.reference_field,
        text_field=args.text_field,
        min_prompt_words=args.min_prompt_words,
        min_words=args.min_words,
        prompt_words=args.prompt_words,
    )
    rows = build_prompt_rows(
        candidates,
        limit=args.limit,
        prefix=args.prefix,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    if len(rows) < args.limit:
        raise RuntimeError(
            f"Only collected {len(rows)} prompts. Lower min-word thresholds or check dataset fields."
        )

    output = Path(args.output)
    write_jsonl(output, rows)
    print(f"wrote {len(rows)} prompts to {output}")


if __name__ == "__main__":
    main()
