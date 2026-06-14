from __future__ import annotations

import argparse
import random
from pathlib import Path

from datasets import load_dataset

from flatness_decoding.io_utils import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset name.")
    parser.add_argument("--config", default=None, help="Optional dataset config/subset name.")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--output", default="prompts/main_prompts.jsonl")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--min-words", type=int, default=40)
    parser.add_argument("--prompt-words", type=int, default=32)
    parser.add_argument("--prefix", default="main")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle eligible examples before selecting prompts.")
    parser.add_argument("--seed", type=int, default=0, help="Seed used when --shuffle is set.")
    return parser.parse_args()


def clean_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").split())


def make_prompt(text: str, prompt_words: int) -> str:
    words = clean_text(text).split()
    return " ".join(words[:prompt_words])


def eligible_texts(dataset, text_field: str, min_words: int) -> list[str]:
    texts: list[str] = []
    for item in dataset:
        raw = str(item.get(text_field, ""))
        text = clean_text(raw)
        if len(text.split()) >= min_words:
            texts.append(text)
    return texts


def build_prompt_rows(
    texts: list[str],
    *,
    limit: int,
    prompt_words: int,
    prefix: str,
    shuffle: bool = False,
    seed: int = 0,
) -> list[dict]:
    selected = list(texts)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(selected)
    selected = selected[:limit]
    return [
        {
            "id": f"{prefix}_{idx:04d}",
            "prompt": make_prompt(text, prompt_words),
        }
        for idx, text in enumerate(selected)
    ]


def main() -> None:
    args = parse_args()
    if args.config:
        dataset = load_dataset(args.dataset, args.config, split=args.split)
    else:
        dataset = load_dataset(args.dataset, split=args.split)

    texts = eligible_texts(dataset, args.text_field, args.min_words)
    rows = build_prompt_rows(
        texts,
        limit=args.limit,
        prompt_words=args.prompt_words,
        prefix=args.prefix,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    if len(rows) < args.limit:
        raise RuntimeError(
            f"Only collected {len(rows)} prompts. Lower --min-words or check dataset/text field."
        )

    output = Path(args.output)
    write_jsonl(output, rows)
    print(f"wrote {len(rows)} prompts to {output}")


if __name__ == "__main__":
    main()
