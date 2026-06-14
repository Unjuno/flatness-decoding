from __future__ import annotations

import argparse
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
    return parser.parse_args()


def clean_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").split())


def make_prompt(text: str, prompt_words: int) -> str:
    words = clean_text(text).split()
    return " ".join(words[:prompt_words])


def main() -> None:
    args = parse_args()
    if args.config:
        dataset = load_dataset(args.dataset, args.config, split=args.split)
    else:
        dataset = load_dataset(args.dataset, split=args.split)

    rows: list[dict] = []
    for item in dataset:
        raw = str(item.get(args.text_field, ""))
        text = clean_text(raw)
        if len(text.split()) < args.min_words:
            continue
        prompt = make_prompt(text, args.prompt_words)
        rows.append({"id": f"{args.prefix}_{len(rows):04d}", "prompt": prompt})
        if len(rows) >= args.limit:
            break

    if len(rows) < args.limit:
        raise RuntimeError(
            f"Only collected {len(rows)} prompts. Lower --min-words or check dataset/text field."
        )

    output = Path(args.output)
    write_jsonl(output, rows)
    print(f"wrote {len(rows)} prompts to {output}")


if __name__ == "__main__":
    main()
