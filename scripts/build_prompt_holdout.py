from __future__ import annotations

import argparse
from pathlib import Path

from flatness_decoding.io_utils import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic holdout from a tracked prompt JSONL.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--prefix", default="holdout")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    selected = rows[args.start : args.start + args.limit]
    if len(selected) != args.limit:
        raise RuntimeError(f"requested {args.limit} rows from offset {args.start}, found {len(selected)}")
    output_rows = []
    for index, row in enumerate(selected):
        copied = dict(row)
        copied["original_prompt_id"] = row.get("id")
        copied["id"] = f"{args.prefix}_{index:06d}"
        copied["holdout_source_offset"] = args.start + index
        output_rows.append(copied)
    write_jsonl(Path(args.output), output_rows)
    print(f"wrote {len(output_rows)} prompts to {args.output}")


if __name__ == "__main__":
    main()
