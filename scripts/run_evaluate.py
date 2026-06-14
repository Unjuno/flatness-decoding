from __future__ import annotations

import argparse
from pathlib import Path

from flatness_decoding.io_utils import write_csv
from flatness_decoding.metrics import summarize_generation_file, summarize_token_log_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Directory containing generation .jsonl files.")
    parser.add_argument("--token-log-dir", default=None, help="Optional directory containing .tokenlog.jsonl files.")
    parser.add_argument("--output", default="results/summary/metrics.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    rows: list[dict] = []

    for path in sorted(input_dir.glob("*.jsonl")):
        rows.append({"kind": "generation", **summarize_generation_file(path)})

    if args.token_log_dir:
        token_dir = Path(args.token_log_dir)
        for path in sorted(token_dir.glob("*.tokenlog.jsonl")):
            rows.append({"kind": "token_log", **summarize_token_log_file(path)})

    write_csv(args.output, rows)
    print(f"wrote {args.output} with {len(rows)} rows")


if __name__ == "__main__":
    main()
