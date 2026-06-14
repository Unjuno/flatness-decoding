from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--columns", nargs="+", default=None)
    parser.add_argument("--sort-by", nargs="+", default=None)
    parser.add_argument("--float-digits", type=int, default=4)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def format_cell(value, digits: int) -> str:
    if pd.isna(value):
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def to_markdown(df: pd.DataFrame, digits: int) -> str:
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(format_cell(row[col], digits) for col in cols) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    if args.columns:
        missing = [col for col in args.columns if col not in df.columns]
        if missing:
            raise SystemExit(f"missing columns: {missing}")
        df = df[args.columns]
    if args.sort_by:
        df = df.sort_values(args.sort_by, na_position="last")
    if args.max_rows is not None:
        df = df.head(args.max_rows)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(to_markdown(df, args.float_digits), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
