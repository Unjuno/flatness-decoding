from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


GROUP_KEY = ["experiment_name", "model", "method", "method_family", "n_trials"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Base compact table CSV.")
    parser.add_argument("--extras", nargs="*", default=[], help="Additional metric CSV files.")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in GROUP_KEY:
        if col not in df.columns:
            df[col] = None
    return df


def aggregate_extra(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = normalize_keys(df)
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    numeric_cols = [col for col in numeric_cols if col not in {"seed"}]
    keep = GROUP_KEY + numeric_cols
    df = df[[col for col in keep if col in df.columns]]
    return df.groupby(GROUP_KEY, dropna=False).mean(numeric_only=True).reset_index()


def add_suffix_to_extra_columns(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    rename = {col: f"{col}_{suffix}" for col in df.columns if col not in GROUP_KEY}
    return df.rename(columns=rename)


def main() -> None:
    args = parse_args()
    merged = normalize_keys(pd.read_csv(args.base))

    for extra_path in args.extras:
        suffix = Path(extra_path).stem
        extra = aggregate_extra(extra_path)
        extra = add_suffix_to_extra_columns(extra, suffix)
        merged = pd.merge(merged, extra, on=GROUP_KEY, how="left")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    print(f"wrote {output} with {len(merged)} rows")


if __name__ == "__main__":
    main()
