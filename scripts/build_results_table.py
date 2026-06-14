from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, help="CSV from scripts/run_evaluate.py")
    parser.add_argument("--output", default="results/summary/comparison_table.csv")
    return parser.parse_args()


def normalize_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["experiment_name", "model", "method", "method_family", "seed"]:
        if col not in df.columns:
            df[col] = None
    if "n_trials" not in df.columns:
        df["n_trials"] = None
    return df


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.metrics)
    df = normalize_key_columns(df)

    gen = df[df["kind"] == "generation"].copy()
    token = df[df["kind"] == "token_log"].copy()

    key = ["experiment_name", "model", "method", "method_family", "n_trials", "seed"]

    gen_cols = [
        *key,
        "num_outputs",
        "mean_continuation_words",
        "mean_output_tokens",
        "mean_runtime_seconds",
        "mean_tokens_per_second",
        "distinct_1",
        "distinct_2",
        "distinct_3",
        "repetition_4",
    ]
    token_cols = [
        *key,
        "num_steps",
        "mean_chosen_flatness",
        "mean_candidate_flatness",
        "mean_min_candidate_flatness",
        "mean_max_candidate_flatness",
        "mean_selection_gain",
        "mean_candidate_unique_count",
    ]

    gen = gen[[col for col in gen_cols if col in gen.columns]]
    token = token[[col for col in token_cols if col in token.columns]]

    if token.empty:
        merged = gen
    else:
        merged = pd.merge(gen, token, on=key, how="outer")

    # A second aggregation over seeds gives the thesis-level compact table.
    group_key = ["experiment_name", "model", "method", "method_family", "n_trials"]
    numeric_cols = [col for col in merged.columns if col not in group_key and col != "seed"]
    numeric_cols = [col for col in numeric_cols if pd.api.types.is_numeric_dtype(merged[col])]
    table = merged.groupby(group_key, dropna=False)[numeric_cols].mean().reset_index()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(f"wrote {output} with {len(table)} rows")


if __name__ == "__main__":
    main()
