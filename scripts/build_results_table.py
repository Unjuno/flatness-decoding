from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

DISPLAY_KEY = ["method", "n_trials"]
METADATA_COLUMNS = ["experiment_name", "model", "method_family"]
GENERATION_METRIC_COLUMNS = [
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
TOKEN_METRIC_COLUMNS = [
    "num_steps",
    "mean_chosen_flatness",
    "mean_candidate_flatness",
    "mean_min_candidate_flatness",
    "mean_max_candidate_flatness",
    "mean_selection_gain",
    "mean_candidate_unique_count",
]
OUTPUT_COLUMNS = [
    *METADATA_COLUMNS,
    *DISPLAY_KEY,
    *GENERATION_METRIC_COLUMNS,
    *TOKEN_METRIC_COLUMNS,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, help="CSV from scripts/run_evaluate.py")
    parser.add_argument("--output", default="results/summary/comparison_table.csv")
    return parser.parse_args()


def normalize_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in [*METADATA_COLUMNS, "method", "seed"]:
        if col not in df.columns:
            df[col] = None
    if "n_trials" not in df.columns:
        df["n_trials"] = None
    df["n_trials"] = pd.to_numeric(df["n_trials"], errors="coerce")
    return df


def first_non_null(values: pd.Series) -> Any:
    for value in values:
        if pd.notna(value) and value != "":
            return value
    return None


def summarize_metric_block(df: pd.DataFrame, metric_columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[*DISPLAY_KEY, *METADATA_COLUMNS, *metric_columns])
    existing_metrics = [col for col in metric_columns if col in df.columns]
    cols = [*DISPLAY_KEY, *METADATA_COLUMNS, *existing_metrics]
    block = df[[col for col in cols if col in df.columns]].copy()
    for col in existing_metrics:
        block[col] = pd.to_numeric(block[col], errors="coerce")
    agg: dict[str, Any] = {col: first_non_null for col in METADATA_COLUMNS if col in block.columns}
    agg.update({col: "mean" for col in existing_metrics})
    return block.groupby(DISPLAY_KEY, dropna=False).agg(agg).reset_index()


def merge_metric_blocks(generation: pd.DataFrame, token_log: pd.DataFrame) -> pd.DataFrame:
    gen_summary = summarize_metric_block(generation, GENERATION_METRIC_COLUMNS)
    token_summary = summarize_metric_block(token_log, TOKEN_METRIC_COLUMNS)
    merged = pd.merge(
        gen_summary,
        token_summary,
        on=DISPLAY_KEY,
        how="outer",
        suffixes=("_generation", "_token"),
    )
    for col in METADATA_COLUMNS:
        gen_col = f"{col}_generation"
        token_col = f"{col}_token"
        if gen_col in merged.columns and token_col in merged.columns:
            merged[col] = merged[gen_col].combine_first(merged[token_col])
            merged = merged.drop(columns=[gen_col, token_col])
        elif gen_col in merged.columns:
            merged = merged.rename(columns={gen_col: col})
        elif token_col in merged.columns:
            merged = merged.rename(columns={token_col: col})
        elif col not in merged.columns:
            merged[col] = None
    existing_output_columns = [col for col in OUTPUT_COLUMNS if col in merged.columns]
    remaining_columns = [col for col in merged.columns if col not in existing_output_columns]
    table = merged[existing_output_columns + remaining_columns]
    return table.sort_values(DISPLAY_KEY, na_position="first").reset_index(drop=True)


def build_results_table(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_key_columns(df)
    generation = df[df["kind"] == "generation"].copy()
    token_log = df[df["kind"] == "token_log"].copy()
    return merge_metric_blocks(generation, token_log)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.metrics)
    table = build_results_table(df)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(f"wrote {output} with {len(table)} rows")


if __name__ == "__main__":
    main()
