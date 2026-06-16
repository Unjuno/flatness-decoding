from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

DISPLAY_KEY = ["method", "n_trials"]
METADATA_COLUMNS = ["experiment_name", "model", "method_family"]
BASE_NUMERIC_COLUMNS = [
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


def summarize_metric_block(
    df: pd.DataFrame,
    metric_columns: list[str],
    *,
    keep_columns: list[str] | None = None,
) -> pd.DataFrame:
    if df.empty:
        columns = [*DISPLAY_KEY, *METADATA_COLUMNS]
        if keep_columns:
            columns.extend(keep_columns)
        return pd.DataFrame(columns=columns)
    existing_metrics = [col for col in metric_columns if col in df.columns]
    cols = [*DISPLAY_KEY, *METADATA_COLUMNS, *existing_metrics]
    if keep_columns:
        cols.extend([col for col in keep_columns if col in df.columns])
    block = df[[col for col in cols if col in df.columns]].copy()
    numeric_cols = existing_metrics + [col for col in (keep_columns or []) if col in block.columns]
    for col in numeric_cols:
        block[col] = pd.to_numeric(block[col], errors="coerce")
    agg: dict[str, Any] = {col: first_non_null for col in METADATA_COLUMNS if col in block.columns}
    agg.update({col: "mean" for col in numeric_cols})
    return block.groupby(DISPLAY_KEY, dropna=False).agg(agg).reset_index()


def summarize_compact_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[*METADATA_COLUMNS, *DISPLAY_KEY, *OUTPUT_COLUMNS])

    keep_columns = [*METADATA_COLUMNS, *DISPLAY_KEY]
    block = df.copy()
    for col in [*METADATA_COLUMNS, "method", "n_trials"]:
        if col not in block.columns:
            block[col] = None

    block["n_trials"] = pd.to_numeric(block["n_trials"], errors="coerce")

    numeric_columns = [
        col
        for col in block.columns
        if col not in {*keep_columns, "kind", "file", "seed"} and pd.api.types.is_numeric_dtype(block[col])
    ]
    for col in numeric_columns:
        block[col] = pd.to_numeric(block[col], errors="coerce")

    agg: dict[str, Any] = {col: first_non_null for col in METADATA_COLUMNS}
    agg.update({col: "mean" for col in numeric_columns})
    summary = block.groupby(DISPLAY_KEY, dropna=False).agg(agg).reset_index()

    for col in METADATA_COLUMNS:
        if col not in summary.columns:
            summary[col] = None

    ordered_columns = [
        *METADATA_COLUMNS,
        *DISPLAY_KEY,
        *[
            col
            for col in [
                *GENERATION_METRIC_COLUMNS,
                *TOKEN_METRIC_COLUMNS,
            ]
            if col in summary.columns
        ],
    ]
    extra_columns = [
        col
        for col in summary.columns
        if col not in {*METADATA_COLUMNS, *DISPLAY_KEY, *GENERATION_METRIC_COLUMNS, *TOKEN_METRIC_COLUMNS}
    ]
    return summary[ordered_columns + extra_columns].sort_values(DISPLAY_KEY, na_position="first").reset_index(
        drop=True
    )


def build_results_table(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_key_columns(df)
    return summarize_compact_table(df)


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
