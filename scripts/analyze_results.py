from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


N_TRIAL_METHODS = {"random", "minflat", "maxflat"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", required=True, help="CSV from scripts/build_results_table.py")
    parser.add_argument("--output", default="results/summary/analysis_report.md")
    return parser.parse_args()


def fmt(value: object, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def available_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    cols = available_columns(df, columns)
    if not cols or df.empty:
        return "_No data._\n"
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in df[cols].iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in cols) + " |")
    return "\n".join(lines) + "\n"


def minflat_vs_random(df: pd.DataFrame) -> pd.DataFrame:
    needed = {"method", "n_trials", "mean_chosen_flatness"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()
    part = df[df["method"].isin(["minflat", "random"])].copy()
    part = part[part["n_trials"].notna()]
    if part.empty:
        return pd.DataFrame()
    pivot = part.pivot_table(
        index="n_trials",
        columns="method",
        values="mean_chosen_flatness",
        aggfunc="mean",
    ).reset_index()
    if not {"minflat", "random"}.issubset(pivot.columns):
        return pd.DataFrame()
    pivot["flatness_delta_minflat_minus_random"] = pivot["minflat"] - pivot["random"]
    pivot["relative_delta"] = pivot["flatness_delta_minflat_minus_random"] / pivot["random"]
    return pivot.sort_values("n_trials")


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "mean_chosen_flatness",
        "mean_selection_gain",
        "distinct_1",
        "distinct_2",
        "distinct_3",
        "repetition_4",
        "mean_runtime_seconds",
        "mean_tokens_per_second",
    ]
    cols = available_columns(df, metrics)
    numeric = df[cols].apply(pd.to_numeric, errors="coerce") if cols else pd.DataFrame()
    if numeric.empty or "mean_chosen_flatness" not in numeric.columns:
        return pd.DataFrame()
    rows = []
    for col in cols:
        if col == "mean_chosen_flatness":
            continue
        corr = numeric["mean_chosen_flatness"].corr(numeric[col])
        rows.append({"metric": col, "corr_with_mean_chosen_flatness": corr})
    return pd.DataFrame(rows)


def n_dependence(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "mean_chosen_flatness",
        "mean_selection_gain",
        "distinct_2",
        "repetition_4",
        "mean_runtime_seconds",
    ]
    if "method" not in df.columns or "n_trials" not in df.columns:
        return pd.DataFrame()
    part = df[df["method"].isin(N_TRIAL_METHODS)].copy()
    part = part[part["n_trials"].notna()]
    cols = ["method", "n_trials"] + available_columns(part, metrics)
    if len(cols) <= 2:
        return pd.DataFrame()
    return part[cols].sort_values(["method", "n_trials"])


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.table)

    lines = []
    lines.append("# Automatic analysis report\n")
    lines.append("## Input\n")
    lines.append(f"- table: `{args.table}`")
    lines.append(f"- rows: {len(df)}\n")

    lines.append("## Method summary\n")
    summary_cols = [
        "method",
        "method_family",
        "n_trials",
        "mean_chosen_flatness",
        "mean_selection_gain",
        "distinct_2",
        "repetition_4",
        "mean_runtime_seconds",
        "mean_tokens_per_second",
    ]
    lines.append(markdown_table(df.sort_values(["method", "n_trials"], na_position="first"), summary_cols))

    lines.append("## Flatness-N vs Random-N\n")
    diff = minflat_vs_random(df)
    lines.append(markdown_table(diff, ["n_trials", "random", "minflat", "flatness_delta_minflat_minus_random", "relative_delta"]))
    if not diff.empty:
        better = (diff["flatness_delta_minflat_minus_random"] < 0).sum()
        lines.append(
            f"Flatness-N has lower chosen flatness than Random-N for {better}/{len(diff)} N values.\n"
        )

    lines.append("## N dependence\n")
    lines.append(markdown_table(n_dependence(df), ["method", "n_trials", "mean_chosen_flatness", "mean_selection_gain", "distinct_2", "repetition_4", "mean_runtime_seconds"]))

    lines.append("## Correlation with chosen flatness\n")
    lines.append(markdown_table(correlations(df), ["metric", "corr_with_mean_chosen_flatness"]))

    lines.append("## Interpretation checklist\n")
    lines.append("- Does `minflat` reduce `mean_chosen_flatness` relative to `random` at the same N?")
    lines.append("- Does increasing N monotonically or approximately reduce flatness?")
    lines.append("- Does lower flatness improve, worsen, or trade off with distinct-n and repetition?")
    lines.append("- How quickly does runtime increase as N grows?")
    lines.append("- Does `maxflat` behave as a useful negative control?\n")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
