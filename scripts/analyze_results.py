from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


N_TRIAL_METHODS = {"random", "minflat", "maxflat"}
FLATNESS_METHOD_ORDER = ["random", "minflat", "maxflat"]


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


def sorted_table(df: pd.DataFrame) -> pd.DataFrame:
    if {"method", "n_trials"}.issubset(df.columns):
        return df.sort_values(["method", "n_trials"], na_position="first")
    return df


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


def method_n_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    if not {"method", "n_trials"}.issubset(df.columns):
        return pd.DataFrame()
    counts = df.groupby(["method", "n_trials"], dropna=False).size().reset_index(name="rows")
    return counts[counts["rows"] > 1].sort_values(["method", "n_trials"], na_position="first")


def flatness_control_table(df: pd.DataFrame) -> pd.DataFrame:
    needed = {"method", "n_trials", "mean_chosen_flatness"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()
    part = df[df["method"].isin(FLATNESS_METHOD_ORDER)].copy()
    part = part[part["n_trials"].notna()]
    if part.empty:
        return pd.DataFrame()
    pivot = part.pivot_table(
        index="n_trials",
        columns="method",
        values="mean_chosen_flatness",
        aggfunc="mean",
    ).reset_index()
    for method in FLATNESS_METHOD_ORDER:
        if method not in pivot.columns:
            pivot[method] = pd.NA
    pivot["minflat_minus_random"] = pivot["minflat"] - pivot["random"]
    pivot["maxflat_minus_random"] = pivot["maxflat"] - pivot["random"]
    pivot["maxflat_minus_minflat"] = pivot["maxflat"] - pivot["minflat"]
    return pivot[
        [
            "n_trials",
            "random",
            "minflat",
            "maxflat",
            "minflat_minus_random",
            "maxflat_minus_random",
            "maxflat_minus_minflat",
        ]
    ].sort_values("n_trials")


def minflat_vs_random(df: pd.DataFrame) -> pd.DataFrame:
    control = flatness_control_table(df)
    if control.empty or not {"random", "minflat"}.issubset(control.columns):
        return pd.DataFrame()
    diff = control[["n_trials", "random", "minflat", "minflat_minus_random"]].copy()
    diff["relative_delta"] = diff["minflat_minus_random"] / diff["random"]
    return diff.sort_values("n_trials")


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
        "mean_candidate_flatness",
        "mean_selection_gain",
        "mean_candidate_unique_count",
        "distinct_1",
        "distinct_2",
        "distinct_3",
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


def key_findings(df: pd.DataFrame) -> list[str]:
    findings: list[str] = []
    duplicates = method_n_duplicates(df)
    if duplicates.empty:
        findings.append("The comparison table is one row per `method, n_trials` configuration.")
    else:
        findings.append(
            "The comparison table still has duplicated `method, n_trials` configurations; "
            "inspect the duplicate table below."
        )

    diff = minflat_vs_random(df)
    if not diff.empty:
        better = int((diff["minflat_minus_random"] < 0).sum())
        findings.append(
            f"`minflat` has lower chosen flatness than `random` for {better}/{len(diff)} "
            "tested N values."
        )

    control = flatness_control_table(df)
    if not control.empty:
        higher = int((control["maxflat_minus_minflat"] > 0).sum())
        findings.append(
            f"`maxflat` is higher than `minflat` for {higher}/{len(control)} tested N "
            "values, which is the expected negative-control direction."
        )

    return findings


def build_analysis_markdown(df: pd.DataFrame, *, table_path: str | Path) -> str:
    lines = []
    lines.append("# Automatic analysis report\n")
    lines.append("## Input\n")
    lines.append(f"- table: `{table_path}`")
    lines.append(f"- rows: {len(df)}\n")

    lines.append("## Key findings\n")
    for finding in key_findings(df):
        lines.append(f"- {finding}")
    lines.append("")

    duplicates = method_n_duplicates(df)
    if not duplicates.empty:
        lines.append("## Duplicate method/N rows\n")
        lines.append(markdown_table(duplicates, ["method", "n_trials", "rows"]))

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
    lines.append(markdown_table(sorted_table(df), summary_cols))

    lines.append("## Flatness-N vs Random-N\n")
    diff_cols = [
        "n_trials",
        "random",
        "minflat",
        "minflat_minus_random",
        "relative_delta",
    ]
    lines.append(markdown_table(minflat_vs_random(df), diff_cols))

    lines.append("## Flatness control table\n")
    control_cols = [
        "n_trials",
        "random",
        "minflat",
        "maxflat",
        "minflat_minus_random",
        "maxflat_minus_random",
        "maxflat_minus_minflat",
    ]
    lines.append(markdown_table(flatness_control_table(df), control_cols))

    lines.append("## N dependence\n")
    n_cols = [
        "method",
        "n_trials",
        "mean_chosen_flatness",
        "mean_candidate_flatness",
        "mean_selection_gain",
        "mean_candidate_unique_count",
        "distinct_2",
        "repetition_4",
        "mean_runtime_seconds",
    ]
    lines.append(markdown_table(n_dependence(df), n_cols))

    lines.append("## Correlation with chosen flatness\n")
    lines.append(markdown_table(correlations(df), ["metric", "corr_with_mean_chosen_flatness"]))

    lines.append("## Interpretation checklist\n")
    lines.append("- Does `minflat` reduce chosen flatness relative to `random` at the same N?")
    lines.append("- Does `maxflat` move in the opposite direction as a negative control?")
    lines.append("- Does increasing N change chosen flatness monotonically or approximately?")
    lines.append("- Does lower flatness trade off with distinct-n or repetition?")
    lines.append("- How quickly does runtime increase as N grows?\n")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.table)
    markdown = build_analysis_markdown(df, table_path=args.table)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
