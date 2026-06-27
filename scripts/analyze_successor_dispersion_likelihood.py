#!/usr/bin/env python3
"""Descriptive correlations between successor logit dispersion and likelihood-related metrics."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GPT2_MERGED = ROOT / "results" / "summary" / "comparison_gpt2_writingprompts_merged_table.csv"
DISTIL_MERGED = ROOT / "results" / "summary" / "comparison_distilgpt2_writingprompts_merged_table.csv"
CASE_REVIEW = ROOT / "results" / "analysis" / "writingprompts_gpt2_small_case_review_summary.csv"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def norm(s: object) -> str:
    return str(s).strip().lower()


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {norm(c): c for c in df.columns}
    for candidate in candidates:
        key = norm(candidate)
        if key in lookup:
            return lookup[key]
    return None


def find_ppl_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if "ppl" in norm(col):
            return col
    return None


def find_n_trials_column(df: pd.DataFrame) -> str | None:
    for candidate in ["n_trials", "N", "n", "num_trials"]:
        col = find_column(df, [candidate])
        if col is not None:
            return col
    return None


def to_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def pearson(x: pd.Series, y: pd.Series) -> float | None:
    data = pd.DataFrame({"x": to_numeric_series(x), "y": to_numeric_series(y)}).dropna()
    if len(data) < 2:
        return None
    if data["x"].nunique() < 2 or data["y"].nunique() < 2:
        return None
    return float(data["x"].corr(data["y"], method="pearson"))


def spearman(x: pd.Series, y: pd.Series) -> float | None:
    data = pd.DataFrame({"x": to_numeric_series(x), "y": to_numeric_series(y)}).dropna()
    if len(data) < 2:
        return None
    if data["x"].nunique() < 2 or data["y"].nunique() < 2:
        return None
    xr = data["x"].rank(method="average")
    yr = data["y"].rank(method="average")
    return float(xr.corr(yr, method="pearson"))


def corr_record(
    model: str,
    subset: str,
    x_name: str,
    y_name: str,
    n_rows: int,
    pearson_value: float | None,
    spearman_value: float | None,
    note: str = "",
) -> dict[str, object]:
    return {
        "model": model,
        "subset": subset,
        "x": x_name,
        "y": y_name,
        "n_rows": n_rows,
        "pearson": pearson_value,
        "spearman": spearman_value,
        "note": note,
    }


def subset_by_n(df: pd.DataFrame, n_col: str | None, target_n: int) -> pd.DataFrame:
    if n_col is None:
        return df.iloc[0:0].copy()
    numeric = to_numeric_series(df[n_col])
    return df.loc[numeric == target_n].copy()


def maybe_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return find_column(df, candidates)


def analyze_model(df: pd.DataFrame, model: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    flat_col = maybe_column(df, ["mean_chosen_flatness"])
    if flat_col is None:
        rows.append(corr_record(model, "all", "mean_chosen_flatness", "missing", 0, None, None, "missing mean_chosen_flatness"))
        return rows

    ppl_col = find_ppl_column(df)
    rank_col = maybe_column(df, ["mean_chosen_current_rank"])
    distinct2_col = maybe_column(df, ["distinct_2"])
    rep4_col = maybe_column(df, ["repetition_4"])
    n_col = find_n_trials_column(df)

    pairs = [
        ("ppl", ppl_col),
        ("mean_chosen_current_rank", rank_col),
        ("distinct_2", distinct2_col),
        ("repetition_4", rep4_col),
    ]

    subsets: list[tuple[str, pd.DataFrame]] = [("all rows", df)]
    if n_col is not None:
        nvals = to_numeric_series(df[n_col])
        ntrial_df = df.loc[nvals.notna()].copy()
        if not ntrial_df.empty:
            subsets.append(("N-trial rows", ntrial_df))
            for n in [8, 16, 32]:
                subset = subset_by_n(df, n_col, n)
                if len(subset) >= 2:
                    subsets.append((f"N={n}", subset))

    for subset_name, subset in subsets:
        x = subset[flat_col]
        for label, y_col in pairs:
            if y_col is None or y_col not in subset.columns:
                rows.append(corr_record(model, subset_name, flat_col, label, len(subset), None, None, f"missing {label}"))
                continue
            rows.append(
                corr_record(
                    model,
                    subset_name,
                    flat_col,
                    label,
                    len(subset),
                    pearson(x, subset[y_col]),
                    spearman(x, subset[y_col]),
                )
            )
    return rows


def analyze_case_review(gpt2_df: pd.DataFrame, review_df: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if gpt2_df.empty or review_df.empty:
        rows.append(
            corr_record(
                "gpt2_case_review",
                "joined by method+n_trials",
                "mean_chosen_flatness",
                "case_review_metrics",
                0,
                None,
                None,
                "missing input tables",
            )
        )
        return rows

    gpt2 = gpt2_df.copy()
    review = review_df.copy()
    gpt2_n = find_n_trials_column(gpt2)
    review_n = find_n_trials_column(review)
    if gpt2_n is not None:
        gpt2[gpt2_n] = to_numeric_series(gpt2[gpt2_n])
    if review_n is not None:
        review[review_n] = to_numeric_series(review[review_n])

    joined = pd.merge(review, gpt2, on=["method", "n_trials"], how="inner", suffixes=("_review", "_gpt2"))
    if len(joined) < 2:
        rows.append(
            corr_record(
                "gpt2_case_review",
                "joined by method+n_trials",
                "mean_chosen_flatness",
                "case_review_metrics",
                len(joined),
                None,
                None,
                "join too small to compute correlations",
            )
        )
        return rows

    flat_col = "mean_chosen_flatness"
    if flat_col not in joined.columns:
        rows.append(
            corr_record(
                "gpt2_case_review",
                "joined by method+n_trials",
                "mean_chosen_flatness",
                "case_review_metrics",
                len(joined),
                None,
                None,
                "missing mean_chosen_flatness after join",
            )
        )
        return rows

    targets = [
        "mean_severity",
        "severe_issue_rate",
        "mean_story_fit_score",
        "usable_rate",
    ]
    for target in targets:
        if target not in joined.columns:
            rows.append(corr_record("gpt2_case_review", "joined by method+n_trials", flat_col, target, len(joined), None, None, f"missing {target}"))
            continue
        rows.append(
            corr_record(
                "gpt2_case_review",
                "joined by method+n_trials",
                flat_col,
                target,
                len(joined),
                pearson(joined[flat_col], joined[target]),
                spearman(joined[flat_col], joined[target]),
            )
        )
    return rows


def format_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return f"{value:.6f}" if isinstance(value, float) else str(value)


def main() -> int:
    if not GPT2_MERGED.exists():
        raise SystemExit(f"missing input: {GPT2_MERGED}")
    if not DISTIL_MERGED.exists():
        raise SystemExit(f"missing input: {DISTIL_MERGED}")
    if not CASE_REVIEW.exists():
        raise SystemExit(f"missing input: {CASE_REVIEW}")

    gpt2 = read_csv(GPT2_MERGED)
    distil = read_csv(DISTIL_MERGED)
    review = read_csv(CASE_REVIEW)

    records: list[dict[str, object]] = []
    records.extend(analyze_model(gpt2, "gpt2"))
    records.extend(analyze_model(distil, "distilgpt2"))
    records.extend(analyze_case_review(gpt2, review))

    out_df = pd.DataFrame(records)
    corr_path = OUT_DIR / "successor_dispersion_likelihood_correlations.csv"
    out_df.to_csv(corr_path, index=False)

    lines = [
        "# Successor Dispersion vs Likelihood Diagnostics",
        "",
        "This note is descriptive. It compares summary-table successor logit dispersion with summary-table likelihood-related columns and a small GPT-2 case review. The analysis is method-level, small-sample, and does not establish causality.",
        "",
        "## Questions",
        "",
        "1. Is successor logit dispersion identical to PPL/current likelihood? No. The summary tables show related but non-identical behavior.",
        "2. Is it likelihood-related? Yes. The descriptive correlations are consistent with a likelihood-sensitive signal.",
        "3. What does the correlation analysis suggest? The signal tracks likelihood-related columns in some settings, but the relationship varies by method family and subset.",
        "4. Why should the paper avoid claiming orthogonality? Because the data do not support it and the signal clearly interacts with likelihood-sensitive behavior.",
        "5. Why should the paper avoid claiming it is just PPL? Because successor logit dispersion measures candidate-induced successor-state shape, while PPL measures realized sequence surprisal.",
        "6. Which method patterns illustrate the distinction?",
        "   - `minflat`: low dispersion but bad rank/PPL and severe degeneration.",
        "   - `highprob_centerflat`: likelihood-friendly but repetitive/generic.",
        "   - `centerflat_q50` / `centerflat_q75`: intermediate balanced regimes.",
        "   - `maxflat`: high-dispersion diagnostic.",
        "",
        "## Correlation summary",
        "",
    ]
    if out_df.empty:
        lines.append("No correlations could be computed.")
    else:
        for _, row in out_df.iterrows():
            lines.append(
                f"- {row['model']} | {row['subset']} | {row['x']} vs {row['y']} | n={int(row['n_rows'])} | pearson={format_value(row['pearson'])} | spearman={format_value(row['spearman'])}"
            )
            note = str(row.get("note", "")).strip()
            if note:
                lines.append(f"  - caveat: {note}")

    lines += [
        "",
        "## Interpretation",
        "",
        "The descriptive correlations should be interpreted cautiously. They are computed on summary tables, not token-level predictions, and they mix baseline and N-trial rows when the relevant subset is broad. The main takeaway is conservative: successor logit dispersion is likelihood-related but not identical to PPL/current-token likelihood, and the completed methods behave as distinct control regimes rather than as a single monotonic likelihood axis.",
        "",
        "## Caveats",
        "",
        "- Small sample.",
        "- Method-level aggregate.",
        "- Does not establish causality.",
        "- Missing columns are recorded as caveats rather than failures.",
    ]
    diag_path = OUT_DIR / "successor_dispersion_likelihood_diagnostics.md"
    diag_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {corr_path}")
    print(f"wrote {diag_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
