#!/usr/bin/env python3
"""Create paper-facing figure and table artifacts from existing results only."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Iterable

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(f"pandas is required for this script: {exc}")

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "paper" / "figures"
PAPER_DIR.mkdir(parents=True, exist_ok=True)

GPT2_MERGED = ROOT / "results" / "summary" / "comparison_gpt2_writingprompts_merged_table.csv"
KEY_RESULTS = ROOT / "results" / "summary" / "writingprompts_key_results.csv"
CASE_REVIEW = ROOT / "results" / "analysis" / "writingprompts_gpt2_small_case_review_summary.csv"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def norm(name: str) -> str:
    return str(name).strip().lower()


def find_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    cols = {norm(c): c for c in df.columns}
    for candidate in candidates:
        key = norm(candidate)
        if key in cols:
            return cols[key]
    return None


def find_ppl_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if "ppl" in norm(col):
            return col
    return None


def find_n_trials_column(df: pd.DataFrame) -> str | None:
    for candidate in ("n_trials", "N", "n", "num_trials"):
        col = find_column(df, [candidate])
        if col is not None:
            return col
    return None


def normalize_method(value: object) -> str:
    return str(value).strip()


def normalize_n_trials(value: object) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        num = float(value)
    except Exception:
        text = str(value).strip()
        try:
            return int(text)
        except Exception:
            return None
    if pd.isna(num):
        return None
    if abs(num - round(num)) < 1e-9:
        return int(round(num))
    return int(num)


def select_rows(df: pd.DataFrame, method: str, n_trials: int | None = None) -> pd.DataFrame:
    work = df.copy()
    method_col = find_column(work, ["method"])
    if method_col is None:
        raise SystemExit("Could not find method column")
    mask = work[method_col].astype(str).str.strip().str.lower() == method.lower()
    if n_trials is not None:
        n_col = find_n_trials_column(work)
        if n_col is not None:
            work_n = work[n_col].apply(normalize_n_trials)
            mask &= work_n == n_trials
        else:
            # fallback: infer from method labels if encoded there
            mask &= work[method_col].astype(str).str.contains(rf"\b{n_trials}\b", regex=True)
    out = work.loc[mask].copy()
    if out.empty and n_trials is not None:
        return out
    return out


def choose_best_row(df: pd.DataFrame, method: str, n_trials: int | None = None) -> pd.Series | None:
    rows = select_rows(df, method, n_trials)
    if rows.empty:
        return None
    return rows.iloc[0]


def build_selected_table(
    df: pd.DataFrame,
    selections: list[tuple[str, int | None]],
    columns: list[str],
    out_path: Path,
) -> pd.DataFrame:
    rows = []
    for method, n_trials in selections:
        row = choose_best_row(df, method, n_trials)
        if row is None:
            continue
        record = {}
        for col in columns:
            record[col] = row.get(col, None)
        rows.append(record)
    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        if "method" in out_df.columns:
            out_df["method"] = out_df["method"].map(normalize_method)
        if "n_trials" in out_df.columns:
            out_df["n_trials"] = out_df["n_trials"].map(normalize_n_trials)
        out_df = out_df.sort_values(by=[c for c in ["method", "n_trials"] if c in out_df.columns], kind="stable")
    out_df.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)
    return out_df


def cast_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def add_if_present(source: pd.Series, target: dict[str, object], source_name: str, target_name: str | None = None) -> None:
    if source_name in source.index:
        target[target_name or source_name] = source[source_name]


def generate_flatness_vs_n(df: pd.DataFrame) -> pd.DataFrame:
    selections = [(m, n) for m in ["random", "minflat", "maxflat", "centerflat_q50", "centerflat_q75", "highprob_centerflat"] for n in [1, 8, 16, 32]]
    # Keep only rows that actually exist in the source.
    cols = ["method", "n_trials", "mean_chosen_flatness", "mean_runtime_seconds", "ppl", "mean_chosen_current_rank", "distinct_2", "repetition_4"]
    out_rows = []
    for method, n_trials in selections:
        row = choose_best_row(df, method, n_trials)
        if row is None:
            continue
        record: dict[str, object] = {"method": row.get("method"), "n_trials": normalize_n_trials(row.get(find_n_trials_column(df) or "n_trials"))}
        for col in cols[2:]:
            source_col = col
            if col == "ppl":
                source_col = find_ppl_column(df) or "ppl"
            if source_col in row.index:
                record[col] = row.get(source_col)
        out_rows.append(record)
    out = pd.DataFrame(out_rows)
    if not out.empty:
        out["method"] = out["method"].map(normalize_method)
        out["n_trials"] = out["n_trials"].map(normalize_n_trials)
        out = out.sort_values(["method", "n_trials"], kind="stable")
    out_path = PAPER_DIR / "writingprompts_gpt2_flatness_vs_n.csv"
    out.to_csv(out_path, index=False)
    return out


def generate_tradeoff_table(df: pd.DataFrame) -> pd.DataFrame:
    selections = [
        ("top_k", None),
        ("top_p", None),
        ("random", 8),
        ("minflat", 8),
        ("maxflat", 8),
        ("centerflat_q50", 8),
        ("centerflat_q75", 8),
        ("highprob_centerflat", 8),
        ("centerflat_q75", 16),
        ("highprob_centerflat", 16),
        ("random", 32),
        ("minflat", 32),
        ("maxflat", 32),
        ("centerflat_q50", 32),
        ("centerflat_q75", 32),
        ("highprob_centerflat", 32),
    ]
    rows = []
    ppl_col = find_ppl_column(df)
    n_col = find_n_trials_column(df)
    for method, n_trials in selections:
        row = choose_best_row(df, method, n_trials)
        if row is None:
            continue
        record = {
            "method": row.get("method"),
            "n_trials": normalize_n_trials(row.get(n_col)) if n_col and n_col in row.index else (n_trials if n_trials is not None else None),
        }
        for col in ["ppl", "distinct_2", "repetition_4", "mean_chosen_flatness", "mean_chosen_current_rank", "mean_runtime_seconds"]:
            source_col = ppl_col if col == "ppl" else col
            record[col] = row.get(source_col) if source_col in row.index else None
        rows.append(record)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["method"] = out["method"].map(normalize_method)
        out["n_trials"] = out["n_trials"].map(normalize_n_trials)
    out_path = PAPER_DIR / "writingprompts_gpt2_tradeoff_table.csv"
    out.to_csv(out_path, index=False)
    return out


def generate_case_review_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map = {}
    for target, candidates in {
        "method": ["method"],
        "n_trials": ["n_trials", "N", "n", "num_trials"],
        "n_items": ["n_items"],
        "mean_severity": ["mean_severity"],
        "severe_issue_rate": ["severe_issue_rate"],
        "mean_story_fit_score": ["mean_story_fit_score"],
        "usable_rate": ["usable_rate"],
        "prompt_drift_rate": ["prompt_drift_rate"],
        "generic_rate": ["generic_rate", "generic_filler_rate"],
        "repetition_rate": ["repetition_rate", "repetition_loop_rate"],
        "symbol_or_encoding_noise_rate": ["symbol_or_encoding_noise_rate"],
        "incoherent_rate": ["incoherent_rate", "incoherent_event_sequence_rate"],
    }.items():
        col = find_column(out, candidates)
        if col is not None:
            rename_map[col] = target
    out = out.rename(columns=rename_map)
    keep = [c for c in ["method", "n_trials", "n_items", "mean_severity", "severe_issue_rate", "mean_story_fit_score", "usable_rate", "prompt_drift_rate", "generic_rate", "repetition_rate", "symbol_or_encoding_noise_rate", "incoherent_rate"] if c in out.columns]
    out = out[keep].copy()
    if "method" in out.columns:
        out["method"] = out["method"].map(normalize_method)
    if "n_trials" in out.columns:
        out["n_trials"] = out["n_trials"].map(normalize_n_trials)
    out_path = PAPER_DIR / "writingprompts_gpt2_case_review_summary.csv"
    out.to_csv(out_path, index=False)
    return out


def maybe_plot_flatness_vs_n(df: pd.DataFrame) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"warning: matplotlib unavailable for flatness plot: {exc}", file=sys.stderr)
        return False
    if df.empty:
        print("warning: no rows for flatness vs N plot", file=sys.stderr)
        return False
    plt.figure(figsize=(8, 5))
    for method, sub in df.groupby("method"):
        if "n_trials" not in sub.columns or "mean_chosen_flatness" not in sub.columns:
            continue
        s = sub.dropna(subset=["n_trials", "mean_chosen_flatness"]).sort_values("n_trials")
        if s.empty:
            continue
        plt.plot(s["n_trials"], s["mean_chosen_flatness"], marker="o", label=method)
    plt.xlabel("n_trials")
    plt.ylabel("mean chosen successor logit dispersion")
    plt.title("WritingPrompts GPT-2: successor dispersion vs N")
    plt.legend(fontsize=8)
    plt.tight_layout()
    out_path = PAPER_DIR / "writingprompts_gpt2_flatness_vs_n.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    return True


def maybe_plot_case_review(df: pd.DataFrame) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"warning: matplotlib unavailable for case review plot: {exc}", file=sys.stderr)
        return False
    if df.empty or "mean_severity" not in df.columns:
        print("warning: no rows for case review plot", file=sys.stderr)
        return False
    s = df.sort_values(["method", "n_trials"], kind="stable").copy()
    labels = s.apply(lambda r: f"{r['method']}" + (f" N={int(r['n_trials'])}" if pd.notna(r.get("n_trials")) else ""), axis=1)
    plt.figure(figsize=(10, max(4, 0.35 * len(s))))
    plt.barh(labels, s["mean_severity"])
    plt.xlabel("mean severity")
    plt.title("WritingPrompts GPT-2 small case review severity")
    plt.tight_layout()
    out_path = PAPER_DIR / "writingprompts_gpt2_case_review_severity.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    return True


def main() -> int:
    if not GPT2_MERGED.exists():
        raise SystemExit(f"missing input: {GPT2_MERGED}")
    if not KEY_RESULTS.exists():
        raise SystemExit(f"missing input: {KEY_RESULTS}")
    if not CASE_REVIEW.exists():
        raise SystemExit(f"missing input: {CASE_REVIEW}")

    gpt2 = read_csv(GPT2_MERGED)
    key = read_csv(KEY_RESULTS)
    case = read_csv(CASE_REVIEW)

    flatness = generate_flatness_vs_n(gpt2)
    tradeoff = generate_tradeoff_table(gpt2)
    case_out = generate_case_review_table(case)

    plotted = []
    if maybe_plot_flatness_vs_n(flatness):
        plotted.append("writingprompts_gpt2_flatness_vs_n.png")
    if maybe_plot_case_review(case_out):
        plotted.append("writingprompts_gpt2_case_review_severity.png")

    print("generated CSVs:")
    for name in [
        "writingprompts_gpt2_flatness_vs_n.csv",
        "writingprompts_gpt2_tradeoff_table.csv",
        "writingprompts_gpt2_case_review_summary.csv",
    ]:
        print(f"- {PAPER_DIR / name}")
    if plotted:
        print("generated PNGs:")
        for name in plotted:
            print(f"- {PAPER_DIR / name}")
    else:
        print("generated PNGs: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
