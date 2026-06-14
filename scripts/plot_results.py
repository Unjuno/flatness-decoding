from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/summary/metrics.csv")
    parser.add_argument("--output-dir", default="results/figures")
    return parser.parse_args()


def n_trial_only(df: pd.DataFrame) -> pd.DataFrame:
    part = df.copy()
    if "n_trials" not in part.columns:
        return part.iloc[0:0]
    part = part[part["n_trials"].notna()].copy()
    part["n_trials"] = part["n_trials"].astype(float)
    return part


def plot_metric_by_n(df: pd.DataFrame, metric: str, ylabel: str, output_path: Path) -> None:
    if df.empty or metric not in df.columns:
        return
    grouped = df.groupby(["method", "n_trials"], as_index=False)[metric].mean(numeric_only=True)
    if grouped.empty:
        return
    for method, part in grouped.groupby("method"):
        part = part.sort_values("n_trials")
        plt.plot(part["n_trials"], part[metric], marker="o", label=str(method))
    plt.xlabel("N trials")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    token_df = n_trial_only(df[df["kind"] == "token_log"])
    plot_metric_by_n(
        token_df,
        metric="mean_chosen_flatness",
        ylabel="Mean chosen flatness ↓",
        output_path=out_dir / "flatness_vs_n.png",
    )
    plot_metric_by_n(
        token_df,
        metric="mean_selection_gain",
        ylabel="Mean selection gain ↑",
        output_path=out_dir / "selection_gain_vs_n.png",
    )

    gen_df = n_trial_only(df[df["kind"] == "generation"])
    plot_metric_by_n(
        gen_df,
        metric="distinct_2",
        ylabel="Distinct-2 ↑",
        output_path=out_dir / "distinct2_vs_n.png",
    )
    plot_metric_by_n(
        gen_df,
        metric="repetition_4",
        ylabel="Repetition-4 ↓",
        output_path=out_dir / "repetition4_vs_n.png",
    )
    plot_metric_by_n(
        gen_df,
        metric="mean_runtime_seconds",
        ylabel="Mean runtime seconds ↓",
        output_path=out_dir / "runtime_vs_n.png",
    )

    print(f"wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
