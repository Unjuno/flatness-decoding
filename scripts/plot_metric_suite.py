from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_METRICS = [
    "mean_chosen_flatness",
    "mean_selection_gain",
    "distinct_1",
    "distinct_2",
    "distinct_3",
    "entropy_1",
    "entropy_2",
    "entropy_3",
    "repetition_2",
    "repetition_3",
    "repetition_4",
    "mean_runtime_seconds",
    "mean_tokens_per_second",
    "ppl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", nargs="+", required=True, help="One or more CSV tables to merge by columns.")
    parser.add_argument("--output-dir", default="results/figures/metric_suite")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    return parser.parse_args()


def load_tables(paths: list[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        frames.append(pd.read_csv(p))
    if not frames:
        raise SystemExit("no input tables found")
    # Conservative concat: works for summary-like files with method/n columns.
    return pd.concat(frames, ignore_index=True, sort=False)


def plot_bar_by_method(df: pd.DataFrame, metric: str, out_dir: Path) -> None:
    if metric not in df.columns or "method" not in df.columns:
        return
    part = df[["method", metric]].dropna()
    if part.empty:
        return
    grouped = part.groupby("method", as_index=False)[metric].mean().sort_values(metric)
    plt.bar(grouped["method"], grouped[metric])
    plt.xlabel("Method")
    plt.ylabel(metric)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / f"bar_{metric}.png")
    plt.close()


def plot_n_curves(df: pd.DataFrame, metric: str, out_dir: Path) -> None:
    required = {"method", "n_trials", metric}
    if not required.issubset(df.columns):
        return
    part = df[list(required)].dropna()
    if part.empty:
        return
    part["n_trials"] = pd.to_numeric(part["n_trials"], errors="coerce")
    part[metric] = pd.to_numeric(part[metric], errors="coerce")
    part = part.dropna()
    if part.empty:
        return
    grouped = part.groupby(["method", "n_trials"], as_index=False)[metric].agg(["mean", "std", "count"]).reset_index()
    for method, mpart in grouped.groupby("method"):
        mpart = mpart.sort_values("n_trials")
        yerr = mpart["std"].fillna(0.0)
        plt.errorbar(mpart["n_trials"], mpart["mean"], yerr=yerr, marker="o", capsize=3, label=str(method))
    plt.xlabel("N trials")
    plt.ylabel(metric)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"curve_N_{metric}.png")
    plt.close()


def plot_metric_scatter(df: pd.DataFrame, x_metric: str, y_metric: str, out_dir: Path) -> None:
    required = {"method", x_metric, y_metric}
    if not required.issubset(df.columns):
        return
    part = df[list(required)].dropna()
    if part.empty:
        return
    for method, mpart in part.groupby("method"):
        plt.scatter(mpart[x_metric], mpart[y_metric], label=str(method))
    plt.xlabel(x_metric)
    plt.ylabel(y_metric)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"scatter_{x_metric}_vs_{y_metric}.png")
    plt.close()


def main() -> None:
    args = parse_args()
    df = load_tables(args.tables)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric in args.metrics:
        plot_bar_by_method(df, metric, out_dir)
        plot_n_curves(df, metric, out_dir)

    for y_metric in ["distinct_2", "repetition_4", "mean_runtime_seconds", "ppl"]:
        plot_metric_scatter(df, "mean_chosen_flatness", y_metric, out_dir)

    print(f"wrote metric suite figures to {out_dir}")


if __name__ == "__main__":
    main()
