from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from flatness_decoding.io_utils import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-log-dir", required=True)
    parser.add_argument("--output-dir", default="results/figures/token_analysis")
    parser.add_argument("--summary-output", default="results/summary/token_trace_summary.csv")
    return parser.parse_args()


def load_token_logs(token_log_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(token_log_dir.glob("*.tokenlog.jsonl")):
        for row in read_jsonl(path):
            row["source_file"] = str(path)
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    token_log_dir = Path(args.token_log_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_token_logs(token_log_dir)
    if df.empty:
        print("no token logs found")
        return

    for col in ["step", "n_trials", "chosen_flatness", "selection_gain"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    group_cols = [col for col in ["method", "n_trials", "step"] if col in df.columns]
    trace = (
        df.groupby(group_cols, dropna=False)[["chosen_flatness", "selection_gain"]]
        .mean()
        .reset_index()
    )

    summary_cols = [col for col in ["method", "n_trials"] if col in df.columns]
    summary = (
        df.groupby(summary_cols, dropna=False)[["chosen_flatness", "selection_gain"]]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = ["_".join([str(x) for x in col if str(x)]) for col in summary.columns]
    summary_output = Path(args.summary_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_output, index=False)

    if {"method", "n_trials", "step", "chosen_flatness"}.issubset(trace.columns):
        for (method, n_trials), part in trace.groupby(["method", "n_trials"]):
            part = part.sort_values("step")
            plt.plot(part["step"], part["chosen_flatness"], label=f"{method} N={n_trials}")
        plt.xlabel("Generation step")
        plt.ylabel("Mean chosen flatness ↓")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "flatness_trace.png")
        plt.close()

    if {"method", "n_trials", "step", "selection_gain"}.issubset(trace.columns):
        for (method, n_trials), part in trace.groupby(["method", "n_trials"]):
            part = part.sort_values("step")
            plt.plot(part["step"], part["selection_gain"], label=f"{method} N={n_trials}")
        plt.xlabel("Generation step")
        plt.ylabel("Mean selection gain ↑")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "selection_gain_trace.png")
        plt.close()

    print(f"wrote token trace summary to {summary_output}")
    print(f"wrote figures to {output_dir}")


if __name__ == "__main__":
    main()
