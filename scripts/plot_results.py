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


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    token_df = df[df["kind"] == "token_log"].copy()
    if not token_df.empty and "mean_chosen_flatness" in token_df.columns:
        token_df["n"] = token_df["file"].str.extract(r"_n=(\d+)_").astype(float)
        token_df["method"] = token_df["file"].str.extract(r"method=([^_]+)_")
        grouped = token_df.groupby(["method", "n"], as_index=False)["mean_chosen_flatness"].mean()

        for method, part in grouped.groupby("method"):
            plt.plot(part["n"], part["mean_chosen_flatness"], marker="o", label=method)
        plt.xlabel("N trials")
        plt.ylabel("Mean chosen flatness ↓")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "flatness_vs_n.png")
        plt.close()

    gen_df = df[df["kind"] == "generation"].copy()
    if not gen_df.empty and "distinct_2" in gen_df.columns:
        gen_df["n"] = gen_df["file"].str.extract(r"_n=(\d+)_").astype(float)
        gen_df["method"] = gen_df["file"].str.extract(r"method=([^_]+)_")
        grouped = gen_df.groupby(["method", "n"], as_index=False)["distinct_2"].mean()

        for method, part in grouped.groupby("method"):
            plt.plot(part["n"], part["distinct_2"], marker="o", label=method)
        plt.xlabel("N trials")
        plt.ylabel("Distinct-2 ↑")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "distinct2_vs_n.png")
        plt.close()

    print(f"wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
