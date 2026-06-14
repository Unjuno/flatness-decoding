from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML experiment-set config.")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    generations_dir = cfg["output"]["generations_dir"]
    token_log_dir = cfg["output"]["token_log_dir"]
    summary_output = cfg["output"].get("summary_csv", "results/summary/metrics.csv")
    table_output = cfg["output"].get(
        "table_csv",
        str(Path(summary_output).with_name(Path(summary_output).stem.replace("metrics", "table") + ".csv")),
    )
    analysis_output = cfg["output"].get(
        "analysis_md",
        str(Path(summary_output).with_name(Path(summary_output).stem.replace("metrics", "analysis") + ".md")),
    )
    figures_dir = cfg["output"].get("figures_dir", "results/figures")
    token_trace_summary = cfg["output"].get(
        "token_trace_summary_csv",
        str(Path(summary_output).with_name(Path(summary_output).stem.replace("metrics", "token_trace") + ".csv")),
    )

    if not args.skip_generate:
        run([sys.executable, "scripts/run_experiment_set.py", "--config", args.config])

    run(
        [
            sys.executable,
            "scripts/run_evaluate.py",
            "--input-dir",
            generations_dir,
            "--token-log-dir",
            token_log_dir,
            "--output",
            summary_output,
        ]
    )

    run(
        [
            sys.executable,
            "scripts/build_results_table.py",
            "--metrics",
            summary_output,
            "--output",
            table_output,
        ]
    )

    if not args.skip_analysis:
        run(
            [
                sys.executable,
                "scripts/analyze_results.py",
                "--table",
                table_output,
                "--output",
                analysis_output,
            ]
        )
        run(
            [
                sys.executable,
                "scripts/analyze_token_logs.py",
                "--token-log-dir",
                token_log_dir,
                "--output-dir",
                str(Path(figures_dir) / "token_analysis"),
                "--summary-output",
                token_trace_summary,
            ]
        )

    if not args.skip_plot:
        run([sys.executable, "scripts/plot_results.py", "--input", summary_output, "--output-dir", figures_dir])


if __name__ == "__main__":
    main()
