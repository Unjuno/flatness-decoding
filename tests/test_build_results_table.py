from __future__ import annotations

import pandas as pd

from scripts.build_results_table import build_results_table


def test_build_results_table_merges_generation_and_token_metrics() -> None:
    metrics = pd.DataFrame(
        [
            {
                "kind": "generation",
                "experiment_name": "comparison_smoke",
                "model": "distilgpt2",
                "method": "maxflat",
                "method_family": "flatness",
                "n_trials": 4,
                "seed": 0,
                "mean_output_tokens": 52.75,
                "mean_runtime_seconds": 50.2,
                "distinct_1": 0.76,
                "repetition_4": 0.01,
            },
            {
                "kind": "token_log",
                "experiment_name": None,
                "model": None,
                "method": "maxflat",
                "method_family": None,
                "n_trials": "4",
                "seed": 0,
                "mean_chosen_flatness": 3.9,
                "mean_candidate_flatness": 3.5,
                "mean_selection_gain": 0.1,
                "mean_candidate_unique_count": 3.0,
            },
        ]
    )

    table = build_results_table(metrics)

    assert len(table) == 1
    row = table.iloc[0]
    assert row["method"] == "maxflat"
    assert row["n_trials"] == 4
    assert row["experiment_name"] == "comparison_smoke"
    assert row["model"] == "distilgpt2"
    assert row["method_family"] == "flatness"
    assert row["distinct_1"] == 0.76
    assert row["mean_chosen_flatness"] == 3.9
    assert row["mean_selection_gain"] == 0.1


def test_build_results_table_keeps_generation_only_baseline() -> None:
    metrics = pd.DataFrame(
        [
            {
                "kind": "generation",
                "experiment_name": "comparison_smoke",
                "model": "distilgpt2",
                "method": "greedy",
                "method_family": "baseline",
                "n_trials": None,
                "seed": 0,
                "mean_output_tokens": 60.0,
                "distinct_1": 0.63,
            }
        ]
    )

    table = build_results_table(metrics)

    assert len(table) == 1
    row = table.iloc[0]
    assert row["method"] == "greedy"
    assert pd.isna(row["n_trials"])
    assert row["mean_output_tokens"] == 60.0
    assert row["distinct_1"] == 0.63
