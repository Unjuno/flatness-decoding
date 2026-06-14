from __future__ import annotations

import pandas as pd

from scripts.analyze_results import (
    build_analysis_markdown,
    flatness_control_table,
    method_n_duplicates,
    minflat_vs_random,
)


def clean_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method": "random",
                "method_family": "n_trial",
                "n_trials": 1,
                "mean_chosen_flatness": 3.0,
                "mean_candidate_flatness": 3.0,
                "mean_selection_gain": 0.0,
                "mean_candidate_unique_count": 1.0,
                "distinct_2": 0.90,
                "repetition_4": 0.02,
                "mean_runtime_seconds": 10.0,
            },
            {
                "method": "minflat",
                "method_family": "n_trial",
                "n_trials": 1,
                "mean_chosen_flatness": 2.0,
                "mean_candidate_flatness": 2.8,
                "mean_selection_gain": 0.2,
                "mean_candidate_unique_count": 1.0,
                "distinct_2": 0.92,
                "repetition_4": 0.01,
                "mean_runtime_seconds": 12.0,
            },
            {
                "method": "maxflat",
                "method_family": "n_trial",
                "n_trials": 1,
                "mean_chosen_flatness": 4.0,
                "mean_candidate_flatness": 3.1,
                "mean_selection_gain": 0.3,
                "mean_candidate_unique_count": 1.0,
                "distinct_2": 0.91,
                "repetition_4": 0.03,
                "mean_runtime_seconds": 11.0,
            },
            {
                "method": "greedy",
                "method_family": "baseline",
                "n_trials": None,
                "distinct_2": 0.70,
                "repetition_4": 0.10,
                "mean_runtime_seconds": 1.0,
            },
        ]
    )


def test_minflat_vs_random_reports_delta() -> None:
    diff = minflat_vs_random(clean_table())

    assert len(diff) == 1
    row = diff.iloc[0]
    assert row["n_trials"] == 1
    assert row["random"] == 3.0
    assert row["minflat"] == 2.0
    assert row["minflat_minus_random"] == -1.0


def test_flatness_control_table_includes_maxflat_control() -> None:
    control = flatness_control_table(clean_table())

    assert len(control) == 1
    row = control.iloc[0]
    assert row["maxflat"] == 4.0
    assert row["maxflat_minus_minflat"] == 2.0


def test_method_n_duplicates_is_empty_for_clean_table() -> None:
    assert method_n_duplicates(clean_table()).empty


def test_analysis_markdown_includes_clean_table_sections() -> None:
    markdown = build_analysis_markdown(clean_table(), table_path="table.csv")

    assert "# Automatic analysis report" in markdown
    assert "## Key findings" in markdown
    assert "one row per `method, n_trials`" in markdown
    assert "## Flatness-N vs Random-N" in markdown
    assert "## Flatness control table" in markdown
    assert "maxflat_minus_minflat" in markdown
