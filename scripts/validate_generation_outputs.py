from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from flatness_decoding.io_utils import read_jsonl
from flatness_decoding.metrics import analysis_text_from_row


DEFAULT_REQUIRED_METHODS = [
    "greedy",
    "beam",
    "sampling",
    "temperature",
    "top_p",
    "top_k",
    "random",
    "minflat",
    "maxflat",
]

DEFAULT_REQUIRED_TOKEN_LOG_METHODS = ["random", "minflat", "maxflat"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--token-log-dir", default=None)
    parser.add_argument("--require-methods", nargs="+", default=DEFAULT_REQUIRED_METHODS)
    parser.add_argument("--require-token-log-methods", nargs="+", default=DEFAULT_REQUIRED_TOKEN_LOG_METHODS)
    parser.add_argument("--allow-missing-continuation", action="store_true")
    parser.add_argument("--fail", action="store_true", help="Exit nonzero if errors are found.")
    return parser.parse_args()


def load_generation_rows(input_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    rows_all: list[dict[str, Any]] = []
    files = sorted(input_dir.glob("*.jsonl"))
    if not files:
        errors.append(f"no generation JSONL files found in {input_dir}")
        return rows_all, errors

    for path in files:
        rows = read_jsonl(path)
        if not rows:
            errors.append(f"empty generation file: {path}")
            continue
        for row in rows:
            row = {**row, "_file": str(path)}
            rows_all.append(row)
    return rows_all, errors


def validate_generation_rows(rows: list[dict[str, Any]], allow_missing_continuation: bool) -> list[str]:
    errors: list[str] = []
    required = ["prompt_id", "prompt", "text", "method", "seed"]
    for idx, row in enumerate(rows):
        for key in required:
            if key not in row:
                errors.append(f"row {idx} missing required key {key} in {row.get('_file')}")
        if not allow_missing_continuation and "continuation_text" not in row:
            errors.append(f"row {idx} missing continuation_text in {row.get('_file')}")
        analysis_text = analysis_text_from_row(row)
        if row.get("num_output_tokens") not in (None, 0) and analysis_text == "":
            errors.append(f"row {idx} has empty analysis text but nonzero output tokens in {row.get('_file')}")
    return errors


def validate_required_methods(rows: list[dict[str, Any]], methods: list[str]) -> list[str]:
    errors: list[str] = []
    found = {str(row.get("method")) for row in rows}
    for method in methods:
        if method not in found:
            errors.append(f"required method not found in generation outputs: {method}")
    return errors


def validate_duplicate_rows(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    keys = []
    for row in rows:
        keys.append(
            (
                row.get("prompt_id"),
                row.get("method"),
                row.get("n_trials"),
                row.get("seed"),
                row.get("actual_seed"),
                row.get("shard_index"),
            )
        )
    counts = Counter(keys)
    duplicates = [key for key, count in counts.items() if count > 1]
    if duplicates:
        errors.append(f"found {len(duplicates)} duplicate generation row keys; first={duplicates[0]}")
    return errors


def summarize(rows: list[dict[str, Any]]) -> None:
    by_method: dict[str, int] = defaultdict(int)
    by_method_n: dict[tuple[str, Any], int] = defaultdict(int)
    for row in rows:
        method = str(row.get("method"))
        n_trials = row.get("n_trials")
        by_method[method] += 1
        by_method_n[(method, n_trials)] += 1

    print(f"generation rows: {len(rows)}")
    print("rows by method:")
    for method, count in sorted(by_method.items()):
        print(f"  {method}: {count}")
    print("rows by method/N:")
    for (method, n_trials), count in sorted(by_method_n.items(), key=lambda item: (item[0][0], str(item[0][1]))):
        print(f"  {method}, N={n_trials}: {count}")


def validate_token_logs(token_log_dir: Path | None, required_methods: list[str]) -> list[str]:
    errors: list[str] = []
    if token_log_dir is None:
        return errors
    files = sorted(token_log_dir.glob("*.tokenlog.jsonl"))
    if not files:
        errors.append(f"no token log files found in {token_log_dir}")
        return errors
    found_methods: set[str] = set()
    for path in files:
        rows = read_jsonl(path)
        if not rows:
            errors.append(f"empty token log file: {path}")
            continue
        first = rows[0]
        method = str(first.get("method"))
        found_methods.add(method)
        for key in ["chosen_flatness", "mean_candidate_flatness", "selection_gain"]:
            if key not in first:
                errors.append(f"token log {path} missing key {key}")
    for method in required_methods:
        if method not in found_methods:
            errors.append(f"required token-log method not found: {method}")
    return errors


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    token_log_dir = Path(args.token_log_dir) if args.token_log_dir else None

    rows, errors = load_generation_rows(input_dir)
    if rows:
        summarize(rows)
        errors.extend(validate_generation_rows(rows, args.allow_missing_continuation))
        errors.extend(validate_required_methods(rows, args.require_methods))
        errors.extend(validate_duplicate_rows(rows))
    errors.extend(validate_token_logs(token_log_dir, args.require_token_log_methods))

    if errors:
        print("validation errors:")
        for error in errors:
            print(f"- {error}")
        if args.fail:
            raise SystemExit(1)
    else:
        print("validation passed")


if __name__ == "__main__":
    main()
