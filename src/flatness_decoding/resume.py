from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import read_jsonl


def safe_read_jsonl(path: str | Path) -> list[dict[str, Any]] | None:
    p = Path(path)
    if not p.exists() or not p.is_file() or p.stat().st_size == 0:
        return None
    try:
        return read_jsonl(p)
    except Exception:
        return None


def rows_match_common_fields(
    rows: list[dict[str, Any]],
    *,
    expected_rows: int,
    method: str,
    seed: int,
    shard_index: int,
    num_shards: int,
) -> bool:
    if len(rows) != expected_rows:
        return False
    if expected_rows == 0:
        return False
    for row in rows:
        if row.get("method") != method:
            return False
        if int(row.get("seed", -1)) != int(seed):
            return False
        if int(row.get("shard_index", -1)) != int(shard_index):
            return False
        if int(row.get("num_shards", -1)) != int(num_shards):
            return False
        if "prompt_id" not in row or "prompt_index" not in row:
            return False
        if row.get("text") is None or row.get("continuation_text") is None:
            return False
        if row.get("num_output_tokens") is None:
            return False
    return True


def baseline_output_complete(
    path: str | Path,
    *,
    expected_rows: int,
    method: str,
    seed: int,
    shard_index: int,
    num_shards: int,
) -> bool:
    rows = safe_read_jsonl(path)
    if rows is None:
        return False
    if not rows_match_common_fields(
        rows,
        expected_rows=expected_rows,
        method=method,
        seed=seed,
        shard_index=shard_index,
        num_shards=num_shards,
    ):
        return False
    return all(row.get("method_family") == "baseline" for row in rows)


def ntrial_output_complete(
    generation_path: str | Path,
    token_log_path: str | Path,
    *,
    expected_rows: int,
    method: str,
    n_trials: int,
    seed: int,
    shard_index: int,
    num_shards: int,
) -> bool:
    rows = safe_read_jsonl(generation_path)
    token_rows = safe_read_jsonl(token_log_path)
    if rows is None or token_rows is None:
        return False
    if not rows_match_common_fields(
        rows,
        expected_rows=expected_rows,
        method=method,
        seed=seed,
        shard_index=shard_index,
        num_shards=num_shards,
    ):
        return False
    if not all(row.get("method_family") == "n_trial" for row in rows):
        return False
    if not all(int(row.get("n_trials", -1)) == int(n_trials) for row in rows):
        return False

    expected_token_rows = 0
    for row in rows:
        try:
            expected_token_rows += int(row.get("num_output_tokens", 0))
        except Exception:
            return False
    if len(token_rows) != expected_token_rows:
        return False

    for token_row in token_rows:
        if token_row.get("method") != method:
            return False
        if int(token_row.get("n_trials", -1)) != int(n_trials):
            return False
        if int(token_row.get("seed", -1)) != int(seed):
            return False
        if int(token_row.get("shard_index", -1)) != int(shard_index):
            return False
        if int(token_row.get("num_shards", -1)) != int(num_shards):
            return False
        if token_row.get("candidate_ids") is None:
            return False
        if token_row.get("candidate_flatness") is None:
            return False
        if token_row.get("chosen_id") is None:
            return False
    return True


def resume_config(cfg: dict) -> dict[str, bool]:
    runtime_cfg = cfg.get("runtime", {}) or {}
    resume_cfg = runtime_cfg.get("resume", {}) or {}
    return {"skip_complete": bool(resume_cfg.get("skip_complete", True))}
