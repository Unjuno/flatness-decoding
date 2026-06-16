import json
from pathlib import Path

from scripts.validate_generation_outputs import (
    load_generation_rows,
    validate_duplicate_rows,
    validate_generation_rows,
    validate_required_methods,
    validate_token_logs,
)


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def valid_generation_row(method="minflat"):
    return {
        "prompt_id": "p0",
        "prompt": "Hello",
        "text": "Hello world",
        "continuation_text": "world",
        "method": method,
        "seed": 0,
        "actual_seed": 0,
        "n_trials": 1,
        "shard_index": 0,
        "num_output_tokens": 1,
    }


def test_validate_generation_rows_accepts_valid_row():
    errors, warnings = validate_generation_rows(
        [valid_generation_row()],
        allow_missing_continuation=False,
        allow_empty_analysis_text=False,
    )
    assert errors == []
    assert warnings == []


def test_validate_generation_rows_rejects_missing_continuation():
    row = valid_generation_row()
    row.pop("continuation_text")
    errors, warnings = validate_generation_rows(
        [row],
        allow_missing_continuation=False,
        allow_empty_analysis_text=False,
    )
    assert any("continuation_text" in error for error in errors)
    assert warnings == []


def test_validate_generation_rows_can_warn_on_empty_analysis_text():
    row = valid_generation_row()
    row["continuation_text"] = ""
    errors, warnings = validate_generation_rows(
        [row],
        allow_missing_continuation=False,
        allow_empty_analysis_text=True,
    )
    assert errors == []
    assert any("empty analysis text" in warning for warning in warnings)


def test_validate_required_methods_detects_missing_method():
    errors = validate_required_methods([valid_generation_row(method="minflat")], ["minflat", "random"])
    assert any("random" in error for error in errors)


def test_validate_duplicate_rows_detects_duplicate_key():
    row = valid_generation_row()
    errors = validate_duplicate_rows([row, dict(row)])
    assert any("duplicate" in error for error in errors)


def test_load_generation_rows_reads_jsonl(tmp_path):
    path = tmp_path / "gens" / "out.jsonl"
    write_jsonl(path, [valid_generation_row()])
    rows, errors = load_generation_rows(path.parent)
    assert errors == []
    assert len(rows) == 1


def test_validate_token_logs_detects_required_method(tmp_path):
    log_dir = tmp_path / "logs"
    write_jsonl(
        log_dir / "a.tokenlog.jsonl",
        [
            {
                "method": "minflat",
                "chosen_flatness": 1.0,
                "mean_candidate_flatness": 2.0,
                "selection_gain": 0.5,
            }
        ],
    )
    errors = validate_token_logs(log_dir, required_methods=["minflat", "random"])
    assert any("random" in error for error in errors)
