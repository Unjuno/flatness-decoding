from flatness_decoding.io_utils import write_jsonl
from flatness_decoding.resume import baseline_output_complete, ntrial_output_complete


def test_baseline_output_complete_accepts_complete_file(tmp_path):
    path = tmp_path / "baseline.jsonl"
    write_jsonl(
        path,
        [
            {
                "method": "greedy",
                "method_family": "baseline",
                "seed": 0,
                "shard_index": 0,
                "num_shards": 2,
                "prompt_id": "p0",
                "prompt_index": 0,
                "text": "abc",
                "continuation_text": "bc",
                "num_output_tokens": 2,
            }
        ],
    )

    assert baseline_output_complete(
        path,
        expected_rows=1,
        method="greedy",
        seed=0,
        shard_index=0,
        num_shards=2,
    )


def test_baseline_output_complete_rejects_partial_file(tmp_path):
    path = tmp_path / "baseline.jsonl"
    write_jsonl(path, [])

    assert not baseline_output_complete(
        path,
        expected_rows=1,
        method="greedy",
        seed=0,
        shard_index=0,
        num_shards=2,
    )


def test_ntrial_output_complete_checks_token_log_row_count(tmp_path):
    gen_path = tmp_path / "ntrial.jsonl"
    token_path = tmp_path / "ntrial.tokenlog.jsonl"
    write_jsonl(
        gen_path,
        [
            {
                "method": "minflat",
                "method_family": "n_trial",
                "n_trials": 4,
                "seed": 0,
                "shard_index": 0,
                "num_shards": 2,
                "prompt_id": "p0",
                "prompt_index": 0,
                "text": "abc",
                "continuation_text": "bc",
                "num_output_tokens": 2,
            }
        ],
    )
    write_jsonl(
        token_path,
        [
            {
                "method": "minflat",
                "n_trials": 4,
                "seed": 0,
                "shard_index": 0,
                "num_shards": 2,
                "candidate_ids": [1, 2, 3, 4],
                "candidate_flatness": [1.0, 2.0, 3.0, 4.0],
                "chosen_id": 1,
            },
            {
                "method": "minflat",
                "n_trials": 4,
                "seed": 0,
                "shard_index": 0,
                "num_shards": 2,
                "candidate_ids": [1, 2, 3, 4],
                "candidate_flatness": [1.0, 2.0, 3.0, 4.0],
                "chosen_id": 2,
            },
        ],
    )

    assert ntrial_output_complete(
        gen_path,
        token_path,
        expected_rows=1,
        method="minflat",
        n_trials=4,
        seed=0,
        shard_index=0,
        num_shards=2,
    )


def test_ntrial_output_complete_rejects_missing_token_rows(tmp_path):
    gen_path = tmp_path / "ntrial.jsonl"
    token_path = tmp_path / "ntrial.tokenlog.jsonl"
    write_jsonl(
        gen_path,
        [
            {
                "method": "minflat",
                "method_family": "n_trial",
                "n_trials": 4,
                "seed": 0,
                "shard_index": 0,
                "num_shards": 2,
                "prompt_id": "p0",
                "prompt_index": 0,
                "text": "abc",
                "continuation_text": "bc",
                "num_output_tokens": 2,
            }
        ],
    )
    write_jsonl(token_path, [])

    assert not ntrial_output_complete(
        gen_path,
        token_path,
        expected_rows=1,
        method="minflat",
        n_trials=4,
        seed=0,
        shard_index=0,
        num_shards=2,
    )
