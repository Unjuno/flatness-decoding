from flatness_decoding.implementation import IMPLEMENTATION_PROFILE
from flatness_decoding.io_utils import write_jsonl
from flatness_decoding.resume import baseline_output_complete, ntrial_output_complete


def test_baseline_output_complete_accepts_complete_file(tmp_path):
    path = tmp_path / "baseline.jsonl"
    write_jsonl(
        path,
        [
            {
                "implementation_profile": IMPLEMENTATION_PROFILE,
                "method": "greedy",
                "method_family": "baseline",
                "inference_mode": False,
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
        inference_mode=False,
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
        inference_mode=False,
    )


def test_baseline_output_complete_rejects_old_implementation_profile(tmp_path):
    path = tmp_path / "baseline.jsonl"
    write_jsonl(
        path,
        [
            {
                "implementation_profile": "old-profile",
                "method": "greedy",
                "method_family": "baseline",
                "inference_mode": False,
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

    assert not baseline_output_complete(
        path,
        expected_rows=1,
        method="greedy",
        seed=0,
        shard_index=0,
        num_shards=2,
        inference_mode=False,
    )


def test_ntrial_output_complete_checks_token_log_row_count(tmp_path):
    gen_path = tmp_path / "ntrial.jsonl"
    token_path = tmp_path / "ntrial.tokenlog.jsonl"
    write_jsonl(
        gen_path,
        [
            {
                "implementation_profile": IMPLEMENTATION_PROFILE,
                "method": "minflat",
                "method_family": "n_trial",
                "n_trials": 4,
                "candidate_eval_batch_size": 2,
                "use_kv_cache_lookahead": False,
                "proposal_sampling_device": "cpu",
                "selection_device": "cpu",
                "inference_mode": False,
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
                "implementation_profile": IMPLEMENTATION_PROFILE,
                "method": "minflat",
                "n_trials": 4,
                "candidate_eval_batch_size": 2,
                "use_kv_cache_lookahead": False,
                "inference_mode": False,
                "seed": 0,
                "shard_index": 0,
                "num_shards": 2,
                "candidate_ids": [1, 2, 3, 4],
                "candidate_flatness": [1.0, 2.0, 3.0, 4.0],
                "chosen_id": 1,
            },
            {
                "implementation_profile": IMPLEMENTATION_PROFILE,
                "method": "minflat",
                "n_trials": 4,
                "candidate_eval_batch_size": 2,
                "use_kv_cache_lookahead": False,
                "inference_mode": False,
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
        candidate_eval_batch_size=2,
        inference_mode=False,
        use_kv_cache_lookahead=False,
    )


def test_ntrial_output_complete_rejects_missing_token_rows(tmp_path):
    gen_path = tmp_path / "ntrial.jsonl"
    token_path = tmp_path / "ntrial.tokenlog.jsonl"
    write_jsonl(
        gen_path,
        [
            {
                "implementation_profile": IMPLEMENTATION_PROFILE,
                "method": "minflat",
                "method_family": "n_trial",
                "n_trials": 4,
                "candidate_eval_batch_size": 2,
                "use_kv_cache_lookahead": False,
                "proposal_sampling_device": "cpu",
                "selection_device": "cpu",
                "inference_mode": False,
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
        candidate_eval_batch_size=2,
        inference_mode=False,
        use_kv_cache_lookahead=False,
    )


def test_ntrial_output_complete_rejects_different_batch_size(tmp_path):
    gen_path = tmp_path / "ntrial.jsonl"
    token_path = tmp_path / "ntrial.tokenlog.jsonl"
    write_jsonl(
        gen_path,
        [
            {
                "implementation_profile": IMPLEMENTATION_PROFILE,
                "method": "minflat",
                "method_family": "n_trial",
                "n_trials": 4,
                "candidate_eval_batch_size": 1,
                "use_kv_cache_lookahead": False,
                "proposal_sampling_device": "cpu",
                "selection_device": "cpu",
                "inference_mode": False,
                "seed": 0,
                "shard_index": 0,
                "num_shards": 2,
                "prompt_id": "p0",
                "prompt_index": 0,
                "text": "abc",
                "continuation_text": "bc",
                "num_output_tokens": 1,
            }
        ],
    )
    write_jsonl(
        token_path,
        [
            {
                "implementation_profile": IMPLEMENTATION_PROFILE,
                "method": "minflat",
                "n_trials": 4,
                "candidate_eval_batch_size": 1,
                "use_kv_cache_lookahead": False,
                "inference_mode": False,
                "seed": 0,
                "shard_index": 0,
                "num_shards": 2,
                "candidate_ids": [1, 2, 3, 4],
                "candidate_flatness": [1.0, 2.0, 3.0, 4.0],
                "chosen_id": 1,
            }
        ],
    )

    assert not ntrial_output_complete(
        gen_path,
        token_path,
        expected_rows=1,
        method="minflat",
        n_trials=4,
        seed=0,
        shard_index=0,
        num_shards=2,
        candidate_eval_batch_size=2,
        inference_mode=False,
        use_kv_cache_lookahead=False,
    )
