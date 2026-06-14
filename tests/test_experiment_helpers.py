from scripts.run_experiment_set import select_prompt_shard, shard_suffix


def test_shard_suffix_single_shard_is_empty():
    assert shard_suffix(1, 0) == ""


def test_shard_suffix_multi_shard_is_stable():
    assert shard_suffix(4, 2) == "_shard=2-of-4"


def test_select_prompt_shard_uses_modulo_partition():
    prompts = [{"id": f"p{i}"} for i in range(6)]
    shard = select_prompt_shard(prompts, num_shards=3, shard_index=1)
    assert [row["id"] for row in shard] == ["p1", "p4"]


def test_select_prompt_shard_rejects_bad_index():
    prompts = [{"id": "p0"}]
    try:
        select_prompt_shard(prompts, num_shards=2, shard_index=2)
    except ValueError as exc:
        assert "shard_index" in str(exc)
    else:
        raise AssertionError("expected ValueError")
