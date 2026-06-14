import json

from scripts.estimate_experiment_cost import count_prompts, estimate_cost


def test_count_prompts_respects_limit(tmp_path):
    prompt_path = tmp_path / "prompts.jsonl"
    prompt_path.write_text(
        "\n".join(json.dumps({"id": f"p{i}", "prompt": "hello"}) for i in range(5)) + "\n",
        encoding="utf-8",
    )
    cfg = {"prompts": {"path": str(prompt_path), "limit": 3}}
    assert count_prompts(cfg) == 3


def test_estimate_cost_counts_n_trial_units(tmp_path):
    prompt_path = tmp_path / "prompts.jsonl"
    prompt_path.write_text(
        "\n".join(json.dumps({"id": f"p{i}", "prompt": "hello"}) for i in range(2)) + "\n",
        encoding="utf-8",
    )
    cfg = {
        "experiment_name": "test_exp",
        "model_name": "distilgpt2",
        "prompts": {"path": str(prompt_path), "limit": 2},
        "seeds": [0],
        "generation": {"max_new_tokens": 10},
        "baselines": {"methods": ["greedy", "beam"], "num_beams": 2},
        "n_trial_methods": {"methods": ["random", "minflat"], "n_values": [1, 4]},
    }

    result = estimate_cost(cfg)

    # Baselines: prompts(2) * seeds(1) * tokens(10) * methods(2)
    # plus beam overhead: prompts(2) * seeds(1) * tokens(10) * (num_beams - 1)
    assert result["baseline_relative_units"] == 60
    # N-trial: prompts(2) * seeds(1) * tokens(10) * methods(2) * ((1+1)+(1+4))
    assert result["n_trial_relative_units"] == 280
    assert result["total_relative_units"] == 340
