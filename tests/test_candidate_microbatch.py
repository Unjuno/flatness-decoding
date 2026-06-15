import torch
from transformers import GPT2Config, GPT2LMHeadModel

from flatness_decoding.decoding import evaluate_candidates_next_flatness, expand_past_key_values


class DummyOutput:
    def __init__(self, logits):
        self.logits = logits


class DummyModel:
    def __init__(self, vocab_size=5):
        self.vocab_size = vocab_size
        self.batch_sizes = []
        self.seq_lens = []
        self.used_cache = []
        self.saw_attention_mask = []
        self.saw_position_ids = []

    def __call__(self, input_ids, past_key_values=None, use_cache=None, attention_mask=None, position_ids=None):
        self.batch_sizes.append(int(input_ids.shape[0]))
        self.seq_lens.append(int(input_ids.shape[1]))
        self.used_cache.append(past_key_values is not None)
        self.saw_attention_mask.append(attention_mask is not None)
        self.saw_position_ids.append(position_ids is not None)
        batch = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        logits = torch.zeros(batch, seq_len, self.vocab_size)
        last_tokens = input_ids[:, -1].float()
        for idx, token in enumerate(last_tokens):
            logits[idx, -1, :] = torch.arange(self.vocab_size).float() + token
        return DummyOutput(logits)


def tiny_gpt2() -> GPT2LMHeadModel:
    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=32,
        n_positions=16,
        n_ctx=16,
        n_embd=16,
        n_layer=2,
        n_head=2,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        bos_token_id=None,
        eos_token_id=None,
        use_cache=True,
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    return model


def test_evaluate_candidates_next_flatness_microbatches_candidate_forwards():
    model = DummyModel()
    input_ids = torch.tensor([[1, 2, 3]])
    candidate_ids = torch.tensor([4, 5, 6, 7, 8])

    out = evaluate_candidates_next_flatness(
        model=model,
        input_ids=input_ids,
        candidate_ids=candidate_ids,
        candidate_eval_batch_size=2,
    )

    assert tuple(out.shape) == (5,)
    assert model.batch_sizes == [2, 2, 1]
    assert model.seq_lens == [4, 4, 4]
    assert model.used_cache == [False, False, False]


def test_evaluate_candidates_next_flatness_uses_one_token_forwards_with_cache():
    model = DummyModel()
    input_ids = torch.tensor([[1, 2, 3]])
    candidate_ids = torch.tensor([4, 5, 6, 7, 8])
    cache = ((torch.zeros(1, 2, 3, 4), torch.zeros(1, 2, 3, 4)),)

    out = evaluate_candidates_next_flatness(
        model=model,
        input_ids=input_ids,
        candidate_ids=candidate_ids,
        candidate_eval_batch_size=2,
        past_key_values=cache,
    )

    assert tuple(out.shape) == (5,)
    assert model.batch_sizes == [2, 2, 1]
    assert model.seq_lens == [1, 1, 1]
    assert model.used_cache == [True, True, True]
    assert model.saw_attention_mask == [True, True, True]
    assert model.saw_position_ids == [True, True, True]


def test_expand_past_key_values_expands_legacy_cache_batch_dimension():
    cache = ((torch.zeros(1, 2, 3, 4), torch.ones(1, 2, 3, 4)),)
    expanded = expand_past_key_values(cache, 3)
    key, value = expanded[0]

    assert tuple(key.shape) == (3, 2, 3, 4)
    assert tuple(value.shape) == (3, 2, 3, 4)


@torch.no_grad()
def test_cached_candidate_lookahead_matches_full_prefix_lookahead_on_cpu():
    model = tiny_gpt2()
    input_ids = torch.tensor([[1, 5, 7, 9]])
    candidate_ids = torch.tensor([2, 4, 6, 8])

    full = evaluate_candidates_next_flatness(
        model=model,
        input_ids=input_ids,
        candidate_ids=candidate_ids,
        candidate_eval_batch_size=None,
        past_key_values=None,
    )

    prefix_outputs = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=True)
    cached = evaluate_candidates_next_flatness(
        model=model,
        input_ids=input_ids,
        candidate_ids=candidate_ids,
        candidate_eval_batch_size=1,
        past_key_values=prefix_outputs.past_key_values,
    )

    cached_batched = evaluate_candidates_next_flatness(
        model=model,
        input_ids=input_ids,
        candidate_ids=candidate_ids,
        candidate_eval_batch_size=2,
        past_key_values=prefix_outputs.past_key_values,
    )

    assert torch.allclose(full, cached, atol=1e-6)
    assert torch.allclose(full, cached_batched, atol=1e-6)
