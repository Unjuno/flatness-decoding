import torch

from flatness_decoding.decoding import evaluate_candidates_next_flatness


class DummyOutput:
    def __init__(self, logits):
        self.logits = logits


class DummyModel:
    def __init__(self, vocab_size=5):
        self.vocab_size = vocab_size
        self.batch_sizes = []

    def __call__(self, input_ids):
        self.batch_sizes.append(int(input_ids.shape[0]))
        batch = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        logits = torch.zeros(batch, seq_len, self.vocab_size)
        last_tokens = input_ids[:, -1].float()
        for idx, token in enumerate(last_tokens):
            logits[idx, -1, :] = torch.arange(self.vocab_size).float() + token
        return DummyOutput(logits)


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
