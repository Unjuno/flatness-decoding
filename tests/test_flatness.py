import torch

from flatness_decoding.flatness import flatness_std_from_logits, flatness_std_from_logprobs, selection_gain


def test_flatness_shift_invariant_for_logprobs_equivalence():
    logits = torch.tensor([1.0, 2.0, 4.0, -1.0])
    logprobs = torch.log_softmax(logits, dim=-1)
    assert torch.allclose(flatness_std_from_logits(logits), flatness_std_from_logprobs(logprobs))


def test_selection_gain_positive_when_min_below_mean():
    values = torch.tensor([3.0, 2.0, 4.0])
    gain = selection_gain(values)
    assert gain.item() > 0.0


def test_batch_flatness_shape():
    logits = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    out = flatness_std_from_logits(logits)
    assert tuple(out.shape) == (2,)
    assert out[1].item() == 0.0
