import torch

from flatness_decoding.sampling import sample_candidates_from_logits


def test_sample_candidates_respects_banned_token_ids():
    logits = torch.tensor([100.0, 0.0, 0.0])
    samples = sample_candidates_from_logits(
        logits=logits,
        n=20,
        temperature=1.0,
        replacement=True,
        banned_token_ids={0},
    )
    assert 0 not in samples.tolist()


def test_sample_candidates_shape():
    logits = torch.ones(5)
    samples = sample_candidates_from_logits(logits=logits, n=3, temperature=1.0)
    assert samples.shape == (3,)


def test_sample_candidates_returns_cpu_tensor():
    logits = torch.ones(5)
    samples = sample_candidates_from_logits(logits=logits, n=3, temperature=1.0)
    assert samples.device.type == "cpu"


def test_sample_candidates_rejects_nonpositive_temperature():
    logits = torch.ones(5)
    try:
        sample_candidates_from_logits(logits=logits, n=3, temperature=0.0)
    except ValueError as exc:
        assert "temperature" in str(exc)
    else:
        raise AssertionError("expected ValueError")
