import torch

from scripts.evaluate_ppl import conditional_continuation_nll, full_text_for_ppl


class DummyTokenizer:
    def __call__(self, text, return_tensors="pt", truncation=False, max_length=None, add_special_tokens=False):
        del return_tensors, add_special_tokens
        tokens = text.split()
        ids = list(range(1, len(tokens) + 1))
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
        }


class DummyOutput:
    def __init__(self, loss):
        self.loss = loss


class DummyModel:
    def __init__(self, loss_value=2.0):
        self.loss_value = loss_value
        self.last_labels = None

    def __call__(self, input_ids, attention_mask=None, labels=None):
        del input_ids, attention_mask
        self.last_labels = labels.detach().cpu().clone()
        return DummyOutput(torch.tensor(self.loss_value))


def test_conditional_continuation_nll_masks_prompt_tokens():
    model = DummyModel(loss_value=2.0)
    tokenizer = DummyTokenizer()

    nll, count = conditional_continuation_nll(
        model=model,
        tokenizer=tokenizer,
        prompt="A B",
        full_text="A B C",
        device=torch.device("cpu"),
        max_length=16,
    )

    assert count == 1
    assert nll == 2.0
    assert model.last_labels.tolist()[0][:2] == [-100, -100]
    assert model.last_labels.tolist()[0][2] != -100


def test_conditional_continuation_nll_returns_zero_when_no_continuation_token_is_left():
    model = DummyModel(loss_value=2.0)
    tokenizer = DummyTokenizer()

    nll, count = conditional_continuation_nll(
        model=model,
        tokenizer=tokenizer,
        prompt="A B",
        full_text="A B",
        device=torch.device("cpu"),
        max_length=16,
    )

    assert count == 0
    assert nll == 0.0


def test_full_text_for_ppl_prefers_full_decoded_text():
    row = {"prompt": "Hello", "text": "Hello world", "continuation_text": "world"}
    assert full_text_for_ppl(row) == "Hello world"


def test_full_text_for_ppl_falls_back_to_prompt_plus_continuation_with_space():
    row = {"prompt": "Hello", "continuation_text": "world"}
    assert full_text_for_ppl(row) == "Hello world"


def test_full_text_for_ppl_preserves_continuation_leading_space():
    row = {"prompt": "Hello", "continuation_text": " world"}
    assert full_text_for_ppl(row) == "Hello world"
