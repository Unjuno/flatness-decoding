from flatness_decoding.decoding import banned_candidate_tokens


class DummyTokenizer:
    def __init__(self, pad_token_id, eos_token_id):
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id


def test_banned_candidate_tokens_does_not_ban_eos_when_pad_equals_eos():
    tokenizer = DummyTokenizer(pad_token_id=50256, eos_token_id=50256)
    assert banned_candidate_tokens(tokenizer) == set()


def test_banned_candidate_tokens_bans_distinct_pad():
    tokenizer = DummyTokenizer(pad_token_id=0, eos_token_id=2)
    assert banned_candidate_tokens(tokenizer) == {0}
