from scripts.build_prompt_subset import build_prompt_rows, clean_text, eligible_texts


class TinyDataset:
    def __iter__(self):
        yield {"text": "too short"}
        yield {"text": "alpha beta gamma delta epsilon"}
        yield {"text": "one two three four five"}
        yield {"text": "red blue green yellow purple"}


def test_clean_text_normalizes_whitespace():
    assert clean_text("hello\n  world") == "hello world"


def test_eligible_texts_filters_short_examples():
    texts = eligible_texts(TinyDataset(), "text", min_words=5)
    assert texts == [
        "alpha beta gamma delta epsilon",
        "one two three four five",
        "red blue green yellow purple",
    ]


def test_build_prompt_rows_is_deterministic_with_shuffle_seed():
    texts = [
        "alpha beta gamma delta epsilon",
        "one two three four five",
        "red blue green yellow purple",
    ]
    first = build_prompt_rows(texts, limit=2, prompt_words=3, prefix="p", shuffle=True, seed=7)
    second = build_prompt_rows(texts, limit=2, prompt_words=3, prefix="p", shuffle=True, seed=7)
    assert first == second
    assert [row["id"] for row in first] == ["p_0000", "p_0001"]
    assert all(len(row["prompt"].split()) == 3 for row in first)
