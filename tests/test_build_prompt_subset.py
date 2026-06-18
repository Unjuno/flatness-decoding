from scripts.build_prompt_subset import build_prompt_rows, clean_text, eligible_rows, make_prompt


class TinyDataset:
    def __iter__(self):
        yield {"text": "too short"}
        yield {"text": "alpha beta gamma delta epsilon", "reference": "alpha beta"}
        yield {"text": "one two three four five", "reference": "one two"}
        yield {"text": "red blue green yellow purple", "reference": "red blue"}


def test_clean_text_normalizes_whitespace():
    assert clean_text("hello\n  world") == "hello world"


def test_make_prompt_truncates_words():
    assert make_prompt("one two three four", 3) == "one two three"


def test_eligible_rows_filters_short_examples_and_adds_provenance():
    rows = eligible_rows(
        TinyDataset(),
        dataset_name="wikitext",
        split="validation",
        prompt_field=None,
        reference_field="reference",
        text_field="text",
        min_prompt_words=1,
        min_words=5,
        prompt_words=3,
    )
    assert len(rows) == 3
    assert rows[0]["source_dataset"] == "wikitext"
    assert rows[0]["source_config"] is None
    assert rows[0]["source_split"] == "validation"
    assert rows[0]["source_row_index"] == 1
    assert rows[0]["text_field"] == "text"
    assert rows[0]["reference_field"] == "reference"
    assert rows[0]["reference"] == "alpha beta"


def test_build_prompt_rows_is_deterministic_with_shuffle_seed():
    candidates = [
        {"prompt": "alpha beta gamma", "source_dataset": "wikitext", "source_config": None, "source_split": "validation", "source_row_index": 1, "text_field": "text"},
        {"prompt": "one two three", "source_dataset": "wikitext", "source_config": None, "source_split": "validation", "source_row_index": 2, "text_field": "text"},
        {"prompt": "red blue green", "source_dataset": "wikitext", "source_config": None, "source_split": "validation", "source_row_index": 3, "text_field": "text"},
    ]
    first = build_prompt_rows(candidates, limit=2, prefix="p", shuffle=True, seed=7)
    second = build_prompt_rows(candidates, limit=2, prefix="p", shuffle=True, seed=7)
    assert first == second
    assert [row["id"] for row in first] == ["p_000000", "p_000001"]
    assert all(len(row["prompt"].split()) == 3 for row in first)
