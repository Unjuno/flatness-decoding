from flatness_decoding.metrics import analysis_text_from_row, distinct_n, repetition_rate


def test_distinct_n_basic():
    texts = ["a b c", "a b d"]
    value = distinct_n(texts, 1)
    assert 0.0 < value <= 1.0


def test_repetition_rate_detects_repetition():
    texts = ["a b a b a b"]
    assert repetition_rate(texts, 2) > 0.0


def test_empty_metrics_are_zero():
    assert distinct_n([], 2) == 0.0
    assert repetition_rate([], 4) == 0.0


def test_analysis_text_prefers_continuation_text():
    row = {"prompt": "Hello", "text": "Hello world", "continuation_text": "world"}
    assert analysis_text_from_row(row) == "world"


def test_analysis_text_strips_prompt_prefix_for_legacy_rows():
    row = {"prompt": "Hello", "text": "Hello world"}
    assert analysis_text_from_row(row) == "world"


def test_analysis_text_strips_prompt_prefix_with_leading_space():
    row = {"prompt": "Hello", "text": "  Hello world"}
    assert analysis_text_from_row(row) == "world"
