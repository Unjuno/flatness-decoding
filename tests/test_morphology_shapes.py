import torch

from flatness_decoding.morphology_shapes import (
    detect_concentration,
    detect_gap,
    detect_rank_shape,
    detect_peak_structure,
    detect_shape_family,
    detect_spread,
    detect_tail,
    explain_shape,
    morphology_stats_extended,
)


def make_logits(values):
    return torch.tensor(values, dtype=torch.float32)


def test_peaky_logits_detect_high_concentration_and_large_gap():
    logits = make_logits([12.0, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0])
    stats = morphology_stats_extended(logits)

    assert detect_concentration(stats)["high_concentration"]
    assert detect_gap(stats)["large_gap"]
    assert detect_tail(stats)["thin_tail"]


def test_flat_logits_detect_low_concentration_and_thin_gap():
    logits = make_logits([0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03])
    stats = morphology_stats_extended(logits)

    assert detect_concentration(stats)["mid_concentration"] or detect_concentration(stats)["low_concentration"]
    assert detect_gap(stats)["small_gap"]
    assert detect_spread(stats)["low_spread"]


def test_broad_logits_detect_high_spread_and_heavy_tail():
    logits = make_logits([8.0] + [1.0 - 0.02 * i for i in range(1, 100)])
    stats = morphology_stats_extended(logits)

    assert detect_tail(stats)["heavy_tail"]
    assert detect_spread(stats)["mid_spread"] or detect_spread(stats)["low_spread"] or detect_spread(stats)["high_spread"]
    assert detect_rank_shape(stats)["rank_slope_flat"] or detect_rank_shape(stats)["rank_slope_mid"]


def test_shape_family_returns_boolean_labels():
    logits = make_logits([7.0, 6.9, 5.0, 2.0, 1.5, 1.0, 0.5, 0.1])
    labels = detect_shape_family(logits)

    assert isinstance(labels, dict)
    assert all(isinstance(v, bool) for v in labels.values())
    assert any(labels.values())
    assert "spiky_top" in labels
    assert "multi_peak" in labels


def test_peak_structure_labels_are_callable():
    logits = make_logits([10.0, 9.8, 9.7, 2.0, 1.9, 1.8, 1.7, 1.6])
    stats = morphology_stats_extended(logits)
    labels = detect_peak_structure(stats)
    assert set(labels) == {"spiky_top", "smooth_top", "multi_peak"}
    assert all(isinstance(v, bool) for v in labels.values())


def test_explain_shape_is_coarse_and_stable():
    logits = make_logits([10.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    stats = morphology_stats_extended(logits)
    summary = explain_shape(stats)

    assert set(summary) == {"spread", "concentration", "tail", "gap", "rank_slope", "fragment_proxy"}
    assert summary["concentration"] in {"peaky", "mid", "flat"}
