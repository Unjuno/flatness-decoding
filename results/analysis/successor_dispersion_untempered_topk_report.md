# Untempered Successor Dispersion Report

This report is generated from raw current-logit top-K candidates and is intended to test whether the successor-dispersion median remains tied to the high-probability region.

## Questions

1. Does successor-dispersion p50 overlap with high-probability current-token regions?
2. Is p50 low probability even when candidates are selected from raw top-K logits?
3. Does Gemma place high-probability candidates closer to p75 than p50?
4. Does Qwen show a weaker or different overlap pattern?
5. Was earlier q50 behavior mainly caused by proposal-temperature sampling?
6. Are min/max extremes separated by large flatness gaps, or is the distribution narrow?
7. Does p25-p75 probability mass differ across GPT-2, Gemma, and Qwen?

## Status

- distribution rows: 8
- condition summary rows: 1

If no execute run has been performed, these fields remain placeholders only.
