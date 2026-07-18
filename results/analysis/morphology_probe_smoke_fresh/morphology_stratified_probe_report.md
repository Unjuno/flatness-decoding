# Morphology-Stratified Successor Probing

This is a diagnostic probe, not a quality metric.

The script builds a candidate pool for each prompt/context, computes candidate-induced successor distribution morphology, stratifies candidates into morphology regimes, and optionally generates short continuations from selected candidates.

- conditions: qwen_raw
- candidate_pool_mode: raw_topk
- candidate_pool_size: 50
- samples_per_regime_requested: 2
- samples_per_regime_observed: 32
- total_regime_count: 17
- global_selection: true
- max_steps: 1
- generate_continuations: False

## Regime shortages
- concentration:high_concentration shortage=0
- concentration:low_concentration shortage=0
- concentration:mid_concentration shortage=0
- gap:large_gap shortage=0
- gap:mid_gap shortage=0
- gap:small_gap shortage=0
- spread:high_spread shortage=0
- spread:low_spread shortage=0
- spread:mid_spread shortage=0
- tail:heavy_tail shortage=0
- tail:mid_tail shortage=0
- tail:thin_tail shortage=0
- token_type:fragment_heavy shortage=0
- token_type:list_heavy shortage=0
- token_type:markup_heavy shortage=2
- token_type:prompt_overlap_heavy shortage=0
- token_type:recent_overlap_heavy shortage=0

## Notes

Selection is global across prompts within each condition, and it prefers one candidate per prompt per regime before reusing the same prompt.
This active probe currently uses step 0 / the initial prompt context only.
Morphology-conditioned candidate selection is diagnostic, not a claim of quality improvement.
Token-type regimes are independent and may overlap.
