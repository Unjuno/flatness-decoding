# Successor Distribution Morphology

These are diagnostic morphology proxies, not quality metrics.

Main morphology columns are computed from the successor distribution after the actual generated token.
Current-logit candidate alignment diagnostics are kept separate in the `candidate_*` fields.

## What the proxies cover

- spread: `logit_std`, `logit_iqr`, `logit_mad`, `logit_range`
- concentration: `top1_prob`, `top5_mass`, `top10_mass`, `softmax_entropy`, `effective_vocab_size`
- tail: `tail_mass_after_top50`, `tail_mass_after_top100`, `nucleus_size_p90`, `nucleus_size_p95`, `rank_prob_slope_top50`
- gap / cluster: `gap_1_2`, `gap_1_5`, `gap_5_20`, `max_gap_top50`, `large_gap_count_top50`, `gap_std_top50`
- token type and recycling: markup, bullet, quote, digit, HTML, title, choice, prompt/context overlap, recent overlap
- flatness alignment: `rank_at_flat_p50`, `rank_at_flat_p75`, `prob_mass_p25_p75`, `spearman_flatness_probability_rank`

## Mode mapping

- direct narrative prose
- generic atmospheric prose
- title or character sheet
- commentary or analysis
- quiz or options
- markup or HTML
- prompt marker recycling
- token fragment collapse
- repetition or loop
- semantic drift possible

## Execution model

This script loads one model at a time, processes the requested conditions for that model, streams rows to CSV, and clears memory before moving on.

- feature rows collected: 0
- requested_repeat_count: 20
- observed_seed_count: 0
- observed_repeat_count: 0
- by-condition summary rows: 0
- by-method summary rows: 0
- by-prompt summary rows: 0
- by-mode summary rows: 0

## Presets

- `smoke`: qwen_raw, top_p, 20 top-K, 2 prompts, 4 steps
- `qwen_plain`: qwen_raw and qwen_wrapped, 10 prompts, 16 steps
- `gpt2_main`: GPT-2 main probe; GPT-2 path must be resolved explicitly
- `gemma_wrapped10`: Gemma wrapped10; requires `--allow-gemma`
- `modern_compare`: Gemma wrapped plus Qwen wrapped/chat when available
- `morphology_20x20_top20`: 20 prompts × 20 repeats × top20 morphology slice

## Notes

Prompt_count, repeat_count, and top_k are distinct. `prompt_count` is the number of prompts; `repeat_count`/`seed_count` are stochastic replicates; `top_k` is the candidate set used for morphology.
For the 20x20 preset, actual repeated generation outputs are required to observe repeat_count > 1; otherwise the report will show the available observed repeat count only.
