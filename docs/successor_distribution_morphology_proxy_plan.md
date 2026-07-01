# Successor Distribution Morphology Proxy Plan

Successor distribution morphology is a diagnostic view of the distribution that follows the actual generated token at each recorded step. The point is not to score quality. The point is to describe the *shape* of the successor distribution and ask whether that shape tracks output modes such as direct prose, generic prose, markup, prompt recycling, or token-fragment collapse.

## Why `logit_std` alone is insufficient

`logit_std` captures only one aspect of a distribution: overall spread. Two distributions can have the same spread and still differ sharply in:

- concentration near the top
- tail thickness
- rank gaps and cluster structure
- token-type composition
- prompt or context reuse
- alignment between raw probability rank and successor flatness

That is why this analysis expands the signal into a small morphology profile.

## Proxy families

### Spread

- `logit_std`
- `logit_iqr`
- `logit_mad`
- `logit_range`

These describe how wide the logit landscape is.

### Concentration

- `top1_prob`
- `top5_mass`
- `top10_mass`
- `softmax_entropy`
- `effective_vocab_size`

These describe how much of the mass is concentrated near the top.

### Tail

- `tail_mass_after_top50`
- `tail_mass_after_top100`
- `nucleus_size_p90`
- `nucleus_size_p95`
- `rank_prob_slope_top50`

These describe whether mass stays compact or spreads into the tail.

### Gap / cluster

- `gap_1_2`
- `gap_1_5`
- `gap_5_20`
- `max_gap_top50`
- `large_gap_count_top50`
- `gap_std_top50`

These help detect multi-cluster or sharply separated candidate regimes.

### Token type mass

Measured over the top candidate set with token decoding only for the necessary tokens.

- `markup_mass_top20`
- `newline_or_bullet_mass_top20`
- `quote_mass_top20`
- `digit_mass_top20`
- `subword_fragment_mass_top20`
- `html_marker_mass_top20`
- `title_marker_mass_top20`
- `choice_marker_mass_top20`

### Prompt/context overlap

- `prompt_overlap_mass_top20`
- `prompt_marker_mass_top20`
- `prompt_entity_overlap_mass_top20`
- `recent_overlap_mass_top20`

These help distinguish prompt recycling from ordinary continuation.

### Flatness-probability alignment

- `rank_at_flat_p50`
- `rank_at_flat_p75`
- `prob_mass_p25_p75`
- `spearman_flatness_probability_rank`

These ask whether flatness-selected successors sit in the high-probability region of the raw current-token distribution, or whether they live elsewhere.

## Relation to earlier experiments

This extends the earlier successor-dispersion work by moving from a single scalar to a morphology profile. The earlier analyses asked whether a candidate successor is flat or not. This plan asks what kind of distribution shape the actual generated-token successor comes from, and whether the shape correlates with observed output modes.

## Batch execution design

The script is designed to accept multiple conditions and methods, but it should load only one model at a time:

1. group requested conditions by model name
2. load one model/tokenizer
3. process all conditions for that model
4. stream rows to CSV
5. clear Python and device memory
6. unload the model
7. move to the next model

That matters because modern instruction-tuned models can consume substantial memory even for small diagnostic runs.

## Presets

- `smoke`: qwen_raw, `top_p`, 20 top-K, 2 prompts, 4 steps
- `qwen_plain`: qwen_raw and qwen_wrapped
- `gpt2_main`: GPT-2 main probe; path resolution may need explicit handling
- `gemma_wrapped10`: wrapped Gemma probe; requires `--allow-gemma`
- `modern_compare`: Gemma wrapped plus Qwen wrapped/chat if available
- `morphology_20x20_top20`: prompt_count=20, repeat_count=20, top_k=20, max_steps=20

`prompt_count`, `repeat_count`, and `top_k` are separate axes:

- `prompt_count`: number of prompts analyzed
- `repeat_count` / `seed_count`: stochastic replicates of the same prompt/method slice
- `top_k`: number of raw-logit candidates used to build the morphology profile

The main morphology columns are computed from the successor distribution after the actual generated token. Candidate-set alignment diagnostics are reported separately and are based on the raw current-logit top-K candidates and their successor flatness scores.

## Memory and runtime caveats

Use `torch.inference_mode()`, stream rows, and avoid retaining full logits or intermediate tensors. Always clear CPU/GPU/MPS caches between flushes. Keep `candidate_batch_size` small by default.

## Resume behavior

The script should skip already completed rows when `--resume` is used. The completion key should include:

- `condition`
- `method`
- `prompt_id`
- `seed`
- `repeat_index`
- `step`

For the 20x20 preset, actual repeated generation outputs are required to observe repeat_count > 1. If the stored generations only contain a single seed/repeat, the report will show the observed count and a warning instead of fabricating repetitions.

## Statistical comparison plan

If the goal is to compare output modes statistically, use a repeat-backed generation run first and then analyze the resulting generations:

- `prompt_count = 20`
- `repeat_count = 20`
- `seed_count = 20`
- fixed prompt set from the dataset-backed prompt file

That gives 20 repeated generations per prompt, which is the right unit for comparing mode rates and morphology-band differences across conditions. The analysis then groups by prompt and repeat, computes feature summaries, and can bootstrap over prompts as the primary unit.

The active morphology-stratified probe is separate: it dumps the full candidate pool and also produces regime-stratified samples. It is useful for selecting candidate regimes, but it does not itself synthesize stochastic repeats.

### Repeat-backed generation command

For the actual 20-repeat experiment, use the existing sweep generator with a fixed prompt subset and a seed grid. For example:

```bash
python scripts/run_sweep_n.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --prompts prompts/writingprompts_fixed_100.jsonl \
  --limit 20 \
  --methods random minflat maxflat centerflat_q50 centerflat_q75 highprob_centerflat \
  --n-values 1 4 8 16 \
  --repeat-count 20 \
  --seed-start 0 \
  --resume
```

That gives 20 repeat-backed generations per prompt. You can then run the morphology analysis script over the resulting generation directory.
If the generation job is interrupted, re-run it with `--resume`; completed prompt/seed groups will be skipped.

### One-command launcher

For a repeat-backed morphology run that generates and then analyzes condition-by-condition, use:

```bash
python scripts/run_morphology_repeat_grid.py \
  --conditions qwen_raw \
  --prompt-limit 20 \
  --repeat-count 20 \
  --seed-start 0 \
  --execute \
  --resume
```

Add more conditions as needed. The launcher processes one model at a time, writes repeat-backed generation outputs, and can optionally run the successor morphology analysis immediately after each condition.

## Recommended commands

Dry-run:

```bash
python scripts/analyze_successor_distribution_morphology.py --preset smoke --dry-run
```

Estimate only:

```bash
python scripts/analyze_successor_distribution_morphology.py --preset smoke --estimate-only
```

Execute the small smoke slice later if needed:

```bash
python scripts/analyze_successor_distribution_morphology.py --preset smoke --execute
```

Dry-run for the 20x20 morphology design:

```bash
python scripts/analyze_successor_distribution_morphology.py \
  --preset morphology_20x20_top20 \
  --conditions qwen_raw \
  --methods top_p \
  --dry-run
```

Gemma wrapped10 dry-run:

```bash
python scripts/analyze_successor_distribution_morphology.py \
  --preset gemma_wrapped10 \
  --allow-gemma \
  --dry-run
```
