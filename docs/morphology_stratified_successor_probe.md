# Morphology-Stratified Successor Probing

This note describes the active probe that complements the existing successor distribution morphology analysis.

## Purpose

The probe is diagnostic. It does not claim quality improvement and it does not claim robust format control.
Its purpose is to test whether candidate-induced successor distribution morphology differs across morphology regimes.

## What it does

For each prompt/context:

1. Reconstruct the prompt context.
2. Compute current logits.
3. Build a candidate pool with `raw_topk` or `sampled`.
4. For each candidate, append the candidate to the context and compute successor logits.
5. Measure successor distribution morphology on those successor logits.
6. Stratify candidates into morphology regimes.
7. Select up to `samples_per_regime` candidates per regime globally across prompts, preferring one candidate per prompt before reusing the same prompt.
8. Optionally generate short continuations from the selected candidates.

## Morphology regimes

The probe uses the same morphology families as the analysis script, but to *select* candidates:

- `spread`: `low_spread`, `mid_spread`, `high_spread`
- `concentration`: `low_concentration`, `mid_concentration`, `high_concentration`
- `tail`: `thin_tail`, `mid_tail`, `heavy_tail`
- `gap`: `small_gap`, `mid_gap`, `large_gap`
- `token type`: `markup_heavy`, `list_heavy`, `fragment_heavy`, `prompt_overlap_heavy`, `recent_overlap_heavy`

The selection step is morphology-conditioned candidate selection. It is still diagnostic, not a claim of quality improvement.
Token-type regimes are independent and may overlap on the same candidate.
Selection is global across prompts within each condition, not per prompt.

## Candidate-pool modes

- `raw_topk`: use the raw top logits from the current context.
- `sampled`: sample candidates from the current logits at the configured temperature.

## Regime assignment

For scalar continuous features:

- low: bottom quantile
- mid: middle band around the median
- high: top quantile

Default thresholds:

- low: `<= q25`
- mid: `q40` to `q60`
- high: `>= q75`

For token-type heavy regimes, candidates are ranked by the relevant token-type mass and then selected.

## Outputs

- `morphology_stratified_candidate_morphology.csv` is the exhaustive candidate dump. It records every candidate in the pool for every selected prompt.
- `morphology_stratified_selected_candidates.csv` is the regime-stratified subset used for comparison and diagnostics.
- `results/analysis/morphology_stratified_candidate_morphology.csv`
- `results/analysis/morphology_stratified_selected_candidates.csv`
- `results/analysis/morphology_stratified_probe_generations.jsonl`
- `results/analysis/morphology_stratified_probe_summary.csv`
- `results/analysis/morphology_stratified_probe_report.md`

## Presets

- `morphology_probe_smoke`
  - `qwen_raw`
  - `raw_topk`
  - candidate pool size 50
  - `samples_per_regime=2`
  - `max_prompts=2`
  - `max_steps=1`
  - no continuation generation

- `morphology_probe_20`
  - `qwen_raw` by default
  - `raw_topk`
  - candidate pool size 100
  - `samples_per_regime=20`
  - `max_prompts=20`
  - `max_steps=1`
  - optional continuation generation

## Safety and limits

- Dry-run by default.
- `--execute` is required for model execution.
- `--generate-continuations` is required for continuation generation.
- Gemma requires `--allow-gemma`.
- Candidate pools above 200 are refused unless `--confirm-large-run` is provided.
- One model is loaded at a time.
- The exhaustive candidate dump is `morphology_stratified_candidate_morphology.csv`; that is the artifact to compare against the older flatness-style “full dump” workflow.
- `samples_per_regime` means the total selected candidates per regime across prompts, not per prompt.
- Selection prefers one candidate per prompt before reusing the same prompt.
- This active probe currently uses step 0 / the initial prompt context only.
- If `max_steps > 1` is requested, the script errors for now.
- The total regime count is 17: 3 spread + 3 concentration + 3 tail + 3 gap + 5 token-type regimes.
- `morphology_stratified_candidate_morphology.csv` and `morphology_stratified_selected_candidates.csv` append across runs; use `--resume` for repeated runs.
- `morphology_stratified_probe_summary.csv` and `morphology_stratified_probe_report.md` are overwritten each run from the on-disk selected rows when available.

## Validation

Recommended commands:

```bash
python -m py_compile scripts/run_morphology_stratified_probe.py
python scripts/run_morphology_stratified_probe.py --help
python scripts/run_morphology_stratified_probe.py --preset morphology_probe_smoke --dry-run
python scripts/run_morphology_stratified_probe.py --preset morphology_probe_smoke --estimate-only
```

## Caveat

This is a diagnostic probe. It does not establish quality improvement or robust format control.
