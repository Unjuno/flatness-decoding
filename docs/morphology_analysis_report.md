# Successor Logit Morphology Analysis Report

## Scope

This report consolidates the morphology analysis for three models:

- GPT-2
- Gemma 3 4B-it (raw prompt)
- Qwen 2.5 1.5B-Instruct (raw prompt)

The probe uses 20 WritingPrompts prompts, a top-100 current-token candidate pool, and step 0 successor-state measurements. The analysis is diagnostic: it does not claim that a morphology is a quality metric.

A supplementary Qwen raw top-200 probe was also completed for the previously under-covered token-content regimes. It is kept separate from the main top-100 comparison because changing the candidate-pool size changes regime exposure.

## What is being measured

The project has two related but distinct notions:

1. `successor logit dispersion`: standard deviation of the successor raw logits.
2. `shape family`: a vector of morphology labels derived from spread, concentration, tail, gap, rank slope, and peak structure.

The second notion contains 18 labels:

| Family | Labels |
|---|---|
| Spread | `low_spread`, `mid_spread`, `high_spread` |
| Concentration | `low_concentration`, `mid_concentration`, `high_concentration` |
| Tail | `thin_tail`, `mid_tail`, `heavy_tail` |
| Gap | `small_gap`, `mid_gap`, `large_gap` |
| Rank slope | `rank_slope_flat`, `rank_slope_mid`, `rank_slope_steep` |
| Peak structure | `spiky_top`, `smooth_top`, `multi_peak` |

The separate token-content diagnostics (`markup_heavy`, `list_heavy`, `fragment_heavy`, `prompt_overlap_heavy`, `recent_overlap_heavy`) are not part of these 18 topology labels. They describe what kinds of tokens occupy the candidate distribution.

## Main findings

### 1. The 18 labels are not 18 independent shapes

They are overlapping projections of a multi-dimensional state. `high_concentration`, `large_gap`, `thin_tail`, and `spiky_top` often identify the same sharp successor state. `smooth_top` and `small_gap` often identify a competing-candidate state.

The most useful conceptual coordinates are:

```text
concentration × tail × gap × rank-curve slope × peak topology
```

`logit_std` alone is too coarse.

### 2. Model differences are structural, not merely noise

Gemma raw contains many sharply concentrated successor states. Qwen raw more often retains a broad tail and lower top-1 probability. GPT-2 occupies an intermediate regime but shows strong content-token effects from its tokenizer.

Raw logit standard deviation has weak cross-model rank agreement, while top-1 probability, tail mass, and gap have much stronger rank agreement. Cross-model comparisons should therefore use model-local percentiles or z-scores rather than raw logit values.

### 3. Rank slope and peak topology add real information

`rank_slope_flat` measures slow decay among the upper-ranked probabilities; it is not identical to `heavy_tail`, which measures mass outside the top 50. `multi_peak` identifies multiple discontinuities within the upper rank curve and cannot be reconstructed from top-1 alone.

Some shape/model combinations have zero observations in the top-100 pool. This is itself evidence: under the current model, prompt, and candidate-pool conditions, the corresponding successor state is rare or inaccessible.

### 4. Content differences exist, but we initially under-analyzed them

Candidate-token analysis shows differences in word fragments, whitespace, punctuation, and markup:

- Gemma `rank_slope_steep` is mostly whitespace-like candidates, but has only a few observations.
- GPT-2 `spiky_top` contains a substantial whitespace/newline component.
- Qwen `rank_slope_flat` contains comparatively more brackets, markup, and other format-like tokens.
- `multi_peak` often mixes ordinary words with punctuation or format tokens.

These are content-form differences, not yet semantic or story-quality differences. The morphology probe did not save continuations from every selected candidate, so semantic claims require a follow-up generation stage.

The supplementary Qwen top-200 run removes the `markup_heavy` coverage shortage under the expanded-pool condition. It should be read as a sensitivity/completion probe, not as a directly pooled estimate with the top-100 results.

### 5. One diagnostic was invalid in the old data

The old step-0 probe passed the prompt as `recent_text`, making `prompt_overlap` and `recent_overlap` identical. The probe implementation now uses an empty recent context at step 0. Existing CSVs retain the old value and must be regenerated before using `recent_overlap` as an independent feature.

## Direct interpretation

My current interpretation is that successor morphology is a state-space coordinate, not a single scalar quality axis.

- Sharp / spiky states may be stable but overly deterministic, format-prone, or repetitive.
- Smooth / competing states preserve alternatives but are sensitive to sampling and candidate choice.
- Heavy-tail states offer exploration but may be weakly grounded.
- Multi-peak states are especially interesting because they represent structural ambiguity rather than simply “more” or “less” flatness.
- Token-content labels reveal whether the state is associated with narrative words, whitespace, punctuation, or formatting tokens.

The most promising decoder design is therefore a model-local multi-objective target band, not `minflat` or `maxflat`:

```text
moderate concentration
moderate tail
small-to-mid gap
avoid undesirable peak topology
avoid markup/fragment states when narrative prose is desired
```

## What should be analyzed next

1. Generate continuations from the same selected candidates and compare semantic mode, prompt adherence, repetition, and formatting.
2. Use paired prompt-level comparisons: each prompt should contribute one candidate to each target shape where possible.
3. Re-run the corrected probe for `recent_overlap`.
4. Compare model-local percentile coordinates, not raw logit values.
5. Test whether `multi_peak` predicts semantic drift or merely token-level ambiguity.
6. Build a two-axis selector: concentration/tail target plus a peak/content penalty.
7. Add bootstrap confidence intervals and report zero-coverage shape/model combinations explicitly.

## 既存出力ログの個別突合（追加）

`outputs/generations/*shapes_20/` と `outputs/token_logs/*shapes_20/` を、形状 method・model・prompt_id で突合する個別解析を追加した。各形状について、20 prompt × 3 model の continuation と各120 stepの token log を保存済みログから集計している。再生成は行っていない。

個別成果物は `results/analysis/shape_<shape>_output_analysis.md`、`shape_<shape>_output_rows.csv`、`shape_<shape>_token_rows.csv` として出力される。再現コマンドは次の通り。

```bash
for shape in low_spread mid_spread high_spread low_concentration mid_concentration high_concentration thin_tail mid_tail heavy_tail small_gap mid_gap large_gap rank_slope_flat rank_slope_mid rank_slope_steep spiky_top smooth_top multi_peak; do
  PYTHONPATH=src .venv/bin/python scripts/analyze_shape_output_one.py --shape "$shape"
done
```

最初の `low_spread` では、同じ形状名でも出力挙動はモデルごとに分岐した。GPT-2 は比較的通常の続き生成、Gemma は反復・縮退、Qwen は非ASCII／形式トークンの混入が目立つ。したがって `low_spread` は単一の意味クラスではなく、モデル内の出力挙動と併記して分類する必要がある。

## Reproduction

```bash
PYTHONPATH=src .venv/bin/python scripts/analyze_morphology_18_shapes.py
PYTHONPATH=src .venv/bin/python scripts/analyze_morphology_content.py
PYTHONPATH=src .venv/bin/python scripts/analyze_morphology_candidate_pool.py
PYTHONPATH=src .venv/bin/python scripts/analyze_morphology_deep.py
PYTHONPATH=src .venv/bin/python scripts/analyze_morphology_uncertainty.py
PYTHONPATH=src .venv/bin/python scripts/analyze_shape_output_summary.py
PYTHONPATH=src .venv/bin/python scripts/analyze_prompt_framing_outputs.py
PYTHONPATH=src .venv/bin/python scripts/classify_shape_output_modes.py
PYTHONPATH=src .venv/bin/python scripts/analyze_shape_token_output_link.py
PYTHONPATH=src .venv/bin/python scripts/classify_output_from_token_features.py
PYTHONPATH=src .venv/bin/python scripts/evaluate_shape_holdout_classifier.py
PYTHONPATH=src .venv/bin/python scripts/analyze_shape_paired_effects.py
PYTHONPATH=src .venv/bin/python scripts/classify_cross_model_shape_effects.py
PYTHONPATH=src .venv/bin/python scripts/analyze_shape_paired_uncertainty.py
MPLCONFIGDIR=/tmp/flatness-mpl .venv/bin/python scripts/plot_morphology_diagnostics.py
MPLCONFIGDIR=/tmp/flatness-mpl .venv/bin/python scripts/plot_morphology_content_coverage.py
```

## Linked artifacts

- [18-shape summary](../results/analysis/morphology_18_shape_analysis.md)
- [content analysis](../results/analysis/morphology_content_analysis.md)
- [candidate-pool analysis](../results/analysis/morphology_candidate_pool_analysis.md)
- [deep cross-model analysis](../results/analysis/morphology_deep_analysis.md)
- [uncertainty analysis](../results/analysis/morphology_uncertainty_analysis.md)
- [interpretation and predictions](../results/analysis/interpretation_and_predictions.md)
- [Qwen top-200 supplementary probe](../results/analysis/morphology_probe_qwen_raw_markup_completion/morphology_stratified_probe_report.md)
- [low_spread individual output analysis](../results/analysis/shape_low_spread_output_analysis.md)
- [all-shape output-log summary](../results/analysis/morphology_shape_output_summary.md)
- [raw/wrapped output comparison](../results/analysis/prompt_framing_output_analysis.md)
- [shape output-mode classification](../results/analysis/morphology_shape_output_modes.md)
- [token/output prompt-level link analysis](../results/analysis/morphology_shape_token_output_link.md)
- [token-feature output classifier](../results/analysis/morphology_output_feature_classifier.md)
- [unseen-shape holdout classifier](../results/analysis/morphology_shape_holdout_classifier.md)
- [paired shape effects](../results/analysis/morphology_shape_paired_effects.md)
- [cross-model shape effect classes](../results/analysis/morphology_cross_model_effect_classes.md)
- [paired effect uncertainty](../results/analysis/morphology_shape_paired_uncertainty.md)
