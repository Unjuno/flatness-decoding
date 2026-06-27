# WritingPrompts Qualitative Evidence Map

## 1. Purpose
This note synthesizes the qualitative evidence across the WritingPrompts audits, the GPT-2 small case review, the summary tables, and the likelihood diagnostic. The goal is not to adjudicate story quality in a formal sense. The goal is to map how different decoding regimes fail or succeed in recognizable ways, and to separate a safe paper claim from stronger claims that would overreach the evidence.

## 2. Evidence Sources Inspected
The synthesis draws on the following completed artifacts:

- `results/summary/comparison_gpt2_writingprompts_qualitative_audit.md`
- `results/summary/comparison_distilgpt2_writingprompts_qualitative_audit.md`
- `results/analysis/writingprompts_gpt2_small_case_review_report.md`
- `results/analysis/writingprompts_gpt2_small_case_review_summary.csv`
- `results/analysis/writingprompts_gpt2_small_case_review_scored_items.csv`
- `results/analysis/successor_dispersion_likelihood_diagnostics.md`
- `results/analysis/successor_dispersion_likelihood_correlations.csv`
- `results/summary/comparison_gpt2_writingprompts_merged_table.csv`
- `results/summary/comparison_distilgpt2_writingprompts_merged_table.csv`
- `results/summary/writingprompts_experiment_summary.md`
- `results/summary/writingprompts_key_results.csv`
- `paper/figures/writingprompts_gpt2_flatness_vs_n.csv`
- `paper/figures/writingprompts_gpt2_tradeoff_table.csv`
- `paper/figures/writingprompts_gpt2_case_review_summary.csv`

Raw generated continuations are not required for the conclusions below. When local examples are available, they align with the same qualitative patterns; when they are not, the summary tables and case-review artifacts are sufficient for this synthesis.

## 3. Method-Family Map
The completed evidence groups naturally into a few families:

- Baselines: `greedy`, `beam`, `sampling`, `temperature`, `top_p`, `top_k`
- Random control: `random`
- Low-dispersion extreme: `minflat`
- High-dispersion extreme: `maxflat`
- Center-region selectors: `centerflat_q25`, `centerflat_q50`, `centerflat_q75`
- Likelihood-constrained center-region selector: `highprob_centerflat`

The key interpretive question is not whether any single method is always good. The question is which families reveal stable, interpretable behavior under the fixed model and prompt set.

## 4. Failure-Mode Taxonomy
The recurring failure classes across the audits and case review are:

- prompt drift
- generic filler
- repetition loop
- symbol or encoding noise
- abrupt truncation
- metadata-like or list-like continuation
- incoherent event sequence
- contradiction or role confusion
- safe/conservative blandness
- local fluency but global story failure

The taxonomy is intentionally conservative. A continuation can be fluent at the sentence level and still fail the prompt, or it can be likely-looking and still be unusable as a story continuation.

## 5. Method-by-Method Qualitative Profile

### Greedy
Greedy decoding is fluent but conservative. It often produces collapsed or repetitious continuations and can stop exploring too early. It is a useful baseline because it shows that “safe” current-step selection is not equivalent to good open-ended continuation.

### Beam
Beam search is similarly fluent but often even more conservative. It can collapse into repetitive or generic endings, especially when the prompt needs longer-range creative continuation. Beam is a strong reminder that higher search effort does not automatically translate to more usable story output.

### Sampling
Sampling is the most familiar stochastic baseline. It is often usable and diverse, but it can drift, meander, or become overly generic depending on the prompt. It supports the paper’s framing by showing that likelihood/diversity/repetition trade-offs are real even before the successor-state signal is introduced.

### Temperature
Temperature usually increases lexical variety and can reduce some repetition, but it also raises the risk of weaker coherence or prompt drift. In the completed summaries it is not the main baseline winner; it is better read as a contrastive control for diversity-heavy decoding.

### top_p
Top-p is the strongest baseline trade-off in the completed audits. It is often a reasonable balance of fluency and diversity, though it still drifts into generic or non-story text. This is important because it means the paper should not claim successor logit dispersion dominates the standard decoding family.

### top_k
Top-k is usable but less balanced than top-p. It often looks safe locally while failing globally, and the case review shows it can drift into wiki-like, essay-like, or metadata-like text. It is a useful comparator because it often looks plausible for several tokens before the global failure becomes obvious.

### random
Random-N is a neutral control. It is often less collapsed than the worst extremes, but it does not by itself produce a clearly better qualitative result as `N` increases. The key point is that Random-N provides a baseline for the reranking family without imposing a successor-state preference.

### minflat
Minflat is the clearest risky extreme. It concentrates the signal strongly toward low successor dispersion, but the qualitative evidence shows that this often comes with off-distribution fragments, abrupt truncation, repetition collapse, and symbol or encoding noise. Minflat is the best diagnostic example of why low dispersion is not a quality objective.

### maxflat
Maxflat is the opposite diagnostic extreme. It is less catastrophic than minflat, but it is often diffuse, weakly grounded, or generic. It may remain sentence-level fluent while failing at story-level coherence. This is useful because it shows that high dispersion is not a free win either.

### centerflat_q50
Centerflat_q50 is one of the most balanced regimes. It often preserves local fluency while reducing the most obvious degeneration, but it still fails on prompt drift or generic filler in many cases. Its main value is that it sits in the middle of the method family and therefore helps motivate the paper’s center-region interpretation.

### centerflat_q75
Centerflat_q75 is often as good as or slightly safer than `centerflat_q50` in the case review. It remains imperfect, but it repeatedly appears among the more plausible center-region options. This makes it a natural candidate for the main paper claim about a practical center regime.

### highprob_centerflat
Highprob_centerflat is the most likelihood-friendly and conservative n-trial selector. It often looks safer than the extremes, but the cost of that safety is repetitive, markup-like, or generic continuation. The evidence supports describing it as likelihood-friendly but not necessarily story-optimal.

## 6. Quantitative-to-Qualitative Alignment
The quantitative tables and qualitative audits are aligned in broad outline:

- `minflat` has the strongest low-dispersion profile and the worst qualitative failure modes.
- `maxflat` raises dispersion but often becomes diffuse or weakly grounded.
- `centerflat_q50` and `centerflat_q75` occupy the most balanced region.
- `highprob_centerflat` is low-rank / low-PPL / conservative, but that safety often comes with genericness or repetition.
- Larger `N` strengthens control, but the qualitative gain from `N=32` over `N=8` or `N=16` is usually modest.

The correlation diagnostic supports the same conservative interpretation: successor logit dispersion is likelihood-related, but it is not identical to PPL/current-token likelihood.

## 7. Claim-Safe Interpretation
The evidence supports the following paper-safe statements:

- successor logit dispersion is measurable and controllable;
- it is likelihood-related but not identical to PPL;
- the center region is the most balanced part of the observed signal;
- extreme regimes are useful diagnostics;
- the study is empirical and diagnostic, not a universal quality-improvement claim.

## 8. Claim-Unsafe Interpretation
The evidence does not support the following stronger claims:

- that successor logit dispersion is Shannon entropy control;
- that it is orthogonal to likelihood;
- that it is a universal replacement for top-p/top-k/temperature;
- that it guarantees better story quality;
- that the qualitative reviews constitute formal human evaluation;
- that the method generalizes to larger or instruction-tuned models without further study.

## 9. Suggested Report Language
The paper can safely say that candidate-induced successor-state shape is a distinct decoding-time signal whose regimes produce systematically different likelihood, diversity, repetition, runtime, and degeneration profiles. It should also say that center-region selectors appear to offer the best compromise in the completed WritingPrompts runs, while minflat and maxflat function as diagnostic extremes.

## 10. Open Caveats

- The audits are LLM-assisted exploratory reviews.
- The case review is small and method-level, not a benchmark.
- The correlations are descriptive, not causal.
- The evidence is strongest for the specific WritingPrompts setting used here.
- The raw generations should remain externally archived rather than being treated as a repository-internal evaluation source.
