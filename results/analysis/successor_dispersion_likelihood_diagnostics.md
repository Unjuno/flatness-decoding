# Successor Dispersion vs Likelihood Diagnostics

This note is descriptive. It compares summary-table successor logit dispersion with summary-table likelihood-related columns and a small GPT-2 case review. The analysis is method-level, small-sample, and does not establish causality.

## Questions

1. Is successor logit dispersion identical to PPL/current likelihood? No. The summary tables show related but non-identical behavior.
2. Is it likelihood-related? Yes. The descriptive correlations are consistent with a likelihood-sensitive signal.
3. What does the correlation analysis suggest? The signal tracks likelihood-related columns in some settings, but the relationship varies by method family and subset.
4. Why should the paper avoid claiming orthogonality? Because the data do not support it and the signal clearly interacts with likelihood-sensitive behavior.
5. Why should the paper avoid claiming it is just PPL? Because successor logit dispersion measures candidate-induced successor-state shape, while PPL measures realized sequence surprisal.
6. Which method patterns illustrate the distinction?
   - `minflat`: low dispersion but bad rank/PPL and severe degeneration.
   - `highprob_centerflat`: likelihood-friendly but repetitive/generic.
   - `centerflat_q50` / `centerflat_q75`: intermediate balanced regimes.
   - `maxflat`: high-dispersion diagnostic.

## Correlation summary

- gpt2 | all rows | mean_chosen_flatness vs ppl | n=46 | pearson=-0.804934 | spearman=-0.909762
- gpt2 | all rows | mean_chosen_flatness vs mean_chosen_current_rank | n=46 | pearson=-0.931456 | spearman=-0.951570
- gpt2 | all rows | mean_chosen_flatness vs distinct_2 | n=46 | pearson=-0.305441 | spearman=-0.582104
- gpt2 | all rows | mean_chosen_flatness vs repetition_4 | n=46 | pearson=0.005055 | spearman=0.182179
- gpt2 | N-trial rows | mean_chosen_flatness vs ppl | n=40 | pearson=-0.804934 | spearman=-0.909762
- gpt2 | N-trial rows | mean_chosen_flatness vs mean_chosen_current_rank | n=40 | pearson=-0.931456 | spearman=-0.951570
- gpt2 | N-trial rows | mean_chosen_flatness vs distinct_2 | n=40 | pearson=-0.305441 | spearman=-0.582104
- gpt2 | N-trial rows | mean_chosen_flatness vs repetition_4 | n=40 | pearson=0.005055 | spearman=0.182179
- gpt2 | N=8 | mean_chosen_flatness vs ppl | n=8 | pearson=-0.835744 | spearman=-0.903614
- gpt2 | N=8 | mean_chosen_flatness vs mean_chosen_current_rank | n=8 | pearson=-0.956070 | spearman=-0.927711
- gpt2 | N=8 | mean_chosen_flatness vs distinct_2 | n=8 | pearson=-0.391665 | spearman=-0.638554
- gpt2 | N=8 | mean_chosen_flatness vs repetition_4 | n=8 | pearson=0.069666 | spearman=-0.108434
- gpt2 | N=16 | mean_chosen_flatness vs ppl | n=8 | pearson=-0.791548 | spearman=-0.734940
- gpt2 | N=16 | mean_chosen_flatness vs mean_chosen_current_rank | n=8 | pearson=-0.950967 | spearman=-0.927711
- gpt2 | N=16 | mean_chosen_flatness vs distinct_2 | n=8 | pearson=-0.207565 | spearman=-0.156627
- gpt2 | N=16 | mean_chosen_flatness vs repetition_4 | n=8 | pearson=-0.142303 | spearman=-0.228916
- gpt2 | N=32 | mean_chosen_flatness vs ppl | n=8 | pearson=-0.767619 | spearman=-0.734940
- gpt2 | N=32 | mean_chosen_flatness vs mean_chosen_current_rank | n=8 | pearson=-0.945230 | spearman=-0.927711
- gpt2 | N=32 | mean_chosen_flatness vs distinct_2 | n=8 | pearson=-0.223421 | spearman=-0.469880
- gpt2 | N=32 | mean_chosen_flatness vs repetition_4 | n=8 | pearson=-0.042064 | spearman=-0.228916
- distilgpt2 | all rows | mean_chosen_flatness vs ppl | n=46 | pearson=-0.609370 | spearman=-0.515702
- distilgpt2 | all rows | mean_chosen_flatness vs mean_chosen_current_rank | n=46 | pearson=-0.922940 | spearman=-0.872683
- distilgpt2 | all rows | mean_chosen_flatness vs distinct_2 | n=46 | pearson=-0.145431 | spearman=-0.350738
- distilgpt2 | all rows | mean_chosen_flatness vs repetition_4 | n=46 | pearson=0.022182 | spearman=0.031215
- distilgpt2 | N-trial rows | mean_chosen_flatness vs ppl | n=40 | pearson=-0.609370 | spearman=-0.515702
- distilgpt2 | N-trial rows | mean_chosen_flatness vs mean_chosen_current_rank | n=40 | pearson=-0.922940 | spearman=-0.872683
- distilgpt2 | N-trial rows | mean_chosen_flatness vs distinct_2 | n=40 | pearson=-0.145431 | spearman=-0.350738
- distilgpt2 | N-trial rows | mean_chosen_flatness vs repetition_4 | n=40 | pearson=0.022182 | spearman=0.031215
- distilgpt2 | N=8 | mean_chosen_flatness vs ppl | n=8 | pearson=-0.680580 | spearman=-0.445783
- distilgpt2 | N=8 | mean_chosen_flatness vs mean_chosen_current_rank | n=8 | pearson=-0.934966 | spearman=-0.903614
- distilgpt2 | N=8 | mean_chosen_flatness vs distinct_2 | n=8 | pearson=-0.090063 | spearman=0.012048
- distilgpt2 | N=8 | mean_chosen_flatness vs repetition_4 | n=8 | pearson=-0.047161 | spearman=-0.277108
- distilgpt2 | N=16 | mean_chosen_flatness vs ppl | n=8 | pearson=-0.592481 | spearman=-0.325301
- distilgpt2 | N=16 | mean_chosen_flatness vs mean_chosen_current_rank | n=8 | pearson=-0.927649 | spearman=-0.903614
- distilgpt2 | N=16 | mean_chosen_flatness vs distinct_2 | n=8 | pearson=-0.118813 | spearman=-0.108434
- distilgpt2 | N=16 | mean_chosen_flatness vs repetition_4 | n=8 | pearson=-0.023649 | spearman=-0.277108
- distilgpt2 | N=32 | mean_chosen_flatness vs ppl | n=8 | pearson=-0.566356 | spearman=-0.325301
- distilgpt2 | N=32 | mean_chosen_flatness vs mean_chosen_current_rank | n=8 | pearson=-0.916564 | spearman=-0.734940
- distilgpt2 | N=32 | mean_chosen_flatness vs distinct_2 | n=8 | pearson=-0.293688 | spearman=-0.277108
- distilgpt2 | N=32 | mean_chosen_flatness vs repetition_4 | n=8 | pearson=0.153262 | spearman=-0.012048
- gpt2_case_review | joined by method+n_trials | mean_chosen_flatness vs mean_severity | n=8 | pearson=-0.647571 | spearman=-0.441367
- gpt2_case_review | joined by method+n_trials | mean_chosen_flatness vs severe_issue_rate | n=8 | pearson=-0.150168 | spearman=0.097590
- gpt2_case_review | joined by method+n_trials | mean_chosen_flatness vs mean_story_fit_score | n=8 | pearson=0.606150 | spearman=0.441367
- gpt2_case_review | joined by method+n_trials | mean_chosen_flatness vs usable_rate | n=8 | pearson= | spearman=

## Interpretation

The descriptive correlations should be interpreted cautiously. They are computed on summary tables, not token-level predictions, and they mix baseline and N-trial rows when the relevant subset is broad. The main takeaway is conservative: successor logit dispersion is likelihood-related but not identical to PPL/current-token likelihood, and the completed methods behave as distinct control regimes rather than as a single monotonic likelihood axis.

## Caveats

- Small sample.
- Method-level aggregate.
- Does not establish causality.
- Missing columns are recorded as caveats rather than failures.
