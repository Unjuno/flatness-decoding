# WritingPrompts Experiment Summary

## Scope
This summary covers the completed `comparison_distilgpt2_writingprompts` and `comparison_gpt2_writingprompts` runs on `euclaise/writingprompts` validation prompts (fixed 100-prompt subset). The prompt set was generated once and reused across the completed experiments.

## Method
Flatness here is not Shannon entropy. It is successor-distribution flatness operationalized as the standard deviation of the next-state raw logits after temporarily appending a candidate token.
This makes the method a decoding-time control over successor logit dispersion, not temperature scaling and not direct entropy control.

## Key quantitative pattern
### distilgpt2
- random: mean_chosen_flatness=3.150; for N-trial rows this is the mean successor logit standard deviation of the selected candidate, distinct-2=0.896, repetition-4=0.034, runtime=65.49s
- minflat: mean_chosen_flatness=1.751; for N-trial rows this is the mean successor logit standard deviation of the selected candidate, distinct-2=0.845, repetition-4=0.067, runtime=89.56s
- maxflat: mean_chosen_flatness=3.950; for N-trial rows this is the mean successor logit standard deviation of the selected candidate, distinct-2=0.788, repetition-4=0.056, runtime=49.20s
- centerflat_q50: mean_chosen_flatness=3.315; for N-trial rows this is the mean successor logit standard deviation of the selected candidate, distinct-2=0.637, repetition-4=0.262, runtime=64.27s
- centerflat_q75: mean_chosen_flatness=3.486; for N-trial rows this is the mean successor logit standard deviation of the selected candidate, distinct-2=0.715, repetition-4=0.134, runtime=64.97s
- highprob_centerflat: mean_chosen_flatness=3.346; for N-trial rows this is the mean successor logit standard deviation of the selected candidate, distinct-2=0.418, repetition-4=0.475, runtime=62.28s

### gpt2
- random: mean_chosen_flatness=3.055; for N-trial rows this is the mean successor logit standard deviation of the selected candidate, distinct-2=0.924, repetition-4=0.009, runtime=142.79s
- minflat: mean_chosen_flatness=1.656; for N-trial rows this is the mean successor logit standard deviation of the selected candidate, distinct-2=0.815, repetition-4=0.078, runtime=116.58s
- maxflat: mean_chosen_flatness=4.141; for N-trial rows this is the mean successor logit standard deviation of the selected candidate, distinct-2=0.849, repetition-4=0.017, runtime=87.76s
- centerflat_q50: mean_chosen_flatness=3.298; for N-trial rows this is the mean successor logit standard deviation of the selected candidate, distinct-2=0.808, repetition-4=0.060, runtime=83.51s
- centerflat_q75: mean_chosen_flatness=3.636; for N-trial rows this is the mean successor logit standard deviation of the selected candidate, distinct-2=0.789, repetition-4=0.037, runtime=71.31s
- highprob_centerflat: mean_chosen_flatness=3.519; for N-trial rows this is the mean successor logit standard deviation of the selected candidate, distinct-2=0.584, repetition-4=0.221, runtime=70.52s

## Selector note
`midflat` is implementation-equivalent to `centerflat_q50` in the current selector family. In paper discussion it should be treated as a redundant or auxiliary label unless there is a specific reason to retain both names.

## Qualitative findings
The qualitative audits are LLM-assisted exploratory audits, not formal human evaluations.
Across both models, `minflat` is the clearest risky extreme: it lowers successor logit dispersion strongly, but it often selects low-probability, high-rank tokens and can produce abrupt, off-distribution, or repetitive outputs.
`maxflat` raises dispersion and is less catastrophic than `minflat`, but it can be diffuse or weakly grounded.
The center region is the most promising: `centerflat_q50` and `centerflat_q75` are the main candidates to discuss as a practical center-flat regime, with `highprob_centerflat` as the conservative/low-PPL option that sometimes becomes repetitive or generic.

## Cross-model consistency
The two models agree on the broad picture: extreme dispersion control is not a monotonic quality objective, center-flat regimes are the best compromise, and `N=32` usually costs much more runtime for only modest qualitative gains over `N=8` or `N=16`.

## Limitations
- The qualitative audit is sample-based and heuristic, not formal human annotation.
- The output-level quality findings should be treated as exploratory.
- The metrics and qualitative judgments do not support a claim of universal improvement over `top_p`, `top_k`, or `temperature`.

## Safe claims
- Successor logit dispersion is a controllable decoding-time signal.
- Extreme dispersion choices are not monotonic quality objectives.
- Center-flat regimes are the most promising practical compromise in these runs.

## Claims to avoid
- Do not say flatness is Shannon entropy control.
- Do not say flatness directly means quality.
- Do not claim formal human evaluation.
- Do not claim a universal win over standard baselines.

## Next recommended evaluation step
Use the WritingPrompts evidence to motivate a narrower follow-up on `centerflat_q50`, `centerflat_q75`, and `highprob_centerflat`, with `minflat` kept as a diagnostic extreme and `N=8` or `N=16` treated as the practical default unless a stronger `N=32` argument is required.

## Results links
- `results/summary/writingprompts_key_results.csv`
- `results/summary/comparison_distilgpt2_writingprompts_qualitative_audit.md`
- `results/summary/comparison_gpt2_writingprompts_qualitative_audit.md`
- `docs/public_release_audit.md`
