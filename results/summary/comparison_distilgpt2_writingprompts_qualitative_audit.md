# Qualitative Audit

## 1. Scope and caveat
This is an LLM-assisted qualitative audit, not a formal human evaluation. It is intended to check whether the quantitative trends line up with obvious generation quality issues.

## 2. Sampling method
- Prompt count sampled: 10 prompt IDs
- Methods and settings sampled: `random` N=1/8/32, `minflat` N=8/32, `maxflat` N=8/32, `centerflat_q25` N=8/32, `centerflat_q50` N=8/32, `centerflat_q75` N=8/32, `highprob_centerflat` N=8/32, plus baselines `greedy`, `beam`, `sampling`, `temperature`, `top_p`, `top_k`
- Seed policy: deterministic, using the first 10 shared prompt IDs in sorted order
- Limitation: this is a sample audit over existing outputs only; the scoring is conservative and intentionally biased toward flagging obvious degeneration

## 3. Main findings
- `minflat` is the riskiest setting. It often has high diversity-style metrics, but in this sample it also shows the highest share of unusable or near-unusable outputs.
- `maxflat` is mixed. It can look diverse, but it also drifts into diffuse or weakly grounded continuations more often than the center-flat methods.
- `centerflat_q50`, `centerflat_q75`, and `highprob_centerflat` are the most promising n-trial settings overall.
- The qualitative audit mostly supports the metric interpretation, but with an important caveat: some settings that look good on flatness/diversity metrics still produce weak or unusable continuations.

## 4. Method-specific observations
- `random`: the control is mostly usable, but `N=32` does not create a clearly better qualitative result than `N=8` in this sample.
- `minflat`: strongest risk of off-distribution, abrupt, or repetitive outputs. This is the clearest method to demote.
- `maxflat`: tends to be more diffuse and less grounded than the center-flat methods; better than `minflat` in some cases, but still risky.
- `centerflat_q25`: viable, but generally below `q50` and `q75` in balance and readability.
- `centerflat_q50`: one of the best trade-offs. It usually preserves coherence while avoiding the most obvious degeneration.
- `centerflat_q75`: at least as good as `q50` here, and sometimes a little safer in content quality.
- `highprob_centerflat`: very strong on prompt adherence and conservative behavior, but sometimes too safe or generic.
- Baselines:
  - `greedy` and `beam` are fluent but conservative, with repetitive/collapsed behavior showing up often enough to matter.
  - `sampling`, `temperature`, `top_p`, and `top_k` are mostly usable, with `top_p` looking the strongest among the baselines.

## 5. N-trials interpretation
- `N=32` is not consistently better than `N=8` in qualitative terms.
- The runtime increase is obvious, while the qualitative improvement is modest.
- For the main paper, `N=8` appears closer to the practical knee unless the metric story requires `N=32` for a specific claim.

## 6. Recommended next decision for GPT-2
- Keep the n-trial family, but treat `minflat` as the most likely candidate for demotion or exclusion.
- Keep `centerflat_q50`, `centerflat_q75`, and `highprob_centerflat` as the main focus.
- Keep `maxflat` only if the paper needs a contrastive point; otherwise it is weaker than the center-based variants.
- Do not assume `N=32` should replace `N=8` without a separate justification.

## 7. Safe paper-language paragraph
The qualitative audit suggests that `centerflat_q50`, `centerflat_q75`, and `highprob_centerflat` are the most reliable n-trial settings on the WritingPrompts benchmark, while `minflat` and some `maxflat` outputs more often show truncation, repetition, or weak prompt adherence. These observations are exploratory and do not constitute formal human evaluation.

WritingPrompts に対する定性的監査では、`centerflat_q50`、`centerflat_q75`、`highprob_centerflat` が n-trial 設定の中では比較的安定しており、`minflat` や一部の `maxflat` 出力では切断、反復、あるいはプロンプト追従の弱さが目立ちました。これは探索的な LLM 支援の監査であり、正式な人手評価ではありません。
