# Manual shape report audit

## Scope

This audit checks every one of the 18 report files against the final private
`shapes_20` JSONL outputs used to produce the public summaries. The raw JSONL is
not committed to the public repository. It is an anti-hallucination audit of
the analysis artifacts: file identity, model identity, row counts, and the
strength of the written claims are checked separately.

## File-level checks

All 18 reports have exactly three corresponding output files, one each for
GPT-2, Gemma 3 raw, and Qwen raw. Each corresponding JSONL has exactly 20
generation rows, so every report covers 60 saved continuations. The checked
shapes are:

`heavy_tail`, `high_concentration`, `high_spread`, `large_gap`,
`low_concentration`, `low_spread`, `mid_concentration`, `mid_gap`,
`mid_spread`, `mid_tail`, `multi_peak`, `rank_slope_flat`, `rank_slope_mid`,
`rank_slope_steep`, `small_gap`, `smooth_top`, `spiky_top`, `thin_tail`.

## Claim audit

- No report invents a model, shape, prompt count, or output file.
- The descriptions of GPT-2 article/code fragments, Gemma title/settings
  templates, and Qwen question/answer or multilingual/format drift are
  grounded in the saved continuation text and are consistent with the
  corresponding output-mode rows.
- The reports do not claim that a shape improves semantic quality, factuality,
  or coding ability.
- `multi_peak` was corrected from “the strongest association” to “one
  candidate associated with mode mixing”; the data do not establish a unique
  causal effect for that shape.
- Statements such as “多い”, “目立つ”, and “寄る” are qualitative judgments,
  not statistically tested population claims. They must not be presented as
  significance results.

## What remains unproven

The saved data support a manual description of visible output modes. They do
not prove that the morphology causes those modes, that the same effect holds
for new prompts, or that the modes reflect semantic quality. Prompt-level
paired causal tests and blinded human/LLM evaluation are still required for
those claims.
