# Gemma 3 4B-it postprocess plan

## 1. Expected output location

Gemma smoke outputs should be kept in a dedicated directory such as
`results/gemma3_4b_it_smoke/` or `outputs/gemma3_4b_it_smoke/`, separate from
the completed GPT-2 and DistilGPT-2 experiment directories.

## 2. Required metrics

If the smoke run succeeds, compute the same descriptive metrics used in the
WritingPrompts summary tables:

- `mean_chosen_flatness`
- `mean_chosen_current_logprob`
- `mean_chosen_current_rank`
- `distinct_2`
- `repetition_4`
- runtime and tokens-per-second

Optional secondary checks:

- candidate flatness spread;
- selection gain;
- fallback rate for the high-probability selector.

## 3. Minimum tables

At minimum, produce:

- `results/analysis/gemma3_4b_it_smoke_summary.csv`
- `results/analysis/gemma3_4b_it_method_tradeoff.csv`
- `results/analysis/gemma3_4b_it_qualitative_notes.md`

## 4. Qualitative review approach

Use a small exploratory review of representative outputs from the paper-facing
sanity check. Focus on:

- prompt drift;
- generic filler;
- repetition loops;
- symbol / markup noise;
- abrupt truncation;
- incoherent event sequence;
- metadata-like continuations;
- conservative blandness.

This review is for failure-mode inspection, not formal human evaluation.

## 5. How to compare with GPT-2 without overclaiming

Treat GPT-2 and DistilGPT-2 as lightweight base-model testbeds and Gemma 3 4B-it
as a modern instruct-model sanity check. Compare method-level patterns only. Do
not claim apples-to-apples comparability, universal superiority, or transfer to
all larger models.

## 6. Claim-safe wording if results succeed

The Gemma 3 4B-it smoke experiment suggests that successor logit dispersion
remains measurable and controllable in a modern instruction-tuned model.
Because the run is intentionally small, it is treated as a sanity check rather
than a benchmark.

## 7. Claim-safe wording if results fail

The Gemma 3 4B-it smoke experiment exposed implementation or model-format
constraints that prevent a clean comparison in this pass. This does not
invalidate the GPT-2 diagnostic results, but it limits claims about transfer to
modern instruction-tuned models.

## 8. Archive instructions

Preserve any successful Gemma outputs externally rather than committing raw
artifacts directly to Git. If archiving is needed, package only the smoke output
directory, the manifest, and the summary tables. Keep checksum instructions
portable across Linux and macOS (`sha256sum` or `shasum -a 256`).
