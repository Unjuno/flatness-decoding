# Modern Model Probe Plan

## 1. Purpose

This note records a small cross-model sanity-check plan for modern models that
complements the completed GPT-2 / DistilGPT-2 experiments and the Gemma 3
4B-it smoke work. The probes are not a benchmark suite.

## 2. Why additional models

The goal is to see whether the local output-mode steering observed in Gemma 3
4B-it is specific to Gemma-it, specific to instruction tuning, or more broadly
related to candidate-induced successor-state shape.

## 3. Model candidates

- `google/gemma-2-2b`
- `Qwen/Qwen2.5-1.5B-Instruct`

## 4. Access/cache status

- `google/gemma-2-2b`: gated / manual access may be required; cache not yet
  populated in `models/hf/`
- `Qwen/Qwen2.5-1.5B-Instruct`: public repo; cache not yet populated in
  `models/hf/`

Use `huggingface-cli login` or `hf auth login` if gated access or download
permission is needed.

## 5. Gemma base probe: `google/gemma-2-2b`

This is a Gemma-family base-model probe. It separates instruction-tuning and
prompt-template effects from base-model continuation behavior.

## 6. Qwen instruct probe: `Qwen/Qwen2.5-1.5B-Instruct`

This is a non-Gemma instruction-tuned probe. It tests whether the mode steering
observed in Gemma 3 4B-it is broader than one model family.

## 7. Relation to current Gemma 3 4B-it results

The completed Gemma 3 4B-it runs remain the main modern-model sanity check.
These additional probes are narrower follow-ups meant to probe model-family and
instruction-tuning sensitivity.

## 8. Prompt modes

Use an explicit wrapped continuation prompt, not raw prompt-only continuation.

## 9. Planned experiment shape

- prompts: 10
- prompt mode: wrapped continuation
- methods: `top_p`, `top_k`, `random`, `minflat`, `maxflat`,
  `centerflat_q50`, `centerflat_q75`, `highprob_centerflat`
- `N = 8` only
- `max_new_tokens = 48`

Explicit exclusions:

- `N = 16`
- `N = 32`
- 100 prompts
- claims of universal quality improvement
- claims of robust output-format control

## 10. Download/cache instructions

Use `scripts/cache_hf_models.py` to inspect or cache model metadata and
tokenizers. Use `--download` only when you explicitly want to populate
`models/hf/`. Keep model files out of Git.

## 11. Compatibility risks

- gated access for Gemma-family models
- tokenizer / chat-template differences
- different prompt formatting expectations
- runtime cost for successor-state reranking

## 12. Claim boundaries

These probes are small sanity checks, not a formal benchmark suite. They do not
establish universal quality improvement or robust format control.

## 13. Stop criteria

- model access is unavailable
- tokenizer / chat-template handling is invalid
- runtime is too high for the small probe
- output modes are too unstable for a clean comparison
- a later result would require stronger claims than the data support
