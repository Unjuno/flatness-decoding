# Gemma 3 4B-it extension plan

## 1. Purpose

This note defines a staged modern instruction-tuned model sanity check for the
successor logit dispersion decoder family. The completed GPT-2 and DistilGPT-2
WritingPrompts runs remain the main experimental results. The Gemma pass is
intentionally smaller and more conservative: it is designed to answer whether
candidate-induced successor logit dispersion remains measurable and controllable
in a modern instruct-model setting, not to replace the GPT-2 experiments or to
establish broad quality superiority.

## 2. Why Gemma 3 4B-it

Gemma 3 4B-it is a useful modern open instruction-tuned model for a sanity
check. It is large enough to expose tokenizer, chat-template, padding, cache,
and memory behaviors that may not appear in GPT-2, while still being realistic
enough for local staged probing.

## 3. Why not Gemma 4 for this paper pass

Gemma 4 is not the right target for this pass because the goal is a conservative
compatibility and signal-separation check, not a model-family comparison or a
larger-scale benchmark. The current paper package should first confirm that the
successor-state signal behaves sensibly in a modern instruct model before any
larger or more expensive extension is considered.

## 4. Why 1B is useful only for debugging

`google/gemma-3-1b-it` is a debugging fallback only. It can help diagnose
tokenizer, prompt-formatting, or loading issues when 4B access is blocked or
too expensive, but its behavior should not be treated as paper-facing evidence.

## 5. Why 4B is the paper-facing sanity check

`google/gemma-3-4b-it` is the intended paper-facing Gemma sanity check because
it is large enough to be a meaningful instruct-model target while still being
feasible for a small staged probe. If the signal remains controllable here, that
supports the conservative claim that the method is not limited to GPT-2-style
base models.

## 6. Model IDs and access status

- Primary model: `google/gemma-3-4b-it`
- Debug fallback: `google/gemma-3-1b-it`

Access status should be checked locally with Hugging Face tooling before any
generation is attempted. If gated access or authentication is required, use a
manual login step such as `huggingface-cli login`. The scripts in this repo do
not assume that model downloads are already available.

## 7. Instruct-model interpretation

Gemma uses instruct-style prompt formatting and may expose a chat template.
That means the decoded outputs are not directly comparable to GPT-2 continuations
in a strict apples-to-apples sense. The right interpretation is a modern
instruct-model sanity check: does candidate-induced successor distribution shape
still change generation behavior in a measurable way, and do the same rough
regimes appear?

## 8. Why this is not apples-to-apples with GPT-2

GPT-2 and DistilGPT-2 are lightweight base-model testbeds. Gemma 3 4B-it is a
much larger instruction-tuned model with different tokenization, prompt
formatting, and generation priors. Any comparison should therefore be read as a
signal-robustness probe, not as a direct benchmark.

## 9. Pipeline compatibility findings

Current pipeline inspection suggests the repository can support arbitrary Hugging
Face causal LMs with modest care:

- `src/flatness_decoding/model_io.py` loads models through
  `AutoTokenizer.from_pretrained` and `AutoModelForCausalLM.from_pretrained`.
- `src/flatness_decoding/baselines.py` already falls back from missing
  `pad_token_id` to `eos_token_id`.
- `src/flatness_decoding/decoding.py` operates on tokenized prompts and direct
  logits, so it is not tied to a GPT-2-only architecture.
- `scripts/run_experiment_set.py` already has resource preflight checks and
  general device/dtype handling.

Potential compatibility points to inspect for Gemma:

- tokenizer and chat-template handling;
- padding side and pad token choice;
- max context length / max position handling;
- any special-token or stop-token behavior;
- model cache and dtype selection;
- memory pressure during candidate reranking;
- prompt formatting for instruct-style inputs.

The current code path appears structurally compatible with modern HF causal LMs,
but the Gemma pass should still begin with a metadata / cache / access check and
then a tiny technical smoke run.

## 10. Required code changes, if any

At this stage the preferred path is to add small staged scripts rather than
refactor the core decoding code. If a compatibility issue appears during the
smoke check, then the minimal required code change should be made in the loading
or prompt-formatting layer, not in the already-completed GPT-2 experiment path.

## 11. Staged experiment plan

### Phase 0: metadata / cache / access check

- model: `google/gemma-3-4b-it`
- no generation
- no model download unless explicitly requested
- purpose: confirm access, cache availability, tokenizer class, chat template,
  and basic loading metadata

### Phase 1: technical smoke

- model: `google/gemma-3-4b-it`
- prompts: 2
- methods: `top_p`, `random`, `minflat`, `maxflat`, `centerflat_q50`
- `N = 4`
- `max_new_tokens = 16`
- purpose: tokenizer/logits/memory check

### Phase 2: paper-facing sanity check

- model: `google/gemma-3-4b-it`
- prompts: 5
- methods: `top_p`, `top_k`, `random`, `minflat`, `maxflat`,
  `centerflat_q50`, `centerflat_q75`, `highprob_centerflat`
- `N = 8`
- `max_new_tokens = 32`
- purpose: test whether successor logit dispersion separates by method in a
  modern instruct model

### Phase 3: optional extension only if Phase 2 is cheap and stable

- model: `google/gemma-3-4b-it`
- prompts: 10 or 20
- methods: same as Phase 2
- `N = 8`
- `max_new_tokens = 32` or `64`
- purpose: additional evidence only if runtime is acceptable

## 12. Metrics to compare with GPT-2

If the Gemma smoke run succeeds, compare the same summary-table signals used in
the GPT-2 work:

- `mean_chosen_flatness`
- `mean_chosen_current_logprob`
- `mean_chosen_current_rank`
- `distinct_2`
- `repetition_4`
- runtime / tokens-per-second

The comparison should stay method-level and descriptive. It should not be used
to claim universal quality improvement.

## 13. Qualitative review plan

Use a small qualitative review of representative outputs from Phase 2 only.
Look for:

- prompt drift;
- generic filler;
- repetition loops;
- markup or symbol noise;
- abrupt truncation;
- incoherent event sequence;
- metadata-like continuation;
- conservative blandness.

The review should be exploratory and failure-mode oriented. It is not formal
human evaluation.

## 14. Compute and memory risks

Gemma 3 4B-it is roughly 30x larger than GPT-2 small by parameter count.
Successor-state reranking adds approximately one successor forward pass per
candidate per generated token. Therefore, the Gemma experiment intentionally
uses a fixed small candidate budget rather than repeating the full GPT-2 N
sweep.

The main risks are:

- memory pressure from reranking batches;
- slow candidate evaluation;
- prompt formatting mistakes;
- chat-template issues;
- special-token handling bugs;
- loading / cache failures.

## 15. Claim boundaries

Safe claims:

- successor logit dispersion remains measurable in a modern instruct model if
  the smoke check succeeds;
- the method remains a controllable decoding-time signal;
- the method is still likelihood-related but not identical to PPL/current-token
  likelihood;
- a small staged probe can separate diagnostic regimes.

Unsafe claims:

- the method is a universal quality objective;
- the method is better than all standard decoding baselines;
- the Gemma run is an apples-to-apples replacement for GPT-2;
- the run proves transfer to larger instruction-tuned systems;
- the run establishes formal human preference.

## 16. Success criteria

The staged Gemma pass is considered useful if:

- the model loads cleanly with the intended tokenizer and dtype;
- the prompt formatting is valid;
- Phase 1 runs without special-token or cache failures;
- Phase 2 completes within a reasonable runtime window;
- method-level dispersion / likelihood / repetition patterns are separable;
- the results can be described conservatively as a sanity check.

## 17. Stop criteria

Stop immediately if any of the following occurs:

- model access is gated or unavailable;
- memory use is too high for candidate reranking;
- tokenizer/chat-template handling makes outputs invalid;
- special-token bugs dominate the outputs;
- Phase 1 takes too long;
- Phase 2 is disproportionate relative to paper value;
- results would require quality-improvement claims to be useful.
