# Untempered Successor Dispersion Analysis

This diagnostic asks whether the successor-dispersion `p50` regime is intrinsic to the decoder state, or whether it was partly created by proposal-temperature sampling in the existing `N`-trial flatness runs.

## Why this analysis exists

The current `N`-trial procedure samples candidate tokens from a proposal distribution, then selects among those candidates using successor logit dispersion. That means the candidate pool is already shaped by temperature. This makes it hard to tell whether the `centerflat_q50` regime is truly near the middle of the raw current-token distribution, or whether the proposal sampler pulled it there.

This analysis removes that confound. It reconstructs each generation step, runs one forward pass to get raw current logits, and then evaluates the raw top-K candidates directly from those logits.

## What changes relative to the proposal-temperature runs

The old `N`-trial runs:

- sample a proposal set with temperature
- score candidates by successor dispersion
- select minflat, maxflat, q50, q75, or highprob variants from that proposal set

The untempered diagnostic:

- does not sample candidates with proposal temperature
- uses raw top-K current logits only
- computes successor dispersion for each raw candidate
- summarizes the dispersion distribution and its overlap with current-token likelihood

## Why top-K raw logits

Top-K raw logits provide a simple high-probability slice of the current distribution. If the successor-dispersion median sits inside this region, then `p50` is likely tied to raw likelihood mass rather than to proposal sampling noise.

If `p50` remains low-probability even under raw top-K, then the earlier `centerflat_q50` behavior was probably not just a temperature artifact.

## What `p50` means

`p50` means the median successor-logit-dispersion candidate among the raw top-K candidates at a specific generation step.

It does not mean:

- a globally optimal token
- a quality winner
- a proof of robust control
- a guaranteed stylistic center

It only means the candidate whose successor dispersion sits at the middle of the raw top-K dispersion distribution.

## Memory and runtime caveats

This diagnostic is intentionally written to be memory-conscious:

- `torch.inference_mode()` only
- no retained computation graphs
- no full successor-logit storage
- process one prompt, method, and step at a time
- stream CSV rows incrementally
- clear Python and device caches between flushes

Even so, the analysis can still be expensive because each current token requires `top_k` successor forward passes.

## Safety defaults

The script defaults to dry-run mode. It also refuses large runs unless explicit override flags are given.

Recommended first safe run:

```bash
python scripts/analyze_untempered_successor_dispersion_distribution.py \
  --conditions qwen_raw \
  --methods top_p \
  --top-k 20 \
  --max-prompts 2 \
  --max-steps 4 \
  --candidate-batch-size 1 \
  --dry-run
```

Recommended first execute run:

```bash
python scripts/analyze_untempered_successor_dispersion_distribution.py \
  --conditions qwen_raw \
  --methods top_p \
  --top-k 20 \
  --max-prompts 2 \
  --max-steps 4 \
  --candidate-batch-size 1 \
  --execute
```

## Interpretation guide

The key question is whether successor-dispersion `p50` still overlaps with high-probability raw candidates once proposal temperature is removed.

Comparisons should focus on:

- `p50` rank and logprob
- the mass between `p25` and `p75`
- whether high-probability candidates cluster near `p75`
- whether `q50` behavior persists without proposal sampling
- whether model families differ in overlap patterns

## GPT-2 path note

GPT-2 generation roots are not hard-coded here because repository layouts have differed across experiments. The script should use explicit condition metadata and fail clearly if the GPT-2 source path is ambiguous. Do not guess silently.
