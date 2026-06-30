# Successor Distribution Morphology Proxies

This note defines lightweight proxies for the shape of a successor distribution. The goal is diagnostic, not evaluative.

## Why this exists

Successor logit dispersion alone is too coarse to explain why a continuation turns into direct prose, generic prose, quiz/classification output, markup, prompt recycling, or token-fragment collapse. We want small, cheap shape proxies that can be extracted one step at a time without storing full logits.

The intended use is comparative diagnosis:

- direct narrative prose
- generic atmospheric prose
- quiz/classification/options
- markup or HTML
- prompt-marker recycling
- token-fragment collapse
- repetition or loop
- semantic drift

These are morphology proxies, not quality metrics.

## Proxy families

### Spread

- `logit_std`: standard deviation of logits
- `logit_iqr`: interquartile range of logits

### Concentration

- `top1_prob`: probability mass of the top token
- `top5_mass`: cumulative mass of the top 5 tokens
- `top10_mass`: cumulative mass of the top 10 tokens
- `softmax_entropy`: entropy of the full softmax distribution

### Tail

- `nucleus_size_p90`: smallest top-K nucleus size that captures 90% of mass
- `tail_mass_after_top50`: 1 minus the cumulative mass of the top 50 tokens

### Gap

- `gap_1_2`: logit gap between the top 1 and top 2 tokens
- `gap_1_5`: logit gap between the top 1 and top 5 tokens
- `max_gap_top50`: largest adjacent logit gap inside the top 50
- `large_gap_count_top50`: count of adjacent gaps inside the top 50 above a fixed threshold

### Token Type

Computed by decoding only the top-K tokens needed for the proxy.

- `markup_mass_top20`
- `newline_or_bullet_mass_top20`
- `prompt_overlap_mass_top20`
- `subword_fragment_mass_top20`

## Interpretation notes

- High `top1_prob` and low entropy suggest peaky, conservative continuation.
- Higher tail mass and larger nuclei suggest broader candidate support.
- Large gaps and clustered gaps can indicate multi-cluster structure.
- Markup, prompt overlap, or fragment mass can point to prompt recycling or formatting drift.
- These proxies are not claims about quality. They are only intended to help relate successor distribution morphology to observed output modes.

## Successor mode

For successor morphology, compute these proxies after appending a candidate token and running the successor forward pass. That makes the features candidate-conditioned rather than just current-state descriptors.

## Safe execution

Start with a small slice only:

```bash
python scripts/analyze_successor_distribution_morphology.py \
  --conditions qwen_raw \
  --methods top_p \
  --top-k 50 \
  --max-prompts 2 \
  --max-steps 4 \
  --dry-run
```

If execution is needed later, keep the same small slice and avoid Gemma unless `--allow-gemma` is explicitly provided.
