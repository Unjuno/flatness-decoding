# Independent holdout baseline evaluation

This evaluation tests whether the qualitative generation conclusions are tied
to the original 20-prompt morphology set. It uses the next 20 prompts from the
tracked 100-prompt WritingPrompts file (`wp_000020`–`wp_000039`), renamed to
`wp_holdout_000000`–`wp_holdout_000019`. The holdout therefore has no prompt
overlap with the original `limit: 20` experiments.

## Design

- Model: `gpt2`
- Prompts: 20 independent holdout prompts
- Seeds: 1 and 2
- Maximum continuation: 120 tokens
- Methods: greedy, beam (4 beams), unrestricted sampling, temperature (1.2),
  top-p (0.9), and top-k (50)
- Outputs: 12 JSONL files, 240 generations in total

Reproduction:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_prompt_holdout.py \
  --input prompts/writingprompts_fixed_100.jsonl \
  --output prompts/writingprompts_holdout_20.jsonl \
  --start 20 --limit 20 --prefix wp_holdout
PYTHONPATH=src .venv/bin/python scripts/run_pipeline.py \
  --config configs/comparison_gpt2_holdout_baselines.yaml \
  --skip-plot --skip-analysis
```

## Results

Aggregated across both seeds:

| method | mean words | distinct-2 | repetition-4 |
|---|---:|---:|---:|
| greedy | 108.35 | 0.146 | 0.825 |
| beam | 107.75 | 0.251 | 0.715 |
| sampling | 88.28 | 0.917 | 0.028 |
| temperature | 80.90 | 0.991 | 0.001 |
| top-k | 98.18 | 0.908 | 0.009 |
| top-p | 88.05 | 0.985 | 0.000 |

The holdout reproduces the broad baseline separation: deterministic decoding
is much more repetitive, while stochastic decoding is more diverse. This is a
baseline sanity check, not evidence that any successor-logit shape causes the
difference.

Reference ROUGE scores are exported separately, but are supplementary only:
the reference is a different author's continuation and the model output is
capped at 120 tokens. They are not a substitute for semantic human or blind
LLM evaluation.

## Cost and remaining gap

The run took 43,200 relative baseline units, or 112,320 GPT-2-scaled units in
the repository estimator. This excludes successor-shape lookahead. A shape
experiment is more expensive because each generation step evaluates the
current state plus candidate states.

This experiment validates generalization beyond the original 20 prompts and
supplies standard decoding baselines. It does not yet test whether the 18
morphology labels predict semantic coherence or prompt adherence. That requires
attaching morphology features to the same holdout generations and evaluating
pre-specified blind judgments or semantic metrics.
