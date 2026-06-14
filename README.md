# flatness-decoding

Experimental evaluation of **next-state logit flatness** as a decoding criterion for autoregressive language generation.

This repository is a private research workspace for evaluating a decoding controller called **Flatness-N Decoding**. The goal is not to claim universal improvement, but to test how selecting tokens by next-state logit flatness changes generation trajectories, repetition, diversity, quality proxies, and compute cost.

## Start here

The current operational plan is fixed in:

```text
ROADMAP.md
```

Use it as the main checklist for local setup, smoke tests, main experiments, evaluation, and manuscript filling.

Canonical thesis files:

```text
paper/manuscript.md
paper/thesis_blueprint_final.md
paper/quality_evaluation_plan.md
paper/flatness_validity_argument.md
notes/evaluation_matrix.md
notes/parallel_execution.md
```

## Core idea

At generation step `t`, given the current context `x_t`:

1. Sample `N` candidate tokens from a proposal distribution.
2. Temporarily append each candidate token.
3. Run the model one step ahead for each candidate.
4. Compute the standard deviation of the next-state logits.
5. Select the candidate whose next-state logits are flattest.

Formally:

```text
y_1, ..., y_N ~ q(y | x_t)
F(y_j) = Std(z(x_t + y_j))
y_t* = argmin_j F(y_j)
```

If only probabilities are available, `Std(log p)` can be used because `log p_i = z_i - C`, so `Std(log p) = Std(z)`.

## Main research questions

1. Does Flatness-N reduce next-state logit standard deviation compared with Random-N under the same candidate budget?
2. Does the flatness change align with quality-related metrics such as diversity, repetition, PPL/NLL, qualitative coherence, and runtime?
3. How do internal and output metrics change as `N` increases?

## Comparison set

The repository includes a thesis-level comparison set.

### Standard baselines

- `greedy`
- `beam`
- `sampling`
- `temperature`
- `top_p`
- `top_k`

### N-trial methods

- `random`: Random-N baseline
- `minflat`: Flatness-N / MinFlat-N
- `maxflat`: MaxFlat-N negative control

`N` sweep is applied only to N-trial methods. Standard baselines do not have `N`.

## Hardware support and parallel execution

The code is designed to run on:

- CUDA GPU
- Apple Silicon MPS
- CPU fallback

Use `--device auto` to select `cuda > mps > cpu` for model execution.

For experiments, use the hardware-aware parallel planner:

```bash
make plan-parallel
```

Parallel targets use automatic worker/shard selection:

```bash
make comparison-smoke-parallel
make comparison-distilgpt2-main-parallel
make comparison-main-parallel
```

See:

```text
notes/parallel_execution.md
```

## Model cache policy

Model weights are downloaded locally and are not committed to Git.

Default cache:

```text
models/hf/
```

Download initial models:

```bash
make download-models
```

See:

```text
models/README.md
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
make install
make download-models
make check
make test
make smoke
make plan-parallel
```

## First full smoke comparison

Run:

```bash
make comparison-smoke-parallel
```

Then optional extended checks:

```bash
make extended-smoke
make ppl-smoke
make trends-smoke
make metric-suite-smoke
make export-smoke-table
```

Outputs:

```text
outputs/generations/comparison_smoke/
outputs/token_logs/comparison_smoke/
results/summary/comparison_smoke_metrics.csv
results/summary/comparison_smoke_table.csv
results/summary/comparison_smoke_analysis.md
results/summary/comparison_smoke_token_trace.csv
results/figures/comparison_smoke/
results/figures/comparison_smoke/token_analysis/
paper/tables/comparison_smoke_table.md
```

## Build a main prompt subset

Before the main experiment, create a fixed prompt subset:

```bash
make build-prompts
```

This writes:

```text
prompts/main_prompts.jsonl
```

The prompt file is the experimental set. Every method must use the same rows in the same order.

## Recommended experiment order

Follow `ROADMAP.md`. The short version is:

```bash
make comparison-smoke-parallel
make build-prompts
make comparison-distilgpt2-main-parallel
# inspect outputs before running the full gpt2 experiment
make comparison-main-parallel
```

Do not run the full main experiment until smoke and lightweight main outputs look sane.

## Run a single generation

```bash
PYTHONPATH=src python scripts/run_generate.py \
  --model distilgpt2 \
  --prompt "Once upon a time" \
  --method minflat \
  --n-trials 4 \
  --max-new-tokens 50 \
  --device auto
```

## Data and artifact policy

- Code, configs, prompt subsets, summary CSVs, and figures should be tracked by Git.
- Raw datasets, full generations, token-level logs, model caches, and checkpoints should be tracked by DVC or kept outside Git.
- Do not commit full logits or full probability vectors.

## Paper workflow

Working manuscript:

```text
paper/manuscript.md
```

Check unresolved markers:

```bash
make paper-check
```

Before final submission:

```bash
make paper-check-final
```

Unresolved or result-dependent statements are intentionally marked with:

```text
[TODO]
[RESULT TBD]
[VERIFY]
[CITATION NEEDED]
[FIGURE TBD]
[TABLE TBD]
[ALGORITHM TBD]
```

## Current status

The repository contains:

- Flatness-N, Random-N, and MaxFlat-N implementations
- standard baselines including beam search
- model cache/downloader support
- hardware-aware parallel execution
- comparison experiment configs
- prompt subset builder
- evaluation, trend fitting, and plotting scripts
- paper manuscript and thesis planning files
- roadmap and local verification gates

The next local step is:

```bash
make install
make download-models
make check
make test
make smoke
make plan-parallel
make comparison-smoke-parallel
```

Then inspect whether:

```text
mean_chosen_flatness(minflat) < mean_chosen_flatness(random) < mean_chosen_flatness(maxflat)
```

and how output metrics change as `N` increases.
