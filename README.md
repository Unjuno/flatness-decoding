# flatness-decoding

Experimental evaluation of **next-state logit flatness** as a decoding criterion for autoregressive language generation.

This repository is a private research workspace for evaluating a decoding controller called **Flatness-N Decoding**. The goal is not to claim universal improvement, but to test how selecting tokens by next-state logit flatness changes generation trajectories, repetition, diversity, quality proxies, and compute cost.

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

1. Does Flatness-N reduce next-state logit standard deviation compared with Random-N and ordinary baselines?
2. How does Flatness-N affect repetition, diversity, quality proxies, and runtime?
3. How do these metrics change as `N` increases?

## Comparison set

The repository now includes a thesis-level comparison set.

### Standard baselines

- `greedy`
- `sampling`
- `temperature`
- `top_p`
- `top_k`

### N-trial methods

- `random`: Random-N baseline
- `minflat`: Flatness-N / MinFlat-N
- `maxflat`: MaxFlat-N negative control

`N` sweep is applied only to N-trial methods. Standard baselines do not have `N`.

## Hardware support

The code is designed to run on:

- CUDA GPU
- Apple Silicon MPS
- CPU fallback

Use `--device auto` to select `cuda > mps > cpu`.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-common.txt
PYTHONPATH=src python scripts/check_device.py
PYTHONPATH=src python scripts/smoke_test.py
```

## Run the full comparison smoke pipeline

This is the first command to run after cloning locally.

```bash
PYTHONPATH=src python scripts/run_pipeline.py \
  --config configs/comparison_smoke.yaml
```

This runs:

1. `scripts/run_experiment_set.py`
2. `scripts/run_evaluate.py`
3. `scripts/plot_results.py`

Outputs:

```text
outputs/generations/comparison_smoke/
outputs/token_logs/comparison_smoke/
results/summary/comparison_smoke_metrics.csv
results/figures/comparison_smoke/
```

## Run the main comparison template

Before running this, create `prompts/main_prompts.jsonl` with a fixed benchmark prompt subset.

```bash
PYTHONPATH=src python scripts/run_pipeline.py \
  --config configs/comparison_main_template.yaml
```

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

## Current status

The repository now contains:

- research plan and related-work notes
- standard baseline implementations
- Flatness-N, Random-N, and MaxFlat-N implementations
- comparison experiment-set configs
- an end-to-end pipeline script
- basic evaluation and plotting scripts

The next local step is to run `configs/comparison_smoke.yaml` and inspect whether:

```text
mean_flatness(minflat) < mean_flatness(random)
```

and how the output metrics change as `N` increases.
