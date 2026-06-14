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

1. Does Flatness-N reduce next-state logit standard deviation compared with Random-N and standard sampling?
2. How does Flatness-N affect repetition, diversity, perplexity-like proxies, and runtime?
3. How do these metrics change as `N` increases?

## Methods to compare

- Standard sampling / `N=1`
- Random-N: sample `N` candidates and select randomly
- Flatness-N / MinFlat-N: select the candidate with the minimum next-state logit standard deviation
- MaxFlat-N: select the candidate with the maximum next-state logit standard deviation as a negative control
- Existing baselines later: temperature sampling, top-p, top-k, greedy, etc.

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

## Run a small generation

```bash
PYTHONPATH=src python scripts/run_generate.py \
  --model distilgpt2 \
  --prompt "Once upon a time" \
  --method minflat \
  --n-trials 4 \
  --max-new-tokens 50 \
  --device auto
```

## Run an N sweep

```bash
PYTHONPATH=src python scripts/run_sweep_n.py \
  --model distilgpt2 \
  --prompts prompts/toy_prompts.jsonl \
  --methods random minflat maxflat \
  --n-values 1 4 8 16 \
  --max-new-tokens 80 \
  --output-dir outputs/generations/smoke
```

## Evaluate generated outputs

```bash
PYTHONPATH=src python scripts/run_evaluate.py \
  --input-dir outputs/generations/smoke \
  --output results/summary/smoke_metrics.csv
```

## Data and artifact policy

- Code, configs, prompt subsets, summary CSVs, and figures should be tracked by Git.
- Raw datasets, full generations, token-level logs, model caches, and checkpoints should be tracked by DVC or kept outside Git.
- Do not commit full logits or full probability vectors.

## Status

Initial research scaffold. The first target is to verify:

```text
mean_flatness(Flatness-N) < mean_flatness(Random-N)
```

for a small prompt set and several values of `N`.
