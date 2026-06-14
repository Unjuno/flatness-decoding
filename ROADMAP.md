# Roadmap

This roadmap is the operational plan for finishing the thesis and experiment repository. It is organized by phases, gates, and concrete deliverables.

## Current canonical thesis direction

Title draft:

> 次状態 logit 平滑性に基づくデコーディング制御の実験的評価

Core claim:

> Flatness-N is a training-free decoding controller that selects tokens by the standard deviation of the next-state raw logits after candidate append. The thesis evaluates whether this signal is controllable and whether it aligns with quality-related output metrics.

The thesis must not claim that flatness automatically means quality. The key empirical question is whether flatness behaves like useful learned continuation diversity or merely random uncertainty.

## Canonical files

Use these as the current source of truth.

| Area | Canonical file |
|---|---|
| Manuscript | `paper/manuscript.md` |
| Thesis plan | `paper/thesis_blueprint_final.md` |
| Quality evaluation | `paper/quality_evaluation_plan.md` |
| Flatness validity argument | `paper/flatness_validity_argument.md` |
| Motivation analogy | `paper/motivation_biodiversity_analogy.md` |
| Evaluation matrix | `notes/evaluation_matrix.md` |
| Parallel execution | `notes/parallel_execution.md` |
| Model/cache policy | `models/README.md` |

Older notes can remain, but they should not override the files above.

---

# Phase 0: repository sanity and local environment

## Goal

Verify that the repository can run locally on the available PC.

## Tasks

1. Pull latest repo.
2. Install dependencies.
3. Download model cache.
4. Check device detection.
5. Run unit tests.
6. Run single smoke generation.
7. Check paper markers.

## Commands

```bash
git pull
make install
make download-models
make check
make test
make smoke
make paper-check
```

## Gate

Proceed only if:

- `make check` identifies the expected device.
- `make test` passes.
- `make smoke` generates text without crashing.
- `make paper-check` runs and reports unresolved manuscript markers rather than script failure.

## Status

[LOCAL VERIFY]

---

# Phase 1: hardware-aware execution plan

## Goal

Detect the local PC environment and decide how many workers/shards to use.

## Tasks

1. Run hardware-aware parallelism planner.
2. Inspect backend and recommended workers.
3. Dry-run parallel launcher.
4. Adjust workers manually only if memory errors occur.

## Commands

```bash
make plan-parallel

PYTHONPATH=src python scripts/run_experiment_parallel.py \
  --config configs/comparison_smoke.yaml \
  --num-shards auto \
  --workers auto \
  --dry-run
```

## Gate

Proceed only if:

- planner reports a plausible backend: `cpu`, `mps`, `cuda_single`, or `cuda_multi`.
- dry run resolves workers and shards.

## Status

[LOCAL VERIFY]

---

# Phase 2: smoke comparison

## Goal

Verify the full pipeline on toy prompts.

## Tasks

1. Run smoke comparison with auto parallelism.
2. Aggregate results.
3. Produce summary table and figures.
4. Check whether output files are produced.

## Commands

```bash
make comparison-smoke-parallel
make extended-smoke
make ppl-smoke
make trends-smoke
make metric-suite-smoke
make export-smoke-table
```

## Expected outputs

```text
outputs/generations/comparison_smoke/
outputs/token_logs/comparison_smoke/
results/summary/comparison_smoke_metrics.csv
results/summary/comparison_smoke_table.csv
results/summary/comparison_smoke_analysis.md
results/summary/comparison_smoke_extended_metrics.csv
results/summary/comparison_smoke_ppl.csv
results/summary/comparison_smoke_n_trend_models.csv
results/figures/comparison_smoke/
paper/tables/comparison_smoke_table.md
```

## Gate

Proceed only if:

- generation files exist for baselines and N-trial methods.
- token logs exist for Random-N, MinFlat-N, and MaxFlat-N.
- `comparison_smoke_table.csv` contains `random`, `minflat`, and `maxflat`.
- no obvious duplicated stale files from older runs are mixed into the same output directory.

## Status

[LOCAL VERIFY]

---

# Phase 3: main prompt subset

## Goal

Create a fixed prompt subset for thesis experiments.

## Tasks

1. Build `prompts/main_prompts.jsonl`.
2. Inspect prompt count and quality.
3. Keep the same prompt file for all methods.

## Commands

```bash
make build-prompts
head -n 5 prompts/main_prompts.jsonl
wc -l prompts/main_prompts.jsonl
```

## Gate

Proceed only if:

- prompt file exists.
- prompt count is sufficient for the chosen experiment scale.
- prompts are not empty or malformed.
- every row has `id` and `prompt`.

## Status

[LOCAL VERIFY]

---

# Phase 4: lightweight main experiment

## Goal

Run a thesis-like experiment using `distilgpt2` before committing to heavier `gpt2` runs.

## Tasks

1. Run `comparison_distilgpt2_main` with auto parallelism.
2. Aggregate metrics.
3. Run extended metrics and optional PPL.
4. Produce plots.
5. Check core diagnostics.

## Commands

```bash
make comparison-distilgpt2-main-parallel

PYTHONPATH=src python scripts/evaluate_extended_metrics.py \
  --input-dir outputs/generations/comparison_distilgpt2_main \
  --output results/summary/comparison_distilgpt2_main_extended_metrics.csv

PYTHONPATH=src python scripts/evaluate_ppl.py \
  --input-dir outputs/generations/comparison_distilgpt2_main \
  --model distilgpt2 \
  --output results/summary/comparison_distilgpt2_main_ppl.csv

PYTHONPATH=src python scripts/fit_n_trends.py \
  --table results/summary/comparison_distilgpt2_main_table.csv \
  --output results/summary/comparison_distilgpt2_main_n_trend_models.csv \
  --figure-dir results/figures/comparison_distilgpt2_main/n_trends
```

## Core diagnostics

Check:

```text
mean_chosen_flatness(minflat) < mean_chosen_flatness(random)
mean_chosen_flatness(random) < mean_chosen_flatness(maxflat)
```

Then inspect:

```text
flatness vs repetition
flatness vs distinct-n
flatness vs PPL/NLL
N vs runtime
qualitative samples
```

## Gate

Proceed to full main experiment only if:

- output generation is not obviously broken.
- MinFlat controls flatness at least on some N values.
- runtime is acceptable for the chosen PC.
- output directories are clean and reproducible.

If MinFlat does not control flatness, debug before expanding.

## Status

[LOCAL VERIFY]

---

# Phase 5: full main experiment

## Goal

Run the main thesis experiment using `gpt2`.

## Tasks

1. Confirm model is cached.
2. Run `comparison_main_template.yaml`.
3. Aggregate all metrics.
4. Produce final tables and figures.
5. Export qualitative samples.

## Commands

```bash
make comparison-main-parallel

PYTHONPATH=src python scripts/evaluate_extended_metrics.py \
  --input-dir outputs/generations/comparison_main \
  --output results/summary/comparison_main_extended_metrics.csv

PYTHONPATH=src python scripts/evaluate_ppl.py \
  --input-dir outputs/generations/comparison_main \
  --model gpt2 \
  --output results/summary/comparison_main_ppl.csv

PYTHONPATH=src python scripts/fit_n_trends.py \
  --table results/summary/comparison_main_table.csv \
  --output results/summary/comparison_main_n_trend_models.csv \
  --figure-dir results/figures/comparison_main/n_trends

PYTHONPATH=src python scripts/export_samples_markdown.py \
  --input-dir outputs/generations/comparison_main \
  --output results/samples/comparison_main_samples.md
```

## Gate

Proceed to writing Results only if:

- tables exist.
- figures exist.
- token logs exist.
- qualitative samples exist.
- all result claims can be linked to a table, figure, or generated analysis file.

## Status

[LOCAL VERIFY]

---

# Phase 6: optional quality strengthening

## Goal

Strengthen the quality evaluation if time and compute allow.

## Options

### Option A: MAUVE

Use only if enough human reference distribution exists.

```bash
make install-eval
PYTHONPATH=src python scripts/evaluate_mauve.py \
  --input-dir outputs/generations/comparison_main \
  --reference-prompts prompts/main_prompts_with_references.jsonl \
  --output results/summary/comparison_main_mauve.csv
```

### Option B: reference metrics

Use only for reference-bearing datasets.

```bash
PYTHONPATH=src python scripts/evaluate_reference_metrics.py \
  --input-dir outputs/generations/<experiment> \
  --reference-prompts prompts/<reference_file>.jsonl \
  --output results/summary/<experiment>_reference_metrics.csv
```

### Option C: small human or LLM-judge study

Suggested minimal design:

- 20 prompts.
- Compare Flatness-N vs top-p.
- Compare Flatness-N vs Random-N.
- Pairwise preference or 1-5 rating.
- Blind method labels.

## Gate

Only include optional metrics if the setup is clean and defensible. Do not include weak reference metrics for open-ended tasks without explaining their limitations.

## Status

[OPTIONAL]

---

# Phase 7: manuscript filling

## Goal

Fill the manuscript from verified outputs.

## Tasks

1. Resolve `[VERIFY]` citations.
2. Replace `[RESULT TBD]` only when result files exist.
3. Insert tables and figures.
4. Write Results in RQ order.
5. Write Discussion based on outcomes.

## Writing order

1. Method.
2. Experimental Setup.
3. Related Work.
4. Introduction polishing.
5. Results.
6. Discussion.
7. Conclusion.
8. Abstract last.

## Commands

```bash
make paper-check
```

Before final submission:

```bash
make paper-check-final
```

## Gate

Proceed only if:

- every result claim points to a table, figure, or generated analysis file.
- `[RESULT TBD]` markers are removed or justified.
- citations have been verified.
- limitations are updated based on actual outcomes.

## Status

[IN PROGRESS]

---

# Phase 8: repository cleanup before submission

## Goal

Make the repository readable and reproducible.

## Tasks

1. Mark canonical docs clearly.
2. Leave older notes but avoid contradictions.
3. Ensure large outputs and models are not committed.
4. Track result summaries and figures if appropriate.
5. Confirm `.gitignore` covers large artifacts.
6. Optional: DVC for large outputs.

## Checks

```bash
git status
make paper-check
```

## Gate

Proceed only if:

- no model weights are staged.
- no huge generation/token-log files are accidentally staged unless intentionally DVC-managed.
- README points to canonical roadmap and manuscript.

## Status

[TODO]

---

# Decision tree after first meaningful results

## Case 1: Flatness-N controls flatness and quality proxies improve

Conclusion direction:

> Next-state flatness can act as a useful decoding-time signal under the tested conditions.

## Case 2: Flatness-N controls flatness but quality proxies are mixed

Conclusion direction:

> Next-state flatness is controllable and behaviorally meaningful, but not sufficient as a standalone quality criterion.

## Case 3: Flatness-N controls flatness but quality worsens

Conclusion direction:

> Flatness alone may preserve uncertainty rather than useful learned diversity; additional constraints are needed.

## Case 4: Flatness-N does not control flatness

Conclusion direction:

> The candidate proposal, N, or flatness score is insufficient in the tested setting. Treat as negative result and analyze failure.

---

# Immediate next actions

When local execution is available:

```bash
make install
make download-models
make check
make test
make smoke
make plan-parallel
make comparison-smoke-parallel
```

If smoke passes, proceed to:

```bash
make build-prompts
make comparison-distilgpt2-main-parallel
```

Do not run full `comparison-main-parallel` until lightweight main outputs look sane.
