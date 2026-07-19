# flatness-decoding

Experimental archive for studying **candidate-induced successor-logit
dispersion** as a decoding-time diagnostic/control signal for autoregressive
language generation.

The repository asks a narrow question: after temporarily appending a candidate
token, how does the model's next-step logit distribution change, and what
visible generation modes are associated with that state? It does not claim
universal quality improvement, semantic understanding, or a causal effect from
any one morphology label.

## Start here

- [Public release audit](docs/public_release_audit.md)
- [Morphology analysis report](docs/morphology_analysis_report.md)
- [Independent holdout baseline evaluation](docs/holdout_baseline_evaluation.md)
- [Manual 18-shape reading reports](results/analysis/manual_shape_reports/README.md)
- [Reproducibility notes](docs/reproducibility_writingprompts.md)
- [Artifact archiving policy](docs/artifact_archiving.md)

## Main result archive

The final morphology archive contains 18 successor-logit shape labels across
GPT-2, Gemma 3 4B-it raw, and Qwen 2.5 1.5B-Instruct raw. Each shape/model
condition targets 20 prompts. The saved analysis covers distribution features,
candidate-token content, model-local risk coordinates, paired effects,
uncertainty, and qualitative output modes.

The 18 labels are descriptive projections of several axes—spread,
concentration, tail mass, top-rank gap, rank-curve slope, and peak topology.
They are not 18 independent semantic classes.

## Core method

At generation step `t`, given context `x_t`:

1. Sample `N` candidate tokens from a proposal distribution.
2. Temporarily append each candidate.
3. Run the model one step ahead for each candidate.
4. Measure successor-logit dispersion, normally `Std(logits)`.
5. Select a candidate using a chosen selector such as `minflat`, `maxflat`, or
   a center-region selector.

The implementation is in `src/flatness_decoding/`; experiment definitions are
in `configs/`; analysis entry points are in `scripts/`.

## Reproduction

Install the common and development dependencies, then run the small checks:

```bash
pip install -r requirements-common.txt -r requirements-dev.txt
PYTHONPATH=src pytest -q
```

For a configured experiment, use:

```bash
PYTHONPATH=src python scripts/run_pipeline.py \
  --config configs/comparison_gpt2_shapes_20.yaml
```

Large model caches, raw generation logs, and full token-level logs are not
intended to be committed. Curated summaries, figures, prompts, configs, and
analysis reports are tracked.

## Scope and limitations

- The morphology runs are primarily step-0/candidate-pool diagnostics, not a
  universal estimate of shape frequency throughout generation.
- Output-mode labels are qualitative surface diagnostics, not human semantic
  judgments.
- Model-local calibration matters; a shape name does not have a universal
  quality meaning across models.
- Reference-story metrics are supplementary because the reference is an
  independent continuation, not the unique correct answer.
- The archive is a reproducible research artifact, not a camera-ready paper or
  benchmark leaderboard.

## License and data

See `data/README.md`, `prompts/README.md`, and `third_party/README.md` for data
provenance and usage notes. The repository does not redistribute the original
WritingPrompts dataset.
