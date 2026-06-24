# Reproducibility Note: WritingPrompts

This note records the completed WritingPrompts runs and the associated artifacts needed to reproduce the reported tables, figures, and qualitative analyses.

## Completed configs

- `configs/comparison_distilgpt2_main.yaml`
- `configs/comparison_gpt2_writingprompts` is the corresponding GPT-2 WritingPrompts configuration used in the completed run
- `configs/comparison_distilgpt2_writingprompts` is represented via the prompt artifact and completed outputs for the DistilGPT-2 WritingPrompts run

## Prompt artifact

- `prompts/writingprompts_fixed_100.jsonl`
- Source dataset: `euclaise/writingprompts`
- Split: `validation`
- Prompt subset: fixed 100 prompts

## Models

- `distilgpt2`
- `gpt2`

## Seeds

- Generation seeds were completed in the project’s standard experiment set; the completed outputs include seed 0 and multi-seed runs where configured.
- Qualitative audits used seed 0 outputs when available and retained deterministic prompt selection.

## Methods

- `greedy`
- `beam`
- `sampling`
- `temperature`
- `top_p`
- `top_k`
- `random`
- `minflat`
- `maxflat`
- `midflat` / `centerflat_q50`
- `centerflat_q25`
- `centerflat_q75`
- `highprob_centerflat`

## N values

- Baselines: no `N`
- Random: `N=1, 4, 8, 16, 32` in the completed experiments
- MinFlat: `N=1, 4, 8, 16, 32`
- MaxFlat: `N=1, 4, 8, 16, 32`
- Center-flat variants: `N=1, 4, 8, 16, 32`
- High-probability center-flat: `N=1, 4, 8, 16, 32`

## Postprocess targets

- `make postprocess-distilgpt2-main`
- `make postprocess-main`
- `make postprocess-smoke`

## Qualitative audit files

- `results/summary/comparison_distilgpt2_writingprompts_qualitative_audit.md`
- `results/summary/comparison_gpt2_writingprompts_qualitative_audit.md`

## Small case review files

- `results/analysis/writingprompts_gpt2_small_case_review_report.md`
- `results/analysis/writingprompts_gpt2_small_case_review_summary.csv`
- `results/analysis/writingprompts_gpt2_small_case_review_scored_items.csv`

## Archive instructions

Use:

```bash
bash scripts/package_writingprompts_artifacts.sh
```

Then record checksums with:

```bash
sha256sum artifacts/*.tar.gz > artifacts/SHA256SUMS.txt
```

## Preservation caveat

Raw generations and token logs are intentionally preserved externally rather than committed directly. The repository should carry the summary, note, and analysis artifacts needed for interpretation, while the large generation and token-log directories are archived separately.
