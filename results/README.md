# Results

Small, final research artifacts should be tracked by Git.

Recommended layout:

```text
results/
  summary/
    smoke_metrics.csv
    main_metrics.csv
  figures/
    flatness_vs_n.png
    distinct2_vs_n.png
  samples/
    representative_outputs.jsonl
```

Large raw outputs should remain in `outputs/` and be managed by DVC or external storage.

The same policy applies to model-specific run directories directly under `results/`:
they are local source data for post-processing and are ignored by Git. Curated CSV
tables, reports, and figures under `results/analysis/`, `results/summary/`, and
`results/figures/` remain reviewable and reproducible.

To refresh the local inventory after adding or removing runs:

```bash
PYTHONPATH=src python scripts/summarize_artifacts.py
```
