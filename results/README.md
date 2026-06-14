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
