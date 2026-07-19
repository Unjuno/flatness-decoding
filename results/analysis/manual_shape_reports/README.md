# Manual shape-by-shape reading reports

These reports are a qualitative reading of the saved continuation text, not a
relabeling of the automated counters. Each report covers the three final
`shapes_20` model runs and 20 prompt IDs per model (60 continuations per
shape). The labels describe the dominant output mode observed in the text.

The review separates narrative continuation from assistant-like meta text,
lists/templates, repetition, and token/format corruption. These are human
qualitative interpretations of saved text, not ground-truth semantic labels.
A shape is not treated as a quality label when the three models disagree.

The row-count and file-by-file audit is recorded in
[`manual_shape_reports_audit.md`](manual_shape_reports_audit.md). It checks the
authoritative JSONL files and records where a statement is only a qualitative
impression rather than a directly verifiable fact.

- [heavy_tail](heavy_tail.md)
- [high_concentration](high_concentration.md)
- [high_spread](high_spread.md)
- [large_gap](large_gap.md)
- [low_concentration](low_concentration.md)
- [low_spread](low_spread.md)
- [mid_concentration](mid_concentration.md)
- [mid_gap](mid_gap.md)
- [mid_spread](mid_spread.md)
- [mid_tail](mid_tail.md)
- [multi_peak](multi_peak.md)
- [rank_slope_flat](rank_slope_flat.md)
- [rank_slope_mid](rank_slope_mid.md)
- [rank_slope_steep](rank_slope_steep.md)
- [small_gap](small_gap.md)
- [smooth_top](smooth_top.md)
- [spiky_top](spiky_top.md)
- [thin_tail](thin_tail.md)
