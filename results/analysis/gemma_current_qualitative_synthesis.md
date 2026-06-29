# Gemma Current Output Synthesis

## 1. Scope and caveats
This is a descriptive inventory and heuristic mode analysis over completed Gemma 3 4B-it outputs that are safe to read now. Wrapped10 is complete and is treated as a small sanity check rather than a benchmark. The analysis is not a formal human evaluation.

## 2. Runs analyzed
- optional: completed
- paper: completed
- technical: completed
- 10: completed
- 20: completed

## 3. Raw smoke observations
The raw smoke runs suggest that successor-state selection in a modern instruct model can steer output modes, but the raw WritingPrompts prompt often elicits commentary, title-like metadata, and markup-like states in addition to ordinary prose.

## 4. Wrapped prompt observations, if available
The wrapped prompt runs are complete. The wrapper appears to reduce task drift sharply: the heuristic mode labels do not flag title/character-sheet generation, commentary, markup, or prompt-marker recycling in the completed wrapped10 and wrapped20 outputs. The wrapper therefore looks effective at forcing direct continuation behavior, at least on these small runs.
The wrapped continuations themselves still tend to share a fairly uniform genre register, so the main gain is cleaner task adherence rather than broader stylistic diversity.

## 5. Method-level output-mode patterns
Minflat and maxflat remain the most diagnostic extremes. Centerflat_q50/q75 and highprob_centerflat appear more balanced when prose continues normally, while top_p/top_k remain the simplest baselines. In wrapped10 and wrapped20, all methods stayed in direct prose mode, so the remaining differences are mainly in likelihood/rank behavior and qualitative continuation style rather than obvious format drift. These are heuristic mode labels, not claims of robust format control.
Within the small wrapped10 and wrapped20 tables, centerflat_q50 is the clearest rank outlier, while centerflat_q75 and highprob_centerflat look more likelihood-friendly. That pattern is descriptive only, but it suggests the wrapper does not erase method-level differences.

## 6. Successor dispersion selector behavior from token logs
The token logs indicate that the selector still moves among candidate successor-state shapes in Gemma 3 4B-it. However, the cost is high and the interpretation is currently method-level and descriptive.

## 7. Runtime observations
Gemma 4B-it is much slower than the GPT-2 / DistilGPT-2 experiments. The n-trial methods are especially expensive, and runtime remains the dominant practical concern.

## 8. What appears supported so far
Successor-state shaping appears measurable in the raw smoke runs and remains present under wrapped prompts. The wrapped10 run suggests that the wrapper can suppress task drift while preserving method-level differences in successor-state selection and likelihood-related behavior.

## 9. What is not yet supported
There is no basis yet for robust modern-model claims, general format control claims, or quality improvement claims. The wrapped10 run is a small sanity check, not a benchmark, and should not be generalized beyond its limited scope.

## 10. Recommended next steps after wrapped20 finishes
With wrapped20 complete, repeat the analysis with the complete files only. Compare wrapped outputs against raw smoke outputs to test whether task drift is reduced while method-level successor-state differences remain visible. The completed wrapped10 and wrapped20 runs already support a cautious claim that explicit continuation prompting can reduce drift in this modern instruct-model sanity check.
