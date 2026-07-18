# Gemma Current Output Synthesis

## 1. Scope and caveats
This is a descriptive inventory and heuristic mode analysis over completed Gemma 3 4B-it outputs that are safe to read now. Wrapped10 is currently running/partial and is therefore treated cautiously. The analysis is not a formal human evaluation.

## 2. Runs analyzed
- optional: completed
- paper: completed
- technical: completed
- 10: completed
- 20: completed

## 3. Raw smoke observations
The raw smoke runs suggest that successor-state selection in a modern instruct model can steer output modes, but the raw WritingPrompts prompt often elicits commentary, title-like metadata, and markup-like states in addition to ordinary prose.

## 4. Wrapped prompt observations, if available
The wrapped prompt run is still partial/running, so only limited observations are available. The wrapper is intended to reduce task drift by forcing direct continuation behavior rather than premise commentary or character-sheet generation.

## 5. Method-level output-mode patterns
Minflat and maxflat are the most diagnostic extremes. Centerflat_q50/q75 and highprob_centerflat appear more balanced when prose continues normally, while top_p/top_k remain the simplest baselines. These are heuristic mode labels, not claims of robust format control.

## 6. Successor dispersion selector behavior from token logs
The token logs indicate that the selector still moves among candidate successor-state shapes in Gemma 3 4B-it. However, the cost is high and the interpretation is currently method-level and descriptive.

## 7. Runtime observations
Gemma 4B-it is much slower than the GPT-2 / DistilGPT-2 experiments. The n-trial methods are especially expensive, and runtime remains the dominant practical concern.

## 8. What appears supported so far
Successor-state shaping appears measurable in the raw smoke runs and remains at least partly present under wrapped prompts, although the wrapped run is not yet complete.

## 9. What is not yet supported
There is no basis yet for robust modern-model claims, general format control claims, or quality improvement claims. The wrapped10 run is still partial/running and cannot be used as a finished result.

## 10. Recommended next steps after wrapped20 finishes
Wait for the wrapped runs to complete, then re-run the analysis with complete files only. Compare wrapped outputs against raw smoke outputs to test whether task drift is reduced while method-level successor-state differences remain visible.
