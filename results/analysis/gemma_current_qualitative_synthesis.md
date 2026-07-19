# Gemma Current Output Synthesis

## 1. Scope and caveats
This is a descriptive inventory and heuristic mode analysis over the completed Gemma 3 4B-it outputs that are safe to read now. The wrapped10 and wrapped20 runs are complete; only the configured N=8 results are in scope. The analysis is not a formal human evaluation.

## 2. Runs analyzed
- optional: completed
- paper: completed
- technical: completed
- 10: completed
- 20: completed

## 3. Raw smoke observations
The raw smoke runs suggest that successor-state selection in a modern instruct model can steer output modes, but the raw WritingPrompts prompt often elicits commentary, title-like metadata, and markup-like states in addition to ordinary prose.

## 4. Wrapped prompt observations
The wrapped10 and wrapped20 outputs are available for descriptive comparison. The wrapper is intended to reduce task drift by forcing direct continuation behavior rather than premise commentary or character-sheet generation. This comparison is not a controlled quality benchmark because raw and wrapped prompt text differ.

## 5. Method-level output-mode patterns
Minflat and maxflat are the most diagnostic extremes. Centerflat_q50/q75 and highprob_centerflat appear more balanced when prose continues normally, while top_p/top_k remain the simplest baselines. These are heuristic mode labels, not claims of robust format control.

## 6. Successor dispersion selector behavior from token logs
The token logs indicate that the selector still moves among candidate successor-state shapes in Gemma 3 4B-it. However, the cost is high and the interpretation is currently method-level and descriptive.

## 7. Runtime observations
Gemma 4B-it is much slower than the GPT-2 / DistilGPT-2 experiments. The n-trial methods are especially expensive, and runtime remains the dominant practical concern.

## 8. What appears supported so far
Successor-state shaping appears measurable in the raw smoke runs and remains at least partly present under the completed wrapped prompts.

## 9. What is not yet supported
There is no basis for robust universal modern-model claims, general format-control claims, or quality-improvement claims. The wrapped results are descriptive and limited to the completed N=8 outputs.

## 10. Possible next steps
If the study is extended, compare raw and wrapped outputs with a fixed paired prompt set and an explicit semantic rubric. Do not pool the framing conditions or interpret the current descriptive comparison as a causal estimate of prompt-format quality.
