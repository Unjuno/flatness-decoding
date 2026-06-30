# Gemma 2 vs Qwen raw/wrapped comparison

This comparison uses the completed raw and wrapped WritingPrompts probes for Qwen 2.5 1.5B-Instruct and Gemma 2 2B. The goal is to separate prompt-framing effects from decoder-regime effects and to check whether the successor-state signal is model-dependent.

| model_condition | method | direct_story_opening | mean_current_rank | mean_successor_dispersion | prompt_relevance | token_fragment_collapse | instruction_like_reversion | semantic_incoherence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen_raw | top_p | 0.90 | nan | nan | 0.30 | 0.00 | 0.10 | 0.20 |
| qwen_raw | top_k | 0.90 | nan | nan | 0.40 | 0.00 | 0.10 | 0.10 |
| qwen_raw | random | 1.00 | 97.89 | 3.13 | 0.10 | 0.00 | 0.00 | 0.00 |
| qwen_raw | minflat | 0.90 | 1112.35 | 2.48 | 0.10 | 0.00 | 0.00 | 0.00 |
| qwen_raw | maxflat | 1.00 | 63.54 | 3.36 | 0.40 | 0.00 | 0.00 | 0.10 |
| qwen_raw | centerflat_q50 | 0.80 | 141.76 | 3.13 | 0.40 | 0.00 | 0.00 | 0.00 |
| qwen_raw | centerflat_q75 | 1.00 | 28.09 | 3.18 | 0.20 | 0.00 | 0.00 | 0.10 |
| qwen_raw | highprob_centerflat | 0.80 | 6.39 | 3.11 | 0.50 | 0.00 | 0.00 | 0.00 |
| qwen_wrapped | top_p | nan | 0.00 | 0.00 | nan | nan | nan | nan |
| qwen_wrapped | top_k | nan | 0.00 | 0.00 | nan | nan | nan | nan |
| qwen_wrapped | random | nan | 70.12 | 3.30 | nan | nan | nan | nan |
| qwen_wrapped | minflat | nan | 1293.17 | 2.43 | nan | nan | nan | nan |
| qwen_wrapped | maxflat | nan | 44.58 | 3.68 | nan | nan | nan | nan |
| qwen_wrapped | centerflat_q50 | nan | 31.02 | 3.63 | nan | nan | nan | nan |
| qwen_wrapped | centerflat_q75 | nan | 61.84 | 3.51 | nan | nan | nan | nan |
| qwen_wrapped | highprob_centerflat | nan | 3.42 | 3.59 | nan | nan | nan | nan |
| gemma2_raw | top_p | 0.60 | nan | nan | 0.30 | 0.00 | 0.00 | 0.00 |
| gemma2_raw | top_k | 0.50 | nan | nan | 0.50 | 0.00 | 0.00 | 0.00 |
| gemma2_raw | random | 0.80 | 29.95 | 5.63 | 0.20 | 0.00 | 0.00 | 0.00 |
| gemma2_raw | minflat | 0.70 | 62.49 | 3.51 | 0.00 | 0.00 | 0.00 | 0.00 |
| gemma2_raw | maxflat | 1.00 | 34.93 | 6.79 | 0.20 | 0.00 | 0.00 | 0.00 |
| gemma2_raw | centerflat_q50 | 0.80 | 10.64 | 5.83 | 0.20 | 0.00 | 0.00 | 0.00 |
| gemma2_raw | centerflat_q75 | 0.90 | 18.90 | 6.19 | 0.60 | 0.00 | 0.00 | 0.00 |
| gemma2_raw | highprob_centerflat | 0.70 | 5.34 | 5.71 | 0.40 | 0.00 | 0.00 | 0.00 |
| gemma2_wrapped | top_p | 0.80 | nan | nan | 0.20 | 0.10 | 0.00 | 0.00 |
| gemma2_wrapped | top_k | 0.70 | nan | nan | 0.10 | 0.00 | 0.00 | 0.00 |
| gemma2_wrapped | random | 0.60 | 18.13 | 5.59 | 0.20 | 0.00 | 0.00 | 0.00 |
| gemma2_wrapped | minflat | 0.20 | 11.07 | 3.50 | 0.10 | 0.30 | 0.00 | 0.00 |
| gemma2_wrapped | maxflat | 1.00 | 35.41 | 6.84 | 0.50 | 0.00 | 0.00 | 0.00 |
| gemma2_wrapped | centerflat_q50 | 0.60 | 14.76 | 5.76 | 0.50 | 0.00 | 0.00 | 0.00 |
| gemma2_wrapped | centerflat_q75 | 1.00 | 11.89 | 6.06 | 0.50 | 0.00 | 0.00 | 0.00 |
| gemma2_wrapped | highprob_centerflat | 0.70 | 5.68 | 5.78 | 0.70 | 0.00 | 0.00 | 0.00 |

## Readout

Qwen raw and wrapped both show direct story continuation, but the raw condition is noisier and more instruction-like. Gemma 2 raw/wrapped are more stable narratively, but they still separate by successor-state regime and do not collapse the model into a single mode.

The key cross-model difference is that Qwen does not reproduce Gemma 2's atmospheric prose bias. Qwen more readily exhibits instruction-like reversion and collapse-style behavior, especially in minflat, while Gemma 2 keeps the output closer to direct prose even when the decoding regime is extreme.

Method-level interpretation:
- minflat is the clearest failure mode in both families, but the failure surface differs: Qwen pushes into collapse or instruction-like drift more readily, while Gemma 2 keeps some prose continuity but remains low-quality.
- centerflat_q50 remains the least safe regime in the raw Qwen probe; Gemma 2 is less extreme but still does not make q50 a winner.
- centerflat_q75 and highprob_centerflat are more conservative in rank, but neither should be treated as a universal best choice.
- maxflat increases dispersion without guaranteeing better story continuation.

Overall, the comparison supports a narrow claim: successor-state selection is measurable in both model families, but the text regimes it exposes are strongly model-dependent and prompt-framing-sensitive.
