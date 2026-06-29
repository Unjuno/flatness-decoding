# Qwen vs Gemma wrapped comparison

Qwen wrapped10 is a non-Gemma sanity check against the Gemma wrapped20 control set. The comparison is critical: it looks for recurrence of the broad successor-state signal, but also for model-specific failure modes.

| method | Qwen rank | Qwen direct_story_opening | Qwen token_fragment_collapse | Qwen multilingual_garbage | Qwen repeated_subword_pattern | Qwen instruction_like_reversion | Qwen semantic_incoherence | Gemma rain_opening | Gemma metal_iron_rust_blood_lexicon | Gemma regret_guilt_shadow_lexicon |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| top_p | 0.00 | 1.00 | 0.0 | 0.0 | 0.8 | 0.0 | 0.8 | 0.65 | 0.45 | 0.3 |
| top_k | 0.00 | 1.00 | 0.0 | 0.0 | 0.8 | 0.0 | 0.5 | 0.7 | 0.45 | 0.4 |
| random | 70.12 | 1.00 | 0.0 | 0.0 | 1.0 | 0.0 | 0.3 | 0.7 | 0.45 | 0.2 |
| minflat | 1293.17 | 0.90 | 0.0 | 0.4 | 0.3 | 0.0 | 0.2 | 0.35 | 0.2 | 0.05 |
| maxflat | 44.58 | 1.00 | 0.0 | 0.0 | 0.9 | 0.0 | 0.7 | 0.55 | 0.3 | 0.05 |
| centerflat_q50 | 31.02 | 0.90 | 0.0 | 0.0 | 0.9 | 0.0 | 0.4 | 0.8 | 0.65 | 0.2 |
| centerflat_q75 | 61.84 | 1.00 | 0.0 | 0.0 | 1.0 | 0.0 | 0.5 | 0.75 | 0.65 | 0.35 |
| highprob_centerflat | 3.42 | 1.00 | 0.0 | 0.0 | 0.9 | 0.0 | 0.2 | 0.8 | 0.7 | 0.2 |

Interpretation: Qwen confirms that successor-state selection remains computable and method-dependent in another instruction-tuned model family. However, it does not reproduce Gemma's rain/metal/regret atmospheric opening bias. Instead, the wrapped Qwen outputs are dominated by direct story openings, instruction-like reversion, and a strong minflat collapse profile. Highprob_centerflat is conservative in rank terms, but it can drift into explanatory or instruction-like phrasing. q75 is not universally likelihood-friendly, and the exposed generation regimes are model-dependent rather than uniform across models.
