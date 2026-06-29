# Raw vs wrapped critical comparison

This is a framing comparison, not a claim of better generation quality.

- Raw Gemma prompt: more room for task-mode artifacts and prompt-format leakage.
- Wrapped Gemma prompt: visible drift is suppressed, but the continuation style often collapses toward generic atmospheric prose.
- Decoder effect: method-level rank and dispersion differences remain measurable inside the wrapped condition.
- Interpretation: the wrapper reduces one failure mode while exposing another; that is not robust format control.
- Caution: direct_story_opening is only a weak proxy and should not be treated as a success metric.
- Residual bias: the wrapped outputs repeatedly lean on rain / iron / rust / regret openings, which looks like a generic literary prior rather than evidence of improved control.
