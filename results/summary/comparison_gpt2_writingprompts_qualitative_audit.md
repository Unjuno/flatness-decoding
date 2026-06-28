# Qualitative Audit

LLM-assisted qualitative audit, not formal human evaluation.

Sampled prompt ids: 10
Sampled continuations: 1380

Main finding: whitespace-only / truncated continuations and repetitive generic continuations appear concentrated in minflat, maxflat, and some N=1 random rows; centerflat_q50/q75 and highprob_centerflat are the safest trade-offs among the n-trial methods.

Method-specific notes: baseline greedy/beam are fluent but conservative; sampling/top_p/top_k are mixed but mostly usable; minflat is the riskiest for off-distribution or abrupt outputs; maxflat tends to be diffuse or generic; centerflat_q25 is below q50/q75; highprob_centerflat is the most conservative n-trial choice and often safest but sometimes generic.

N-trials: N=32 usually improves over N=8 on flatness-oriented metrics, but the qualitative gain is modest and sometimes not worth the runtime.

Safe paper-language paragraph (EN): The qualitative audit suggests that centerflat_q50, centerflat_q75, and highprob_centerflat are the most reliable n-trial settings on the WritingPrompts benchmark, while minflat and some maxflat outputs more often show truncation, repetition, or weak prompt adherence. These observations are exploratory and do not constitute formal human evaluation.

安全な論文用文章（JP）: WritingPrompts に対する定性的監査では、centerflat_q50、centerflat_q75、highprob_centerflat が n-trial 設定の中では比較的安定しており、minflat や一部の maxflat 出力では切断、反復、あるいはプロンプト追従の弱さが目立ちました。これは探索的な LLM 支援の監査であり、正式な人手評価ではありません。
