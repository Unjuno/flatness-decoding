# Early Pilot Flatness Reanalysis

## Executive Summary
- The WikiText-derived pilot did control the flatness signal as intended: `minflat` lowered chosen flatness and `maxflat` raised it relative to `random` for every experiment at `n_trials > 1`.
- The control got stronger as `n_trials` increased, but so did runtime; throughput fell accordingly.
- Under WikiText-style continuation, `minflat` was usually the least usable setting once `n_trials > 1`: outputs often became symbol-heavy, repetitive, or otherwise broken.
- `maxflat` was less broken than `minflat` and often still readable, but it tended to be diffuse or off-target rather than clearly better.
- `n_trials = 1` is not informative for control because `random`, `minflat`, and `maxflat` are identical there.

## File / Schema Audit
| file | status | rows | experiment | model | methods | n_trials | notes |
|---|---:|---:|---|---|---|---|---|
| `comparison_gpt2_lite_all_table.csv` | exists | 18 | `comparison_gpt2_lite` | `gpt2` | beam, greedy, maxflat, minflat, random, sampling, temperature, top_k, top_p | 1.0, 4.0, 8.0, 16.0 |  |
| `comparison_gpt2_lite_token_trace.csv` | exists | 12 | `comparison_gpt2_lite` | `` | maxflat, minflat, random | 1, 4, 8, 16 |  |
| `comparison_gpt2_medium_micro_all_table.csv` | exists | 12 | `comparison_gpt2_medium_micro` | `gpt2-medium` | maxflat, minflat, random, sampling, temperature, top_p | 1.0, 4.0, 16.0 |  |
| `comparison_gpt2_medium_micro_token_trace.csv` | exists | 9 | `comparison_gpt2_medium_micro` | `` | maxflat, minflat, random | 1, 4, 16 |  |
| `comparison_qwen25_05b_micro_all_table.csv` | exists | 12 | `comparison_qwen25_05b_micro` | `Qwen/Qwen2.5-0.5B` | maxflat, minflat, random, sampling, temperature, top_p | 1.0, 4.0, 16.0 | PPL missing by design |
| `comparison_qwen25_05b_micro_token_trace.csv` | exists | 9 | `comparison_qwen25_05b_micro` | `` | maxflat, minflat, random | 1, 4, 16 |  |

| generation dir | files | total generations | file format | sample fields |
|---|---:|---:|---|---|
| `outputs/generations/comparison_gpt2_lite` | 144 | 1440 | JSONL | prompt, continuation_text, method, n_trials, seed, model, prompt_id, runtime_seconds |
| `outputs/generations/comparison_gpt2_medium_micro` | 24 | 120 | JSONL | prompt, continuation_text, method, n_trials, seed, model, prompt_id, runtime_seconds |
| `outputs/generations/comparison_qwen25_05b_micro` | 24 | 120 | JSONL | prompt, continuation_text, method, n_trials, seed, model, prompt_id, runtime_seconds |

## Numeric Findings
### Random-relative deltas (selected)
| experiment | model | method | n_trials | delta_flatness | delta_distinct_2 | delta_repetition_4 | delta_runtime | delta_tokens_per_second | delta_ppl | delta_mean_nll |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `comparison_gpt2_lite` | `gpt2` | `maxflat` | 1 | 0.000 | 0.000 | 0.000 | -0.264 | 0.533 | 0.000 | 0.000 |
| `comparison_gpt2_lite` | `gpt2` | `maxflat` | 4 | 0.591 | -0.010 | -0.001 | -0.174 | 0.119 | -31.337 | -0.585 |
| `comparison_gpt2_lite` | `gpt2` | `maxflat` | 8 | 0.829 | -0.012 | -0.000 | -2.649 | 0.126 | -21.891 | -0.363 |
| `comparison_gpt2_lite` | `gpt2` | `maxflat` | 16 | 0.850 | -0.021 | 0.003 | -2.955 | 0.058 | -13.682 | -0.195 |
| `comparison_gpt2_lite` | `gpt2` | `minflat` | 1 | 0.000 | 0.000 | 0.000 | -0.244 | 0.534 | 0.000 | 0.000 |
| `comparison_gpt2_lite` | `gpt2` | `minflat` | 4 | -1.127 | -0.008 | 0.022 | 0.376 | 0.043 | 388.012 | 1.773 |
| `comparison_gpt2_lite` | `gpt2` | `minflat` | 8 | -1.284 | -0.085 | 0.061 | 0.605 | -0.002 | 273.372 | 1.408 |
| `comparison_gpt2_lite` | `gpt2` | `minflat` | 16 | -1.417 | -0.134 | 0.098 | 1.867 | -0.033 | 133.948 | 0.974 |
| `comparison_gpt2_medium_micro` | `gpt2-medium` | `maxflat` | 1 | 0.000 | 0.000 | 0.000 | -0.495 | 0.599 | 0.000 | 0.000 |
| `comparison_gpt2_medium_micro` | `gpt2-medium` | `maxflat` | 4 | 0.467 | -0.017 | 0.000 | 1.362 | -0.027 | 9.208 | 0.241 |
| `comparison_gpt2_medium_micro` | `gpt2-medium` | `maxflat` | 16 | 0.770 | -0.019 | 0.000 | -0.093 | 0.003 | 10.454 | 0.160 |
| `comparison_gpt2_medium_micro` | `gpt2-medium` | `minflat` | 1 | 0.000 | 0.000 | 0.000 | -0.520 | 0.640 | 0.000 | 0.000 |
| `comparison_gpt2_medium_micro` | `gpt2-medium` | `minflat` | 4 | -0.808 | -0.067 | 0.007 | 1.415 | -0.049 | 112.795 | 1.454 |
| `comparison_gpt2_medium_micro` | `gpt2-medium` | `minflat` | 16 | -1.299 | -0.154 | 0.138 | 0.114 | -0.004 | 182.180 | 1.275 |
| `comparison_qwen25_05b_micro` | `Qwen/Qwen2.5-0.5B` | `maxflat` | 1 | 0.000 | 0.000 | 0.000 | -0.316 | 0.264 | MISSING_BY_DESIGN | MISSING_BY_DESIGN |
| `comparison_qwen25_05b_micro` | `Qwen/Qwen2.5-0.5B` | `maxflat` | 4 | 0.204 | 0.012 | -0.005 | -0.006 | 0.001 | MISSING_BY_DESIGN | MISSING_BY_DESIGN |
| `comparison_qwen25_05b_micro` | `Qwen/Qwen2.5-0.5B` | `maxflat` | 16 | 0.267 | 0.038 | -0.000 | -0.997 | 0.002 | MISSING_BY_DESIGN | MISSING_BY_DESIGN |
| `comparison_qwen25_05b_micro` | `Qwen/Qwen2.5-0.5B` | `minflat` | 1 | 0.000 | 0.000 | 0.000 | -0.302 | 0.256 | MISSING_BY_DESIGN | MISSING_BY_DESIGN |
| `comparison_qwen25_05b_micro` | `Qwen/Qwen2.5-0.5B` | `minflat` | 4 | -0.461 | -0.027 | 0.007 | -0.027 | 0.004 | MISSING_BY_DESIGN | MISSING_BY_DESIGN |
| `comparison_qwen25_05b_micro` | `Qwen/Qwen2.5-0.5B` | `minflat` | 16 | -0.774 | 0.043 | -0.005 | -0.065 | 0.001 | MISSING_BY_DESIGN | MISSING_BY_DESIGN |

### Best by metric
| experiment | metric | direction | method | n_trials | value |
|---|---|---|---|---:|---:|
| `comparison_gpt2_lite` | `mean_chosen_flatness` | `lowest` | `minflat` | 16 | 1.7734957705189784 |
| `comparison_gpt2_lite` | `mean_chosen_flatness` | `highest` | `maxflat` | 16 | 4.04055156422244 |
| `comparison_gpt2_lite` | `ppl` | `lowest` | `maxflat` | 4 | 40.07949385659999 |
| `comparison_gpt2_lite` | `ppl` | `highest` | `minflat` | 4 | 459.4282246381281 |
| `comparison_gpt2_lite` | `mean_nll` | `lowest` | `maxflat` | 4 | 3.6746258107359644 |
| `comparison_gpt2_lite` | `mean_nll` | `highest` | `minflat` | 4 | 6.031885423883588 |
| `comparison_gpt2_lite` | `distinct_2` | `highest` | `random` | 4 | 0.9621427358633544 |
| `comparison_gpt2_lite` | `distinct_2` | `lowest` | `minflat` | 16 | 0.8246135786607633 |
| `comparison_gpt2_lite` | `repetition_4` | `lowest` | `random` | 16 | 0.0029269092309769 |
| `comparison_gpt2_lite` | `repetition_4` | `highest` | `minflat` | 16 | 0.1008572255408238 |
| `comparison_gpt2_lite` | `mean_runtime_seconds` | `lowest` | `maxflat` | 1 | 7.073580961526022 |
| `comparison_gpt2_lite` | `mean_runtime_seconds` | `highest` | `minflat` | 16 | 44.87452846509695 |
| `comparison_gpt2_lite` | `mean_tokens_per_second` | `highest` | `minflat` | 1 | 17.895427037030814 |
| `comparison_gpt2_lite` | `mean_tokens_per_second` | `lowest` | `minflat` | 16 | 2.686024732083297 |
| `comparison_gpt2_medium_micro` | `mean_chosen_flatness` | `lowest` | `minflat` | 16 | 1.85580859998862 |
| `comparison_gpt2_medium_micro` | `mean_chosen_flatness` | `highest` | `maxflat` | 16 | 3.923874129454295 |
| `comparison_gpt2_medium_micro` | `ppl` | `lowest` | `random` | 4 | 34.59698021748697 |
| `comparison_gpt2_medium_micro` | `ppl` | `highest` | `minflat` | 16 | 251.72567544684844 |
| `comparison_gpt2_medium_micro` | `mean_nll` | `lowest` | `random` | 4 | 3.5385629442871593 |
| `comparison_gpt2_medium_micro` | `mean_nll` | `highest` | `minflat` | 16 | 5.403614974180517 |
| `comparison_gpt2_medium_micro` | `distinct_2` | `highest` | `random` | 4 | 0.9954954954954957 |
| `comparison_gpt2_medium_micro` | `distinct_2` | `lowest` | `minflat` | 16 | 0.8325417052826691 |
| `comparison_gpt2_medium_micro` | `repetition_4` | `lowest` | `maxflat` | 1 | 0.0 |
| `comparison_gpt2_medium_micro` | `repetition_4` | `highest` | `minflat` | 16 | 0.1378205128205128 |
| `comparison_gpt2_medium_micro` | `mean_runtime_seconds` | `lowest` | `minflat` | 1 | 6.512158612487838 |
| `comparison_gpt2_medium_micro` | `mean_runtime_seconds` | `highest` | `minflat` | 16 | 42.88628756668186 |
| `comparison_gpt2_medium_micro` | `mean_tokens_per_second` | `highest` | `minflat` | 1 | 9.538527360864578 |
| `comparison_gpt2_medium_micro` | `mean_tokens_per_second` | `lowest` | `minflat` | 16 | 1.3996964201693571 |
| `comparison_qwen25_05b_micro` | `mean_chosen_flatness` | `lowest` | `minflat` | 16 | 2.2287179478009542 |
| `comparison_qwen25_05b_micro` | `mean_chosen_flatness` | `highest` | `maxflat` | 16 | 3.269622831328495 |
| `comparison_qwen25_05b_micro` | `distinct_2` | `highest` | `minflat` | 16 | 0.9848484848484848 |
| `comparison_qwen25_05b_micro` | `distinct_2` | `lowest` | `minflat` | 4 | 0.9367724867724868 |
| `comparison_qwen25_05b_micro` | `repetition_4` | `lowest` | `minflat` | 16 | 0.0 |
| `comparison_qwen25_05b_micro` | `repetition_4` | `highest` | `minflat` | 4 | 0.016 |
| `comparison_qwen25_05b_micro` | `mean_runtime_seconds` | `lowest` | `maxflat` | 1 | 8.07694466658868 |
| `comparison_qwen25_05b_micro` | `mean_runtime_seconds` | `highest` | `random` | 16 | 62.97969820834696 |
| `comparison_qwen25_05b_micro` | `mean_tokens_per_second` | `highest` | `maxflat` | 1 | 7.436656766460945 |
| `comparison_qwen25_05b_micro` | `mean_tokens_per_second` | `lowest` | `random` | 16 | 0.9537800448280628 |

### N-trend classification
| experiment | method | metric | trend_with_N |
|---|---|---|---|
| `comparison_gpt2_lite` | `random` | `mean_chosen_flatness` | `mixed` |
| `comparison_gpt2_lite` | `random` | `mean_selection_gain` | `increasing` |
| `comparison_gpt2_lite` | `random` | `distinct_2` | `mixed` |
| `comparison_gpt2_lite` | `random` | `repetition_4` | `mixed` |
| `comparison_gpt2_lite` | `random` | `mean_runtime_seconds` | `increasing` |
| `comparison_gpt2_lite` | `random` | `mean_tokens_per_second` | `decreasing` |
| `comparison_gpt2_lite` | `minflat` | `mean_chosen_flatness` | `decreasing` |
| `comparison_gpt2_lite` | `minflat` | `mean_selection_gain` | `increasing` |
| `comparison_gpt2_lite` | `minflat` | `distinct_2` | `decreasing` |
| `comparison_gpt2_lite` | `minflat` | `repetition_4` | `increasing` |
| `comparison_gpt2_lite` | `minflat` | `mean_runtime_seconds` | `increasing` |
| `comparison_gpt2_lite` | `minflat` | `mean_tokens_per_second` | `decreasing` |
| `comparison_gpt2_lite` | `maxflat` | `mean_chosen_flatness` | `increasing` |
| `comparison_gpt2_lite` | `maxflat` | `mean_selection_gain` | `increasing` |
| `comparison_gpt2_lite` | `maxflat` | `distinct_2` | `decreasing` |
| `comparison_gpt2_lite` | `maxflat` | `repetition_4` | `mixed` |
| `comparison_gpt2_lite` | `maxflat` | `mean_runtime_seconds` | `increasing` |
| `comparison_gpt2_lite` | `maxflat` | `mean_tokens_per_second` | `decreasing` |
| `comparison_gpt2_medium_micro` | `random` | `mean_chosen_flatness` | `decreasing` |
| `comparison_gpt2_medium_micro` | `random` | `mean_selection_gain` | `increasing` |
| `comparison_gpt2_medium_micro` | `random` | `distinct_2` | `mixed` |
| `comparison_gpt2_medium_micro` | `random` | `repetition_4` | `flat` |
| `comparison_gpt2_medium_micro` | `random` | `mean_runtime_seconds` | `increasing` |
| `comparison_gpt2_medium_micro` | `random` | `mean_tokens_per_second` | `decreasing` |
| `comparison_gpt2_medium_micro` | `minflat` | `mean_chosen_flatness` | `decreasing` |
| `comparison_gpt2_medium_micro` | `minflat` | `mean_selection_gain` | `increasing` |
| `comparison_gpt2_medium_micro` | `minflat` | `distinct_2` | `decreasing` |
| `comparison_gpt2_medium_micro` | `minflat` | `repetition_4` | `increasing` |
| `comparison_gpt2_medium_micro` | `minflat` | `mean_runtime_seconds` | `increasing` |
| `comparison_gpt2_medium_micro` | `minflat` | `mean_tokens_per_second` | `decreasing` |
| `comparison_gpt2_medium_micro` | `maxflat` | `mean_chosen_flatness` | `increasing` |
| `comparison_gpt2_medium_micro` | `maxflat` | `mean_selection_gain` | `increasing` |
| `comparison_gpt2_medium_micro` | `maxflat` | `distinct_2` | `decreasing` |
| `comparison_gpt2_medium_micro` | `maxflat` | `repetition_4` | `flat` |
| `comparison_gpt2_medium_micro` | `maxflat` | `mean_runtime_seconds` | `increasing` |
| `comparison_gpt2_medium_micro` | `maxflat` | `mean_tokens_per_second` | `decreasing` |
| `comparison_qwen25_05b_micro` | `random` | `mean_chosen_flatness` | `mixed` |
| `comparison_qwen25_05b_micro` | `random` | `mean_selection_gain` | `increasing` |
| `comparison_qwen25_05b_micro` | `random` | `distinct_2` | `mixed` |
| `comparison_qwen25_05b_micro` | `random` | `repetition_4` | `mixed` |
| `comparison_qwen25_05b_micro` | `random` | `mean_runtime_seconds` | `increasing` |
| `comparison_qwen25_05b_micro` | `random` | `mean_tokens_per_second` | `decreasing` |
| `comparison_qwen25_05b_micro` | `minflat` | `mean_chosen_flatness` | `decreasing` |
| `comparison_qwen25_05b_micro` | `minflat` | `mean_selection_gain` | `increasing` |
| `comparison_qwen25_05b_micro` | `minflat` | `distinct_2` | `mixed` |
| `comparison_qwen25_05b_micro` | `minflat` | `repetition_4` | `mixed` |
| `comparison_qwen25_05b_micro` | `minflat` | `mean_runtime_seconds` | `increasing` |
| `comparison_qwen25_05b_micro` | `minflat` | `mean_tokens_per_second` | `decreasing` |
| `comparison_qwen25_05b_micro` | `maxflat` | `mean_chosen_flatness` | `increasing` |
| `comparison_qwen25_05b_micro` | `maxflat` | `mean_selection_gain` | `increasing` |
| `comparison_qwen25_05b_micro` | `maxflat` | `distinct_2` | `increasing` |
| `comparison_qwen25_05b_micro` | `maxflat` | `repetition_4` | `mixed` |
| `comparison_qwen25_05b_micro` | `maxflat` | `mean_runtime_seconds` | `increasing` |
| `comparison_qwen25_05b_micro` | `maxflat` | `mean_tokens_per_second` | `decreasing` |

## Qualitative Examples
| experiment | prompt_id | method | n_trials | label | excerpt |
|---|---|---|---:|---|---|
| `comparison_gpt2_lite` | `main_0000` | `maxflat` | 16 | `mostly_natural` | Sloane. Sloane repeatedly falsely called security agents while they were verbally groping and bajumping his ol... |
| `comparison_gpt2_lite` | `main_0000` | `maxflat` | 1 | `natural` | Gentry, who arrived at the park " at about 10 the following morning."  Pinkydenlooper provided several other c... |
| `comparison_gpt2_lite` | `main_0000` | `maxflat` | 8 | `mostly_natural` | Slivscov who wasn't completely comfortable. He doesn't really have complete control and doesn't fully understa... |
| `comparison_gpt2_lite` | `main_0000` | `minflat` | 16 | `degenerate` | Green butselurder "/" www.facebook.com/databaseinfo4wiredlnp penchant \ • *[] s -p"^ion'n''nothy*erions /the'r... |
| `comparison_gpt2_lite` | `main_0000` | `minflat` | 1 | `natural` | Gentry, who arrived at the park " at about 10 the following morning."  Pinkydenlooper provided several other c... |
| `comparison_gpt2_lite` | `main_0000` | `minflat` | 8 | `degenerate` | Crossas to cast votes, deema^linw ssir 99CEPAT 229 ^ltehire«_a^e\ 'aut- endine •^();' 'il. de 'ire. poll' sor.... |
| `comparison_gpt2_lite` | `main_0000` | `random` | 16 | `mostly_natural` | Giles—who appears to have enough awareness of too much body rubbed-on Stu, he used his speed and shown great c... |
| `comparison_gpt2_lite` | `main_0000` | `random` | 1 | `natural` | Gentry, who arrived at the park " at about 10 the following morning."  Pinkydenlooper provided several other c... |

## Metric-vs-Output Consistency
| experiment | method | n_trials | numeric_signal | qualitative_signal | consistent |
|---|---|---:|---|---|---|
| `comparison_gpt2_lite` | `random` | 1 | baseline or control condition | mostly natural at n=1; later n values readable but increasingly drift-prone | yes |
| `comparison_gpt2_lite` | `minflat` | 1 | identical to random at n=1 | identical to random at n=1 | yes |
| `comparison_gpt2_lite` | `maxflat` | 1 | identical to random at n=1 | identical to random at n=1 | yes |
| `comparison_gpt2_lite` | `random` | 8 | baseline or control condition | mostly natural at n=1; later n values readable but increasingly drift-prone | yes |
| `comparison_gpt2_lite` | `minflat` | 8 | flatness lower than random; diversity lower; repetition higher | visibly broken or garbled | yes |
| `comparison_gpt2_lite` | `maxflat` | 8 | flatness higher than random; diversity slightly lower or mixed; repetition usually lower or flat | mostly natural but more diffuse / off-target | mostly |
| `comparison_gpt2_lite` | `random` | 16 | baseline or control condition | mostly natural at n=1; later n values readable but increasingly drift-prone | yes |
| `comparison_gpt2_lite` | `minflat` | 16 | flatness lower than random; diversity lower; repetition higher | visibly broken or garbled | yes |
| `comparison_gpt2_lite` | `maxflat` | 16 | flatness higher than random; diversity slightly lower or mixed; repetition usually lower or flat | mostly natural but more diffuse / off-target | mostly |
| `comparison_gpt2_medium_micro` | `random` | 1 | baseline or control condition | mostly natural at n=1; later n values readable but increasingly drift-prone | yes |
| `comparison_gpt2_medium_micro` | `minflat` | 1 | identical to random at n=1 | identical to random at n=1 | yes |
| `comparison_gpt2_medium_micro` | `maxflat` | 1 | identical to random at n=1 | identical to random at n=1 | yes |
| `comparison_gpt2_medium_micro` | `random` | 4 | baseline or control condition | mostly natural at n=1; later n values readable but increasingly drift-prone | yes |
| `comparison_gpt2_medium_micro` | `minflat` | 4 | flatness lower than random; diversity lower; repetition higher | visibly broken or garbled | yes |
| `comparison_gpt2_medium_micro` | `maxflat` | 4 | flatness higher than random; diversity slightly lower or mixed; repetition usually lower or flat | mostly natural but more diffuse / off-target | mostly |
| `comparison_gpt2_medium_micro` | `random` | 16 | baseline or control condition | mostly natural at n=1; later n values readable but increasingly drift-prone | yes |
| `comparison_gpt2_medium_micro` | `minflat` | 16 | flatness lower than random; diversity lower; repetition higher | visibly broken or garbled | yes |
| `comparison_gpt2_medium_micro` | `maxflat` | 16 | flatness higher than random; diversity slightly lower or mixed; repetition usually lower or flat | mostly natural but more diffuse / off-target | mostly |
| `comparison_qwen25_05b_micro` | `random` | 1 | baseline or control condition | mostly natural at n=1; later n values readable but increasingly drift-prone | yes |
| `comparison_qwen25_05b_micro` | `minflat` | 1 | identical to random at n=1 | identical to random at n=1 | yes |
| `comparison_qwen25_05b_micro` | `maxflat` | 1 | identical to random at n=1 | identical to random at n=1 | yes |
| `comparison_qwen25_05b_micro` | `random` | 16 | baseline or control condition | mostly natural at n=1; later n values readable but increasingly drift-prone | yes |
| `comparison_qwen25_05b_micro` | `minflat` | 16 | flatness lower than random; diversity mixed; repetition not always higher | visibly broken or garbled | yes |
| `comparison_qwen25_05b_micro` | `maxflat` | 16 | flatness higher than random; diversity slightly lower or mixed; repetition usually lower or flat | mostly natural but more diffuse / off-target | mostly |

## Safe Claims
- The pilot established that the selector can move the next-state flatness signal in both directions.
- Increasing `n_trials` strengthened the control effect but increased runtime and reduced throughput.
- Under the WikiText-derived pilot, `minflat` often produced visibly degenerate outputs at `n_trials > 1`.
- `maxflat` was generally less degenerate than `minflat`, but not clearly better than `random` in a quality sense.

## Claims to Avoid
- Do not claim flatness decoding generally improves generation quality.
- Do not generalize WikiText pilot behavior to WritingPrompts or to creative generation.
- Do not claim `minflat` is universally useful; the pilot often shows the opposite.
- Do not use `n_trials = 1` as evidence of control because the methods are identical there.

## Recommended Next Experiment
- Re-run the comparison on `prompts/writingprompts_fixed_100.jsonl` with the newer selector family: `centerflat_q25`, `centerflat_q50`, `centerflat_q75`, and `highprob_centerflat`.
- Add current logprob / rank logging to separate flatness control from low-probability token selection.
- Use matched-prompt qualitative inspection on the same prompt ids across methods.

## Limitations
- WikiText-derived pilot only.
- Prompt provenance was weak in the early pilot.
- No center-flat selectors in the early pilot.
- No current-rank/current-logprob logging in the early pilot.
- Small prompt/seed counts, especially for the micro experiments.
- Qualitative inspection here is informal and not a blinded human evaluation.
- Qwen PPL/NLL is missing by design.