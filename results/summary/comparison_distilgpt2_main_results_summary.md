# comparison_distilgpt2_main Results Summary

## 1. Experiment setup

The completed main experiment evaluates next-state logit flatness as a decoding control signal using `distilgpt2`. The generation set contains 6,300 outputs: 1,800 outputs from standard baselines and 4,500 outputs from candidate-based methods. The baseline methods are `greedy`, `beam`, `sampling`, `temperature`, `top_k`, and `top_p`. The candidate-based methods are `random`, `minflat`, and `maxflat` with `N = 1, 4, 8, 16, 32`.

The official implementation profile for this run is the safe full-prefix candidate evaluation path with candidate micro-batching. KV-cache lookahead is disabled. This is the conservative exact implementation used for the thesis results.

## 2. RQ1: Does Flatness-N control next-state logit flatness?

Yes. The token-level flatness analysis shows a stable ordering for every tested `N > 1`:

```text
MinFlat-N < Random-N < MaxFlat-N
```

Mean chosen next-state logit standard deviation:

| N | MinFlat | Random | MaxFlat |
|---:|---:|---:|---:|
| 1 | 3.252156 | 3.252156 | 3.252156 |
| 4 | 2.349088 | 3.246126 | 3.759246 |
| 8 | 2.029755 | 3.269988 | 3.881724 |
| 16 | 1.881312 | 3.261381 | 4.002425 |
| 32 | 1.825126 | 3.263987 | 4.081533 |

At `N=1`, all three methods are identical because the candidate set contains only one token. For `N=4, 8, 16, 32`, MinFlat consistently selects flatter next states than Random-N, while MaxFlat consistently selects sharper next states. This confirms that the proposed selection rule controls the intended next-state logit flatness signal rather than merely reproducing random candidate selection.

## 3. RQ2: How does flatness control affect generation behavior?

Flatness control changes generation behavior, but it does not monotonically improve surface-level generation quality.

### Random-N remains a stable control

Random-N remains close to ordinary sampling across `N`. Its distinct and repetition metrics vary only modestly as `N` increases. This supports the interpretation that changes in MinFlat-N and MaxFlat-N are due to the flatness-based selection rule rather than the existence of multiple candidates alone.

### MinFlat-N increases some diversity measures but also increases repetition

MinFlat-N tends to increase `distinct_1` relative to Random-N. For example:

| Condition | distinct_1 | repetition_4 |
|---|---:|---:|
| random, N=4 | 0.6182 | 0.0122 |
| minflat, N=4 | 0.7104 | 0.0559 |
| random, N=16 | 0.6057 | 0.0181 |
| minflat, N=16 | 0.6619 | 0.0864 |
| random, N=32 | 0.6088 | 0.0219 |
| minflat, N=32 | 0.6571 | 0.0809 |

This suggests that flatter next-state logits can push the model toward more lexically varied continuations, but the same control does not prevent local repetition. At larger `N`, MinFlat-N increases repetition despite maintaining high token counts.

### MaxFlat-N shortens outputs and increases repetition at high N

MaxFlat-N becomes increasingly sharp as `N` grows, but this does not improve output quality. Mean output tokens decrease substantially, while repetition rises:

| N | mean_output_tokens | repetition_4 |
|---:|---:|---:|
| 1 | 111.2153 | 0.0193 |
| 4 | 95.5529 | 0.0122 |
| 8 | 82.4196 | 0.0300 |
| 16 | 67.1595 | 0.0360 |
| 32 | 58.9770 | 0.0814 |

This indicates that selecting sharper next states can lead to shorter and more repetitive generations.

### Perplexity trade-offs

The merged table includes continuation perplexity. Lower PPL does not align directly with better diversity or lower repetition. Representative PPL values include:

| Method | N | PPL |
|---|---:|---:|
| beam | - | 1.9478 |
| greedy | - | 2.4736 |
| sampling | - | 51.7209 |
| top_k | - | 19.5434 |
| top_p | - | 137.1905 |
| temperature | - | 499.1778 |
| random | 1 | 49.3876 |
| minflat | 4 | 192.3889 |
| minflat | 16 | 215.4207 |
| maxflat | 16 | 70.2223 |
| maxflat | 32 | 84.1074 |

Greedy and beam have low PPL but very high repetition. Temperature has high lexical diversity and very low repetition, but very high PPL. MinFlat-N has higher PPL than Random-N at larger `N`, even though it successfully controls flatness. This reinforces the conclusion that next-state flatness is a behavioral control signal, not a direct quality objective.

## 4. RQ3: How does N affect control strength and cost?

Increasing `N` strengthens the flatness control effect. MinFlat-N finds flatter candidates as `N` increases, and MaxFlat-N finds sharper candidates. However, runtime increases substantially.

For MinFlat-N:

| N | mean_chosen_flatness | mean_runtime_seconds |
|---:|---:|---:|
| 1 | 3.252156 | 5.7978 |
| 4 | 2.349088 | 11.4343 |
| 8 | 2.029755 | 20.3881 |
| 16 | 1.881312 | 35.8344 |
| 32 | 1.825126 | 65.5751 |

The marginal flatness improvement from `N=16` to `N=32` is small compared with the runtime increase. Therefore, `N=8` or `N=16` appears to be a more practical operating point, while `N=32` is useful as an upper-bound stress test.

## 5. Main thesis-ready interpretation

### English

Flatness-N successfully controls next-state logit flatness during decoding. Across all tested values of `N > 1`, MinFlat-N selects candidates with lower next-state logit standard deviation than Random-N, while MaxFlat-N selects candidates with higher standard deviation. This demonstrates that next-state logit flatness is a controllable decoding signal.

However, flatness control does not translate into monotonic quality improvement. MinFlat-N can increase lexical diversity, but it also increases repetition at larger `N`. MaxFlat-N tends to shorten outputs and increase repetition as `N` grows. These results suggest that next-state logit flatness is useful for controlling generation behavior, but it should not be treated as a standalone quality objective.

### Japanese

Flatness-N は，デコード時に next-state logit flatness を明示的に制御できることが確認された。`N > 1` のすべての条件で，MinFlat-N は Random-N より低い next-state logit standard deviation を持つ候補を選択し，MaxFlat-N はより高い standard deviation を持つ候補を選択した。この結果は，next-state logit flatness が生成時に制御可能な信号であることを示している。

一方で，flatness の制御は生成品質の単調な改善には直結しなかった。MinFlat-N は distinct-1 など一部の多様性指標を高めるが，N が大きい条件では repetition も増加した。MaxFlat-N は N が大きくなるほど出力が短くなり，反復が増える傾向を示した。したがって，next-state logit flatness は生成挙動を制御する有効な信号ではあるが，単独の品質目的関数として扱うには限界がある。

## 6. Limitations

1. The current main result is from `distilgpt2`; a second model should be used to confirm generality.
2. Surface metrics show trade-offs rather than a single quality improvement.
3. Flatness is computed from model logits, so it reflects the model's learned continuation landscape rather than an external semantic quality measure.
4. Candidate evaluation is computationally expensive because the official implementation uses full-prefix candidate evaluation.
5. High `N` improves control strength but sharply reduces throughput.

## 7. Recommended follow-up experiments

### Qualitative examples

Extract matched examples for the same prompt and seed across:

- `top_p`
- `temperature`
- `random N=16`
- `minflat N=16`
- `maxflat N=16`

This should be done before running another large experiment, because the current metrics suggest important qualitative trade-offs.

### GPT-2 main-lite

Run a smaller confirmation experiment on `gpt2`:

- model: `gpt2`
- prompts: 30--50
- seeds: 0, 1
- N: 1, 4, 8, 16
- methods: `random`, `minflat`, `maxflat`, plus standard baselines
- workers: 1
- candidate_eval_batch_size: 1
- use_kv_cache_lookahead: false

The target question is whether the ordering `MinFlat < Random < MaxFlat` appears beyond `distilgpt2`.

## 8. Bottom line

The experiment supports the core research claim: Flatness-N is an effective mechanism for controlling next-state logit flatness. The method should be presented as a decoding control framework, not as a guaranteed quality-improvement method.
