# token log特徴から本文出力モードを予測する分類解析

保存済みprompt単位データに対して、token logの連続特徴だけで反復・形式崩れを予測できるかを評価した。DecisionTreeは解釈用の浅いモデルで、prompt_idをGroupKFoldのgroupにして同一promptの漏洩を避けた。

## gemma / repeat_target

- n=360, positive_rate=0.028
- GroupKFold balanced accuracy=0.697 ± 0.208

```text
|--- mean_chosen_prob <= 0.63
|   |--- mean_chosen_flatness <= 3.57
|   |   |--- mean_chosen_prob <= 0.31
|   |   |   |--- class: 0
|   |   |--- mean_chosen_prob >  0.31
|   |   |   |--- class: 1
|   |--- mean_chosen_flatness >  3.57
|   |   |--- class: 0
|--- mean_chosen_prob >  0.63
|   |--- mean_chosen_flatness <= 4.34
|   |   |--- mean_chosen_flatness <= 3.95
|   |   |   |--- class: 1
|   |   |--- mean_chosen_flatness >  3.95
|   |   |   |--- class: 1
|   |--- mean_chosen_flatness >  4.34
|   |   |--- mean_chosen_prob <= 0.66
|   |   |   |--- class: 0
|   |   |--- mean_chosen_prob >  0.66
|   |   |   |--- class: 0

```

## gemma / format_target

- n=360, positive_rate=0.403
- GroupKFold balanced accuracy=0.538 ± 0.067

```text
|--- mean_chosen_prob <= 0.39
|   |--- mean_chosen_flatness <= 4.06
|   |   |--- mean_chosen_prob <= 0.29
|   |   |   |--- class: 0
|   |   |--- mean_chosen_prob >  0.29
|   |   |   |--- class: 1
|   |--- mean_chosen_flatness >  4.06
|   |   |--- class: 0
|--- mean_chosen_prob >  0.39
|   |--- mean_chosen_flatness <= 4.50
|   |   |--- fallback_rate <= 0.53
|   |   |   |--- class: 0
|   |   |--- fallback_rate >  0.53
|   |   |   |--- class: 1
|   |--- mean_chosen_flatness >  4.50
|   |   |--- fallback_rate <= 0.15
|   |   |   |--- class: 1
|   |   |--- fallback_rate >  0.15
|   |   |   |--- class: 1

```

## gpt2 / repeat_target

- n=360, positive_rate=0.117
- GroupKFold balanced accuracy=0.841 ± 0.075

```text
|--- mean_chosen_prob <= 0.31
|   |--- mean_chosen_prob <= 0.20
|   |   |--- class: 0
|   |--- mean_chosen_prob >  0.20
|   |   |--- mean_chosen_flatness <= 2.85
|   |   |   |--- class: 1
|   |   |--- mean_chosen_flatness >  2.85
|   |   |   |--- class: 0
|--- mean_chosen_prob >  0.31
|   |--- mean_chosen_rank <= 280.34
|   |   |--- mean_chosen_prob <= 0.43
|   |   |   |--- class: 1
|   |   |--- mean_chosen_prob >  0.43
|   |   |   |--- class: 1
|   |--- mean_chosen_rank >  280.34
|   |   |--- mean_chosen_flatness <= 3.17
|   |   |   |--- class: 0
|   |   |--- mean_chosen_flatness >  3.17
|   |   |   |--- class: 1

```

## gpt2 / format_target

- n=360, positive_rate=0.158
- GroupKFold balanced accuracy=0.584 ± 0.105

```text
|--- mean_selection_gain <= 0.15
|   |--- fallback_rate <= 0.30
|   |   |--- mean_selection_gain <= 0.14
|   |   |   |--- class: 1
|   |   |--- mean_selection_gain >  0.14
|   |   |   |--- class: 1
|   |--- fallback_rate >  0.30
|   |   |--- mean_chosen_rank <= 237.22
|   |   |   |--- class: 1
|   |   |--- mean_chosen_rank >  237.22
|   |   |   |--- class: 0
|--- mean_selection_gain >  0.15
|   |--- mean_chosen_flatness <= 2.71
|   |   |--- class: 0
|   |--- mean_chosen_flatness >  2.71
|   |   |--- mean_chosen_prob <= 0.09
|   |   |   |--- class: 0
|   |   |--- mean_chosen_prob >  0.09
|   |   |   |--- class: 0

```

## qwen / repeat_target

- n=360, positive_rate=0.097
- GroupKFold balanced accuracy=0.771 ± 0.046

```text
|--- mean_chosen_prob <= 0.27
|   |--- mean_chosen_prob <= 0.18
|   |   |--- class: 0
|   |--- mean_chosen_prob >  0.18
|   |   |--- mean_chosen_rank <= 235.80
|   |   |   |--- class: 0
|   |   |--- mean_chosen_rank >  235.80
|   |   |   |--- class: 1
|--- mean_chosen_prob >  0.27
|   |--- mean_chosen_prob <= 0.55
|   |   |--- mean_chosen_flatness <= 3.24
|   |   |   |--- class: 1
|   |   |--- mean_chosen_flatness >  3.24
|   |   |   |--- class: 0
|   |--- mean_chosen_prob >  0.55
|   |   |--- mean_chosen_prob <= 0.67
|   |   |   |--- class: 1
|   |   |--- mean_chosen_prob >  0.67
|   |   |   |--- class: 1

```

## qwen / format_target

- n=360, positive_rate=0.247
- GroupKFold balanced accuracy=0.685 ± 0.023

```text
|--- mean_chosen_flatness <= 2.87
|   |--- mean_chosen_flatness <= 2.72
|   |   |--- mean_selection_gain <= 0.07
|   |   |   |--- class: 1
|   |   |--- mean_selection_gain >  0.07
|   |   |   |--- class: 1
|   |--- mean_chosen_flatness >  2.72
|   |   |--- mean_chosen_rank <= 908.59
|   |   |   |--- class: 1
|   |   |--- mean_chosen_rank >  908.59
|   |   |   |--- class: 0
|--- mean_chosen_flatness >  2.87
|   |--- mean_chosen_rank <= 238.99
|   |   |--- fallback_rate <= 0.13
|   |   |   |--- class: 0
|   |   |--- fallback_rate >  0.13
|   |   |   |--- class: 0
|   |--- mean_chosen_rank >  238.99
|   |   |--- mean_chosen_prob <= 0.11
|   |   |   |--- class: 0
|   |   |--- mean_chosen_prob >  0.11
|   |   |   |--- class: 1

```

## 解釈

- balanced accuracyが0.5付近なら、token log特徴だけでは安定分類できない。
- 0.5を明確に上回る場合は、形状名より連続的な選択状態を制御変数にできる可能性がある。
- これは表層分類の検定であり、意味品質や因果効果の検定ではない。
