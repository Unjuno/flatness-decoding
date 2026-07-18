# 形状のモデル横断効果分類

paired greedy差分を、形状ごとに3モデルの効果方向へ集約した。閾値は記述的な実務分類用で、統計的有意差の判定ではない。

| shape              | repeat_effect_class    | markup_effect_class    | length_effect_class    |
|:-------------------|:-----------------------|:-----------------------|:-----------------------|
| heavy_tail         | consistently_decreases | model_dependent        | model_dependent        |
| high_concentration | model_dependent        | model_dependent        | model_dependent        |
| high_spread        | consistently_decreases | consistently_decreases | model_dependent        |
| large_gap          | model_dependent        | consistently_decreases | model_dependent        |
| low_concentration  | consistently_decreases | model_dependent        | model_dependent        |
| low_spread         | model_dependent        | model_dependent        | consistently_decreases |
| mid_concentration  | consistently_decreases | model_dependent        | model_dependent        |
| mid_gap            | consistently_decreases | model_dependent        | model_dependent        |
| mid_spread         | consistently_decreases | consistently_decreases | model_dependent        |
| mid_tail           | consistently_decreases | consistently_decreases | model_dependent        |
| multi_peak         | model_dependent        | model_dependent        | consistently_decreases |
| rank_slope_flat    | consistently_decreases | model_dependent        | model_dependent        |
| rank_slope_mid     | model_dependent        | consistently_decreases | model_dependent        |
| rank_slope_steep   | model_dependent        | consistently_decreases | consistently_decreases |
| small_gap          | consistently_decreases | model_dependent        | model_dependent        |
| smooth_top         | consistently_decreases | model_dependent        | model_dependent        |
| spiky_top          | model_dependent        | model_dependent        | model_dependent        |
| thin_tail          | model_dependent        | model_dependent        | model_dependent        |

## 分類の意味

- `consistently_increases/decreases`: 3モデルで同じ方向に変化。モデル横断の制御候補。
- `model_dependent`: 効果の向きまたは大きさがモデルで分かれる。model-local calibrationが必要。
- `mostly_neutral`: 指標上の差が小さい。

## 現時点の結論

- 反復率について、全モデルで一貫して増加／減少する形状は限定的で、形状名だけの普遍的な良し悪しは支持されない。
- 多くの形状はmodel-dependentであり、モデル内percentileと出力リスクを組み合わせる設計が妥当である。
- 形式崩れと出力長は反復とは異なる方向を示すことがあり、単一のflatness軸に統合すべきではない。
