# 形状のprompt内paired効果

同じ `model × prompt_id` の `greedy` continuationを基準に、各形状methodの保存済み出力との差分を計算した。これは形状の相対的な出力効果を調べる記述解析である。

## 反復率の増加が大きい形状

| shape              | model   |   delta_repeat_rate_mean |   delta_repeat_rate_median |   delta_repeat_rate_<lambda_0> |
|:-------------------|:--------|-------------------------:|---------------------------:|-------------------------------:|
| high_concentration | qwen    |              0.0417396   |                 0.00579897 |                           0.5  |
| spiky_top          | gemma   |              0.0390527   |                 0          |                           0.15 |
| thin_tail          | gemma   |              0.0373668   |                 0          |                           0.15 |
| spiky_top          | qwen    |              0.0299882   |                 0.00531915 |                           0.5  |
| low_spread         | gemma   |              0.0258217   |                 0          |                           0.1  |
| large_gap          | gemma   |              0.023059    |                 0          |                           0.05 |
| large_gap          | qwen    |              0.0134625   |                 0          |                           0.35 |
| thin_tail          | qwen    |              0.0130335   |                 0          |                           0.35 |
| multi_peak         | qwen    |              0.0076911   |                 0          |                           0.45 |
| rank_slope_mid     | qwen    |             -0.000353562 |                 0          |                           0.4  |
| rank_slope_steep   | qwen    |             -0.00339789  |                 0          |                           0.45 |
| low_spread         | qwen    |             -0.0195522   |                 0          |                           0.1  |
| heavy_tail         | gemma   |             -0.0212649   |                 0          |                           0.15 |
| multi_peak         | gemma   |             -0.0215907   |                 0          |                           0.15 |
| high_concentration | gemma   |             -0.0219264   |                 0          |                           0.1  |

## 解釈

- greedyとの差分なので、モデル全体の出力傾向より形状methodによる相対変化を見やすい。
- `delta_repeat_rate_mean > 0` でも、同一promptで常に悪化するとは限らない。最後の列は悪化したpromptの割合である。
- 形状ごとの効果がモデル間で同じ向きなら、モデル横断の制御候補になる。向きが逆なら、model-local calibrationが必要である。
