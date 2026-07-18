# ロジット形状と本文出力のprompt単位リンク解析

同じ `shape × model × prompt_id` のtoken logとcontinuation表層分類を結合した。相関は記述的なSpearman相関であり、因果効果ではない。

- 結合行数: 1080
- 形状数: 18
- モデル数: 3

## 最も強い関連（絶対値上位）

| model   | shape             | x                    | y                      |   spearman_rho |   n |
|:--------|:------------------|:---------------------|:-----------------------|---------------:|----:|
| qwen    | low_spread        | mean_chosen_flatness | non_ascii_rate         |      -0.865023 |  20 |
| qwen    | smooth_top        | mean_chosen_flatness | output_words           |       0.834777 |  20 |
| qwen    | rank_slope_flat   | fallback_rate        | output_words           |       0.825923 |  20 |
| gpt2    | mid_spread        | fallback_rate        | prompt_word_reuse_rate |      -0.795857 |  20 |
| qwen    | rank_slope_flat   | mean_chosen_flatness | output_words           |       0.79217  |  20 |
| qwen    | low_spread        | mean_chosen_flatness | output_words           |       0.785849 |  20 |
| gpt2    | mid_gap           | mean_selection_gain  | output_words           |       0.782676 |  20 |
| gemma   | mid_spread        | mean_chosen_prob     | output_words           |      -0.768073 |  20 |
| gpt2    | spiky_top         | mean_chosen_prob     | repeated_4gram_rate    |       0.763057 |  20 |
| gpt2    | small_gap         | mean_selection_gain  | output_words           |       0.756414 |  20 |
| gpt2    | spiky_top         | mean_selection_gain  | repeated_4gram_rate    |      -0.747291 |  20 |
| gpt2    | mid_spread        | fallback_rate        | output_words           |      -0.736822 |  20 |
| qwen    | small_gap         | mean_chosen_flatness | output_words           |       0.735218 |  20 |
| gpt2    | rank_slope_steep  | mean_selection_gain  | output_words           |       0.731377 |  20 |
| gpt2    | thin_tail         | mean_chosen_prob     | repeated_4gram_rate    |       0.725206 |  20 |
| gpt2    | spiky_top         | fallback_rate        | repeated_4gram_rate    |      -0.723643 |  20 |
| qwen    | large_gap         | fallback_rate        | prompt_word_reuse_rate |      -0.708466 |  20 |
| gpt2    | spiky_top         | mean_chosen_flatness | repeated_4gram_rate    |       0.704724 |  20 |
| gpt2    | mid_gap           | mean_chosen_flatness | output_words           |       0.674954 |  20 |
| gpt2    | heavy_tail        | mean_chosen_prob     | markup_marker_rate     |      -0.673928 |  20 |
| qwen    | mid_gap           | mean_chosen_flatness | non_ascii_rate         |      -0.673217 |  20 |
| qwen    | smooth_top        | fallback_rate        | output_words           |       0.672193 |  20 |
| gemma   | small_gap         | mean_chosen_flatness | output_words           |      -0.670181 |  20 |
| qwen    | low_concentration | mean_chosen_flatness | non_ascii_rate         |      -0.668834 |  20 |
| gemma   | spiky_top         | mean_chosen_prob     | non_ascii_rate         |      -0.666686 |  20 |
| qwen    | rank_slope_flat   | mean_chosen_prob     | output_words           |       0.666416 |  20 |
| gemma   | mid_spread        | fallback_rate        | output_words           |      -0.664031 |  20 |
| gpt2    | rank_slope_mid    | mean_chosen_prob     | output_words           |      -0.662648 |  20 |
| qwen    | low_concentration | mean_chosen_flatness | output_words           |       0.655368 |  20 |
| gpt2    | rank_slope_steep  | mean_chosen_prob     | output_words           |      -0.653123 |  20 |

## 解釈

- `mean_chosen_rank` と `mean_chosen_prob` は、形状そのものではなく選択候補が元分布のどこにあるかを表す。
- 反復やmarkupとの相関が形状をまたいで同じ向きになる場合、形状名より連続的なtoken-level coordinateを使う方が分類に適する。
- prompt単位n=20なので、絶対値の小さな相関は無視し、複数形状・複数モデルで再現する向きだけを仮説として残す。
