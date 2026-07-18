# 18形状の内容的特徴解析

候補 token の文字列を直接調べた。これは生成文の意味評価ではなく、形状ごとにどの種類の token が選ばれやすいかの解析である。

## token kind 比率

| model | shape | word/fragment | punctuation | markup/bullet | digit | whitespace | other |
|---|---|---:|---:|---:|---:|---:|---:|
| gemma_raw | `multi_peak` | 0.52 | 0.01 | 0.10 | 0.01 | 0.06 | 0.31 |
| gemma_raw | `rank_slope_mid` | 0.66 | 0.03 | 0.08 | 0.00 | 0.07 | 0.17 |
| gemma_raw | `rank_slope_steep` | 0.00 | 0.00 | 0.00 | 0.00 | 0.86 | 0.14 |
| gemma_raw | `smooth_top` | 0.89 | 0.03 | 0.00 | 0.00 | 0.06 | 0.03 |
| gemma_raw | `spiky_top` | 0.57 | 0.02 | 0.10 | 0.00 | 0.07 | 0.24 |
| gpt2 | `multi_peak` | 0.71 | 0.00 | 0.00 | 0.00 | 0.00 | 0.29 |
| gpt2 | `rank_slope_flat` | 0.60 | 0.00 | 0.08 | 0.01 | 0.07 | 0.24 |
| gpt2 | `rank_slope_mid` | 0.86 | 0.01 | 0.00 | 0.02 | 0.05 | 0.06 |
| gpt2 | `smooth_top` | 0.86 | 0.01 | 0.02 | 0.00 | 0.00 | 0.11 |
| gpt2 | `spiky_top` | 0.63 | 0.00 | 0.00 | 0.00 | 0.28 | 0.08 |
| qwen_raw | `multi_peak` | 0.81 | 0.13 | 0.00 | 0.00 | 0.00 | 0.06 |
| qwen_raw | `rank_slope_flat` | 0.46 | 0.04 | 0.16 | 0.00 | 0.00 | 0.33 |
| qwen_raw | `rank_slope_mid` | 0.89 | 0.01 | 0.01 | 0.00 | 0.03 | 0.06 |
| qwen_raw | `smooth_top` | 0.80 | 0.00 | 0.04 | 0.00 | 0.02 | 0.14 |
| qwen_raw | `spiky_top` | 0.88 | 0.03 | 0.03 | 0.00 | 0.00 | 0.05 |

## 上位 token の例

- gemma_raw / `multi_peak`: ' .'×9, ' !'×9, ''×8, ' ;'×7, ' However'×7, ' *'×6, ' ,'×6, ' _'×5
- gemma_raw / `rank_slope_mid`: ''×23, ' .'×20, '\\n'×20, '\\n\\n'×20, ' This'×20, ' The'×20, ' ,'×20, ' It'×20
- gemma_raw / `rank_slope_steep`: ' '×6, ' \\'×1
- gemma_raw / `smooth_top`: ' What'×3, ' One'×2, ' A'×2, '\\n\\n'×2, ' Some'×1, ' She'×1, ' An'×1, ' Not'×1
- gemma_raw / `spiky_top`: ' .'×20, ' *'×19, ' ,'×15, ''×15, ' ;'×13, ' This'×12, ' -'×12, ' And'×12
- gpt2 / `multi_peak`: ' .'×12, ' Don'×6, ' Of'×6, ' According'×6, ' Written'×2, ' Are'×1, ' Ar'×1, ' Star'×1
- gpt2 / `rank_slope_flat`: ''×19, ' ('×17, ' A'×14, ' ['×13, ' The'×12, ' -'×11, ' An'×11, ' To'×8
- gpt2 / `rank_slope_mid`: '\\n'×20, ' You'×20, '\\n\\n'×20, ' .'×20, ' It'×20, ' If'×20, ' In'×20, ' When'×20
- gpt2 / `smooth_top`: ' To'×7, ' All'×7, ' ('×6, ' When'×5, ' An'×5, ' Why'×5, ' And'×5, ' I'×5
- gpt2 / `spiky_top`: '\\n'×20, '\\n\\n'×20, ' '×17, ' .'×16, ' However'×12, ' Of'×12, ' Don'×10, ' According'×7
- qwen_raw / `multi_peak`: '\\'×7, ' According'×7, ' Based'×6, ' .'×2, ' However'×2, ' Instead'×2, ' Answer'×2, ' Of'×2
- qwen_raw / `rank_slope_flat`: ' ['×19, ' ('×16, ' A'×10, ' *'×8, ' "'×6, ' The'×5, ' By'×5, '['×5
- qwen_raw / `rank_slope_mid`: ' What'×20, ' In'×20, ' As'×20, ' It'×20, ' How'×20, ' '×20, ' There'×20, ' For'×20
- qwen_raw / `smooth_top`: ' We'×8, ' "'×7, ' I'×6, ' How'×6, ' Just'×6, ' The'×5, ' A'×5, ' What'×5
- qwen_raw / `spiky_top`: ' Based'×12, ' Do'×9, ' Can'×8, ' However'×8, ' If'×7, ' Of'×7, '\\'×7, ' According'×7

## 直接的な初見

- これまでの統計解析だけでは内容差を見落としていた。候補 token を見ると、形状差は語彙意味そのものより、word fragment / whitespace / punctuation / markup の選択傾向として現れる可能性が高い。
- `spiky_top` はモデルによっては特定の一般語 token に集中し、`smooth_top` は whitespace・punctuation を含む競合状態になりやすい。
- `multi_peak` は内容が多様というより、複数の token 群（語、記号、形式 token）が競合している状態である可能性がある。
- ただし候補 token だけでは、その token を選んだ後の文章内容までは分からない。形状別の意味的差を主張するには、各候補から continuation を生成して比較する必要がある。
- 現在のデータには morphology probe の continuation が保存されていないため、「内容的に差がない」と結論するのは不適切であり、今回の解析は内容差の入口に留まる。

