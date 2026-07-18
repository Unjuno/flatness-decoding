# 形状別出力ログの横断解析

保存済み `shapes_20` の continuation と token-log突合結果を形状×モデルで再集計した。各セルは20 promptを目標とし、再生成は行っていない。

## 読み方

- `mean_repeated_4gram_rate`: 反復・縮退の粗い指標
- `mean_non_ascii_rate`: tokenizer／形式崩れの粗い指標（多言語文字を含むため単独で品質判定しない）
- `mean_markup_marker_rate`: マークアップ・記号混入の粗い指標
- `mean_prompt_word_reuse_rate`: プロンプト語彙の再利用率。接地と単純反復を分離していない

## 形状ごとの暫定分類

- **heavy_tail**: 反復最大=gemma (0.005); 非ASCII最大=gemma (0.004); prompt語彙再利用最大=qwen (0.150)
- **high_concentration**: 反復最大=gpt2 (0.107); 非ASCII最大=gpt2 (0.015); prompt語彙再利用最大=qwen (0.191)
- **high_spread**: 反復最大=gpt2 (0.004); 非ASCII最大=qwen (0.004); prompt語彙再利用最大=qwen (0.144)
- **large_gap**: 反復最大=gpt2 (0.055); 非ASCII最大=gemma (0.003); prompt語彙再利用最大=qwen (0.233)
- **low_concentration**: 反復最大=gemma (0.000); 非ASCII最大=qwen (0.067); prompt語彙再利用最大=gemma (0.115)
- **low_spread**: 反復最大=gemma (0.052); 非ASCII最大=qwen (0.087); prompt語彙再利用最大=gemma (0.130)
- **mid_concentration**: 反復最大=gemma (0.000); 非ASCII最大=gemma (0.004); prompt語彙再利用最大=qwen (0.151)
- **mid_gap**: 反復最大=gemma (0.001); 非ASCII最大=qwen (0.024); prompt語彙再利用最大=gemma (0.130)
- **mid_spread**: 反復最大=gpt2 (0.002); 非ASCII最大=gemma (0.005); prompt語彙再利用最大=gemma (0.136)
- **mid_tail**: 反復最大=gpt2 (0.010); 非ASCII最大=gpt2 (0.008); prompt語彙再利用最大=gemma (0.135)
- **multi_peak**: 反復最大=gpt2 (0.044); 非ASCII最大=qwen (0.028); prompt語彙再利用最大=qwen (0.204)
- **rank_slope_flat**: 反復最大=gemma (0.001); 非ASCII最大=qwen (0.014); prompt語彙再利用最大=gemma (0.129)
- **rank_slope_mid**: 反復最大=qwen (0.026); 非ASCII最大=gemma (0.003); prompt語彙再利用最大=qwen (0.191)
- **rank_slope_steep**: 反復最大=gpt2 (0.023); 非ASCII最大=gpt2 (0.047); prompt語彙再利用最大=qwen (0.160)
- **small_gap**: 反復最大=gemma (0.000); 非ASCII最大=gemma (0.029); prompt語彙再利用最大=gemma (0.132)
- **smooth_top**: 反復最大=gemma (0.000); 非ASCII最大=qwen (0.027); prompt語彙再利用最大=gemma (0.117)
- **spiky_top**: 反復最大=gpt2 (0.100); 非ASCII最大=gpt2 (0.033); prompt語彙再利用最大=qwen (0.192)
- **thin_tail**: 反復最大=gpt2 (0.151); 非ASCII最大=gpt2 (0.037); prompt語彙再利用最大=qwen (0.200)

## 現時点の洞察

1. 出力の差は形状単独よりもモデルとの交互作用が強い。特に `low_spread`、`thin_tail`、`high_concentration`、`spiky_top` で反復または非ASCII混入がモデル間で大きく分かれる。
2. `large_gap` は3モデルで反復率が比較的高く、上位候補の断絶が続きやすいという仮説を支持する。ただし `low_spread` とは異なり、モデル固有の形式崩れだけでは説明しにくい。
3. `rank_slope_flat` や `smooth_top` は平均的には比較的安定だが、Qwenの prompt語彙再利用や非ASCII混入のように、形状の安定性と内容の健全性は別軸である。
4. この表は表層指標であり、意味的な物語品質・遵守性の決定的分類ではない。次の判定では出力本文の手動サンプル監査または意味カテゴリ分類を追加する。
