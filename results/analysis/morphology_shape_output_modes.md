# 形状別の出力モード分類

保存済み `shapes_20` continuation を、監査可能な表層ルールで分類した。これは意味品質の判定ではなく、出力形式・反復・物語らしさの診断である。

- **heavy_tail**: narrative_like最大=qwen (0.35); repetition最大=gemma (0.05)
- **high_concentration**: narrative_like最大=gemma (0.50); repetition最大=gpt2 (0.40)
- **high_spread**: narrative_like最大=gemma (0.40); repetition最大=gpt2 (0.05)
- **large_gap**: narrative_like最大=gemma (0.30); repetition最大=gpt2 (0.25)
- **low_concentration**: narrative_like最大=gemma (0.50); repetition最大=gemma (0.00)
- **low_spread**: narrative_like最大=gemma (0.35); repetition最大=gemma (0.10)
- **mid_concentration**: narrative_like最大=gpt2 (0.50); repetition最大=gemma (0.00)
- **mid_gap**: narrative_like最大=gemma (0.35); repetition最大=gemma (0.00)
- **mid_spread**: narrative_like最大=gemma (0.40); repetition最大=gemma (0.00)
- **mid_tail**: narrative_like最大=gemma (0.40); repetition最大=gpt2 (0.05)
- **multi_peak**: narrative_like最大=gemma (0.35); repetition最大=gpt2 (0.35)
- **rank_slope_flat**: narrative_like最大=gemma (0.25); repetition最大=gemma (0.00)
- **rank_slope_mid**: narrative_like最大=gemma (0.55); repetition最大=qwen (0.25)
- **rank_slope_steep**: narrative_like最大=gemma (0.35); repetition最大=qwen (0.20)
- **small_gap**: narrative_like最大=gemma (0.50); repetition最大=gemma (0.00)
- **smooth_top**: narrative_like最大=gemma (0.40); repetition最大=gemma (0.00)
- **spiky_top**: narrative_like最大=gemma (0.40); repetition最大=gpt2 (0.35)
- **thin_tail**: narrative_like最大=gemma (0.30); repetition最大=gpt2 (0.45)

## 注意

- regexベースなので、文脈理解や文章品質を保証しない。
- `list_or_meta` と `instruction_like` は、形状が形式トークンを誘発するかを調べる補助変数である。
- 形状分類の決定性を上げるには、同一prompt・同一seedで候補形状をpairedにした本文比較が必要である。
