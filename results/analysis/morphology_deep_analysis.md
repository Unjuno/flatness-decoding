# 形状解析の追加診断

対象: GPT-2 / Gemma raw / Qwen raw、17レジーム、各20件（Qwen raw markup_heavyのみ11件）。

## 1. レジーム内の指標相関

### gemma_raw

```text
                                      mean_successor_logit_std  mean_successor_top1_prob  mean_successor_tail_mass_after_top50  mean_successor_gap_1_2
mean_successor_logit_std                                 1.000                    -0.496                                 0.432                  -0.447
mean_successor_top1_prob                                -0.496                     1.000                                -0.892                   0.948
mean_successor_tail_mass_after_top50                     0.432                    -0.892                                 1.000                  -0.919
mean_successor_gap_1_2                                  -0.447                     0.948                                -0.919                   1.000
```

### gpt2

```text
                                      mean_successor_logit_std  mean_successor_top1_prob  mean_successor_tail_mass_after_top50  mean_successor_gap_1_2
mean_successor_logit_std                                 1.000                     0.408                                -0.425                   0.445
mean_successor_top1_prob                                 0.408                     1.000                                -0.956                   0.988
mean_successor_tail_mass_after_top50                    -0.425                    -0.956                                 1.000                  -0.933
mean_successor_gap_1_2                                   0.445                     0.988                                -0.933                   1.000
```

### qwen_raw

```text
                                      mean_successor_logit_std  mean_successor_top1_prob  mean_successor_tail_mass_after_top50  mean_successor_gap_1_2
mean_successor_logit_std                                 1.000                    -0.858                                 0.762                  -0.747
mean_successor_top1_prob                                -0.858                     1.000                                -0.787                   0.921
mean_successor_tail_mass_after_top50                     0.762                    -0.787                                 1.000                  -0.595
mean_successor_gap_1_2                                  -0.747                     0.921                                -0.595                   1.000
```

## 2. モデル間の同型性

- `mean_successor_logit_std`: GPT-2↔Gemma=-0.043, GPT-2↔Qwen=0.156, Gemma↔Qwen=0.425
- `mean_successor_top1_prob`: GPT-2↔Gemma=0.809, GPT-2↔Qwen=0.788, Gemma↔Qwen=0.816
- `mean_successor_tail_mass_after_top50`: GPT-2↔Gemma=0.876, GPT-2↔Qwen=0.781, Gemma↔Qwen=0.743
- `mean_successor_gap_1_2`: GPT-2↔Gemma=0.880, GPT-2↔Qwen=0.811, Gemma↔Qwen=0.825

解釈: レジーム名が同じでも、数値の順位がモデル間で一致しない場合、そのレジームはモデル共通の形状ではなく、各モデル内の相対的な状態を表す。

## 3. 近い／重複しているレジーム

各モデルで4指標を標準化し、レジーム間の平均ユークリッド距離を計算した。

### gpt2
- `large_gap` ≈ `list_heavy`: distance=0.000
- `large_gap` ≈ `fragment_heavy`: distance=0.000
- `list_heavy` ≈ `fragment_heavy`: distance=0.000
- `prompt_overlap_heavy` ≈ `recent_overlap_heavy`: distance=0.000
- `low_concentration` ≈ `prompt_overlap_heavy`: distance=0.081
- `low_concentration` ≈ `recent_overlap_heavy`: distance=0.081

### gemma_raw
- `high_concentration` ≈ `large_gap`: distance=0.000
- `prompt_overlap_heavy` ≈ `recent_overlap_heavy`: distance=0.000
- `high_concentration` ≈ `thin_tail`: distance=0.125
- `thin_tail` ≈ `large_gap`: distance=0.125
- `low_spread` ≈ `thin_tail`: distance=0.143
- `high_concentration` ≈ `fragment_heavy`: distance=0.159

### qwen_raw
- `prompt_overlap_heavy` ≈ `recent_overlap_heavy`: distance=0.000
- `high_spread` ≈ `prompt_overlap_heavy`: distance=0.366
- `high_spread` ≈ `recent_overlap_heavy`: distance=0.366
- `low_concentration` ≈ `heavy_tail`: distance=0.501
- `high_spread` ≈ `heavy_tail`: distance=0.537
- `low_concentration` ≈ `small_gap`: distance=0.563

## 4. 形状ごとの初見

- `low_spread`: 低 spread は flat ではなく、Gemma ではほぼ一語集中、Qwen では競合候補を残す。名前だけで安全性を判断できない。
- `mid_spread`: 3モデルの std は比較的近いが、tail と gap が違う。単独の spread 指標としては最も標準的だが、分布形状はまだ多様。
- `high_spread`: 広い幅と厚い tail は同義でない。GPT-2/Gemma は尖った広幅、Qwen は比較的 diffuse。
- `low_concentration`: top1 が低い探索型。Qwen/GPT-2 は tail も厚いが、Gemma は tail が薄く、top1 だけでは判別不足。
- `mid_concentration`: 中間という名前に反し、Gemma は top1=.623 まで上がる。モデルごとの calibration 差が大きい。
- `high_concentration`: 単峰・大 gap・薄 tail の安定型。ただし選択が固定化しやすく、反復／保守性の入口になり得る。
- `thin_tail`: large_gap/high_concentration とほぼ同じ cluster。独立形状ではなく同一現象の別投影。
- `mid_tail`: tail を中間に保つ形状だが、Gemma だけ top1/gap が高い。tail 制御もモデル依存。
- `heavy_tail`: 3モデルで std がほぼ同じなのに top1/tail は大きく違う。raw logit std だけでは探索性を説明できない。
- `small_gap`: top1/top2 が競合する状態。候補選択の微小な差に敏感で、sampling の揺らぎが出やすい。
- `mid_gap`: gap の中間だが Gemma の gap はQwen/GPT-2より大きい。gap の絶対値比較は危険。
- `large_gap`: high concentration/thin tail と重複する強い単峰状態。高確率だが、常に良いとは限らない。
- `markup_heavy`: tokenizer と prompt framing の影響が最も強い形状。Gemma は尖り、GPT-2/Qwen は tail を残す。
- `list_heavy`: GPT-2 では大 gap・高 top1、Qwen では低 top1。『list』という token 内容はモデル固有。
- `fragment_heavy`: Gemma/GPT-2 は単峰的 fragment、Qwen はより分散。fragment collapse の危険度は tokenizer 依存。
- `prompt_overlap_heavy`: prompt 再利用は3モデルで低 gap・厚めの tail と結びつくが、接地と leakage を分離できていない。
- `recent_overlap_heavy`: 現データでは prompt_overlap と完全に同じ統計値。現行 detector は両者を分けられていない。

## 5. いちばん重要な洞察

1. 形状は18個の独立カテゴリではなく、spread / concentration / tail / gap / token-content の5軸上の状態である。
2. raw logit std はモデル間で絶対比較できない。比較にはモデル内 percentile、z-score、または calibrated morphology coordinates が必要。
3. `markup_heavy`、`fragment_heavy`、`list_heavy` は distribution shape というより token-content state であり、形状解析と出力形式解析を分けるべきである。
4. `prompt_overlap_heavy` と `recent_overlap_heavy` の同値性は、特徴量または probe 実装の重複を示す。時系列 token identity まで保存して再定義すべきである。
5. flatness selector を作るなら一軸 min/max ではなく、モデル内 percentile の2軸以上（例: concentration × tail、または gap × token-content）で target region を指定するのが自然である。

## 6. 次に検証すべき仮説

- H1: `low_spread` かつ `large_gap` は反復／保守性、`low_spread` かつ `small_gap` は不安定な競合状態を予測する。
- H2: `markup_heavy` の出力形式逸脱率は raw/wrapped framing で変わるが、モデル内 morphology percentile は維持される。
- H3: `recent_overlap_heavy` は prompt overlap と異なり、生成 step が進むほど repetition_4 と強く相関する。
- H4: morphology coordinates をモデル内 percentile に変換すれば、q50/q75 の安全領域がモデル間で部分的に揃う。

