# 18種類のロジット形状解析（3モデル）

`detect_shape_family()` の定義に合わせ、既存 candidate pool の successor features から18ラベルを再構成した。これは新たなforwardではなく、保存済み特徴量の再分類である。1行が複数shapeに入ることがある。

## 18種類の定義

| family | shapes |
|---|---|
| spread | low_spread / mid_spread / high_spread |
| concentration | low_concentration / mid_concentration / high_concentration |
| tail | thin_tail / mid_tail / heavy_tail |
| gap | small_gap / mid_gap / large_gap |
| rank shape | rank_slope_flat / rank_slope_mid / rank_slope_steep |
| peak structure | spiky_top / smooth_top / multi_peak |

## カバレッジと平均特徴

| model | shape | rows | prompts | std | top1 | tail50 | gap | rank slope | large gaps | gap std |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gemma_raw | `multi_peak` | 303 | 20 | 3.166 | 0.928 | 0.002 | 3.902 | -0.122 | 2.34 | 0.667 |
| gpt2 | `multi_peak` | 41 | 18 | 2.497 | 0.933 | 0.019 | 4.671 | -0.087 | 2.10 | 0.699 |
| qwen_raw | `multi_peak` | 53 | 19 | 2.730 | 0.917 | 0.005 | 4.307 | -0.105 | 2.17 | 0.698 |
| gemma_raw | `smooth_top` | 35 | 15 | 3.198 | 0.144 | 0.119 | 0.071 | -0.088 | 0.06 | 0.124 |
| gpt2 | `smooth_top` | 244 | 20 | 2.909 | 0.098 | 0.357 | 0.071 | -0.064 | 0.10 | 0.113 |
| qwen_raw | `smooth_top` | 206 | 20 | 3.043 | 0.116 | 0.272 | 0.071 | -0.070 | 0.12 | 0.121 |
| gemma_raw | `spiky_top` | 1031 | 20 | 3.146 | 0.841 | 0.010 | 3.199 | -0.114 | 1.87 | 0.519 |
| gpt2 | `spiky_top` | 201 | 20 | 2.918 | 0.845 | 0.040 | 4.318 | -0.087 | 1.47 | 0.633 |
| qwen_raw | `spiky_top` | 229 | 20 | 2.872 | 0.812 | 0.023 | 3.211 | -0.101 | 1.71 | 0.503 |
| gemma_raw | `rank_slope_flat` | 0 | 0 | nan | nan | nan | nan | nan | nan | nan |
| gpt2 | `rank_slope_flat` | 262 | 20 | 2.907 | 0.076 | 0.590 | 0.661 | -0.042 | 0.24 | 0.130 |
| qwen_raw | `rank_slope_flat` | 147 | 20 | 2.979 | 0.082 | 0.553 | 0.661 | -0.044 | 0.24 | 0.130 |
| gemma_raw | `rank_slope_mid` | 1993 | 20 | 3.147 | 0.620 | 0.032 | 1.985 | -0.108 | 1.42 | 0.387 |
| gpt2 | `rank_slope_mid` | 1738 | 20 | 2.883 | 0.299 | 0.230 | 1.136 | -0.075 | 0.64 | 0.232 |
| qwen_raw | `rank_slope_mid` | 1853 | 20 | 3.004 | 0.337 | 0.149 | 1.007 | -0.084 | 0.72 | 0.229 |
| gemma_raw | `rank_slope_steep` | 7 | 6 | 5.911 | 0.880 | 0.000 | 3.180 | -0.243 | 3.29 | 0.583 |
| gpt2 | `rank_slope_steep` | 0 | 0 | nan | nan | nan | nan | nan | nan | nan |
| qwen_raw | `rank_slope_steep` | 0 | 0 | nan | nan | nan | nan | nan | nan | nan |

## 18形状それぞれの初見

- `low_spread`: logit幅が狭い。ただし尖ったGemmaと競合的なQwenを含むため、flatness単独では意味が決まらない。
- `mid_spread`: 中間幅。3モデルで比較的stdが近く、基準形状として使いやすい。
- `high_spread`: 幅が広いが、厚いtailとは限らない。GPT-2/Gemmaでは尖った広幅も含む。
- `low_concentration`: top1が低い探索型。tailも厚くなりやすいがGemmaでは必ずしもそうならない。
- `mid_concentration`: 複数候補を残す中間型。ただしモデルごとの確率校正差が大きい。
- `high_concentration`: 単峰・top1優勢。安定性と保守性、反復リスクが同居する。
- `thin_tail`: tailがほぼ上位候補に吸収された状態。large_gap/high_concentrationと重複しやすい。
- `mid_tail`: tailの折衷型。探索性を残しつつ極端な拡散を避ける候補。
- `heavy_tail`: 上位50外に質量が残る。候補の多さは増えるが、意味的接地は保証しない。
- `small_gap`: top1/top2が競合。sampling・seedに敏感で、候補選択の不安定性を示す。
- `mid_gap`: 候補間の優劣が中程度。単峰と競合の中間状態。
- `large_gap`: top1が明確。thin_tailと強く重複し、保守的な次状態になりやすい。
- `rank_slope_flat`: 上位rank間の確率減衰が緩い。top1だけでなく上位候補全体に質量がある。
- `rank_slope_mid`: rank減衰の中間。形状の連続的な補間点として有用。
- `rank_slope_steep`: 上位から急落するrank曲線。単峰性・選択固定化と結びつきやすい。
- `spiky_top`: top1が高くgapも大きい明確な単峰。high_concentration/large_gap/thin_tailと重複する。
- `smooth_top`: top1もgapも小さい競合型。候補の微差に敏感で、探索的だが不安定になりやすい。
- `multi_peak`: 上位50内に複数の大きな段差がある多峰型。単純なtop1/tailでは表せない。

## 観測されなかったモデル×形状

- `gpt2` × `rank_slope_steep`: 0 rows / 0 prompts
- `gemma_raw` × `rank_slope_flat`: 0 rows / 0 prompts
- `qwen_raw` × `rank_slope_steep`: 0 rows / 0 prompts

0件は単なるファイル欠損ではない。現行のモデル・prompt・top-100 candidate pool・step 0 条件では、そのshape ruleを満たす successor state が観測されなかったことを意味する。

## 重要な新しい洞察

1. これまで欠けていた rank shape と peak structure を加えると、18形状は『3つの独立な分布軸』ではなく、spread・concentration・tail・gap・rank curve・peak topology の6軸になる。
2. `spiky_top` は単なる高 concentration ではなく、top1とtop2の明確な分離を要求する。高 concentrationでもmulti-peakや厚いtailはあり得るため、peak familyは追加情報を持つ。
3. `rank_slope_flat` と `heavy_tail` は似て見えるが、前者は上位rankの減衰、後者はtop50外の質量を測る。両者が一致しないモデル・形状を探す価値がある。
4. `multi_peak` はflatness研究にとって重要な候補である。単純にminflat/maxflatを選ぶより、複数の確率峰を持つ successor state を避ける／選ぶ制御が可能になる。
5. 18形状を横断しても、raw logit stdの絶対値より model-local percentile の方が比較可能である。

## 解析上の限界

- candidate pool は top-100、step 0 の初期promptだけである。生成時系列の18形状頻度ではない。
- token-type five shapes は今回の18 shape familyとは別の診断で、内容依存のためdistribution topologyと分けて解釈する。
- `recent_overlap` は旧データではprompt overlapと同じ値になっていたため、形状解析の中心18には含めていない。
- Qwen raw markup_heavyの不足は別問題として残っている。
