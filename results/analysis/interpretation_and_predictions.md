# 実験データの解釈と予測

## 結論

現時点のデータは、successor logit dispersion（次状態 logits の標準偏差）を「品質を直接最大化する目的関数」とみなすより、生成軌道を分ける診断・制御軸とみなす解釈を支持する。

1. 低 dispersion（`minflat`）は低尤度・逸脱しやすい出力へ進む。
2. 中間 dispersion（`centerflat_q50/q75`）は、候補再ランキングと既存の確率分布への近さのバランスがよい。
3. 高 dispersion（`maxflat`）は minflat より破綻は少ないが、拡散的・プロンプト接地の弱い出力になる。

したがって、この手法の主な貢献は「flatness で高品質化すること」ではなく、同じモデル・同じ proposal 分布から、保守的、中心的、拡散的、逸脱的な生成レジームを選択できることだと予想する。

## データから確認できるパターン

### 1. dispersion は likelihood と強く関係するが同一ではない

GPT-2 の WritingPrompts 集計では、method/N 集計における mean chosen flatness と mean chosen current rank の Spearman 相関は -0.951、PPL との相関は -0.910 だった。DistilGPT-2 でもそれぞれ -0.873、-0.516 である。

これは dispersion がランダムな別軸ではなく、候補の確率・順位と結びついていることを示す。一方で、同じ flatness 付近でも repetition や distinctness が大きく異なるため、PPL の代替指標でもない。

### 2. minflat は最も明確な失敗側の極端値

GPT-2, N=8 では、`minflat` は mean chosen flatness 1.75、mean chosen current rank 1676、PPL 345 だった。対して `centerflat_q50` は 3.29、249、37、`centerflat_q75` は 3.54、168、28 である。DistilGPT-2 でも minflat は flatness 1.92、rank 1652、PPL 335 となり、同じ方向を示す。

低い次状態 dispersion を持つ候補が「自然な次語」ではなく、モデルを局所的な低エネルギー状態へ押し込む候補になっていると解釈する。失敗は単純な n-gram 反復だけでなく、短い・不自然・逸脱した継続として現れている。

### 3. centerflat は局所的な安全領域

GPT-2 では N=8 の `centerflat_q50` の distinct_2 は 0.854、repetition_4 は 0.0345、PPL は 37.0、`centerflat_q75` は 0.824、0.0385、27.6 だった。`random` は distinct_2 0.921、repetition_4 0.0073 だが PPL 85.3 で、ランダム候補数を増やすだけでは確率的な接地が弱い。

q50/q75 は全指標で最良ではないが、ランダムより尤度に近く、minflat より破綻しにくく、greedy/beam より多様性がある中間領域と読むのが妥当である。q75 を普遍的最適値とは解釈できない。

### 4. N の増加は効果よりコストを先に増やす

GPT-2 の `centerflat_q75` では、N=8→32 で runtime は 24.2→71.3 秒、distinct_2 は 0.824→0.789、repetition_4 は 0.038→0.036、PPL は 27.6→23.0 だった。制御は強くなるが、質的な改善は小さく、計算量は約 3 倍である。

`minflat` は N を増やしても低 dispersion・低尤度の傾向が改善しない。N は目的関数の誤りを修正するのではなく、誤った方向への選択圧を強める。

### 5. モデル依存性はかなり強い

Qwen では minflat が multilingual garbage や collapse を伴う一方、Gemma では rain/metal/regret の atmospheric opening が目立つ。両者に共通するのは、successor dispersion によって method ごとの差が観測できる点であり、具体的な語彙バイアスや最適 quantile ではない。

転移するのは制御軸の存在であって、制御軸上の安全な位置ではない。

## 私の予測

以下は未検証の予測であり、現在のデータからの外挿である。

### 予測 A: 有効な selector は単調な min/max ではなく target band になる

モデル・prompt・temperature を増やしても、minflat から q50/q75 に移ると接地性が改善し、q75 を越えて maxflat 側へ移ると散漫さが増える、という逆 U 字型の関係が最も再現すると予想する。固定 quantile より、prompt ごとの baseline dispersion の分位点を使った band 制御の方が頑健になる。

### 予測 B: flatness 単独より「flatness × current likelihood」が強い

minflat の失敗は、低 dispersion を満たす候補の current rank が極端に悪いことに対応している。候補 score に current negative logprob の下限を加える hybrid を使えば、minflat の破綻が大きく減ると予想する。これは q50/q75 の良さを保ちつつ、モデルごとの尺度差にも対応しやすい。

### 予測 C: N=8〜16 が実用上の knee になる

N=32 は選択圧を強めるが、WritingPrompts では多くの指標の改善が飽和し、runtime だけが増える。追加モデルでも、N=8→16 には差が残るが、16→32 は小さくなると予想する。N=32 は実用設定ではなく、失敗モードの診断設定として使うべきである。

### 予測 D: morphology proxy は selector の説明変数になる

`top1_prob`、tail mass、gap、rank slope は、単一の flatness 値より「なぜその候補が危険か」を説明できる可能性がある。特に低 dispersion かつ大きな top-1 gap／低 tail mass の状態は、minflat の逸脱候補を予測するはずである。ただし現行の repeat-grid report には feature rows が 0 の実験があるため、20×20 の実データを再投入してから検証する必要がある。

## 研究上の判断

現時点で採用すべき主張は次の一文である。

> Successor logit dispersion は品質そのものではなく、尤度・多様性・反復・逸脱のトレードオフを動かす decoding-time control axis であり、最も安全な領域はモデル依存の中間 band に現れる。

避けるべき主張は「flatness が品質を改善する」「q75 が最適」「モデルをまたいで同じ方向に制御できる」である。これらは現在のデータでは支持されない。

## 次に検証すべき最小実験

1. GPT-2 / Qwen / Gemma で N=8,16、q25,q50,q75,maxflat,minflat を同一 prompt・同一 seed で再実行する。
2. 各 token の current logprob、successor flatness、top1 probability、tail mass、gap を保存する。
3. `minflat` に current-rank または logprob の下限を加えた hybrid を比較する。
4. prompt 単位の paired delta と bootstrap CI を出し、method 平均だけでなく prompt 間の再現性を評価する。

## 根拠ファイル

- `results/summary/writingprompts_key_results.csv`
- `results/summary/comparison_gpt2_writingprompts_table.csv`
- `results/analysis/successor_dispersion_likelihood_diagnostics.md`
- `results/analysis/qwen2_5_1_5b_vs_gemma_wrapped_comparison.md`
- `results/analysis/morphology_repeat_grid_distilgpt2_raw/successor_distribution_morphology_report.md`
- `results/analysis/paper_claim_audit.md`
