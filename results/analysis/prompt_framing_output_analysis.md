# raw／wrapped 出力ログ比較

同じ `prompt_id` の保存済み出力を、raw（`shapes_20`）と wrapped（`shapes`）で突合した。raw／wrappedでprompt本文が異なるため、これはプロンプト形式の効果を含む記述比較であり、形状の因果検定ではない。

- 対象モデル: 2
- 共通prompt数: 20
- 共通method数: 2

## 形状methodの解釈上の注意

`shapes_20` の形状methodはraw側にだけ存在する。したがって、raw／wrapped差を形状ごとに直接比較するには、wrapped側にも同じ形状methodを保存したログが必要である。今回の比較は `greedy` と `beam` の共有methodに限定した。

## 観測された差

- **gemma / beam**: raw→wrapped の平均語数 86.0→91.8; 反復率 0.005→0.000; マークアップ率 0.016→0.000
- **gemma / greedy**: raw→wrapped の平均語数 88.3→92.0; 反復率 0.027→0.001; マークアップ率 0.020→0.001
- **qwen / greedy**: raw→wrapped の平均語数 100.8→99.2; 反復率 0.026→0.000; マークアップ率 0.005→0.000

## 結論

1. wrapped条件は出力の長さ・改行・形式を変え得るため、rawの形状ログからwrappedの生成挙動を推定してはいけない。
2. 現存ログだけでは、18形状それぞれのraw／wrapped対応を完全には検定できない。形状methodをwrapped条件でも保存したtoken logが必要である。
3. それでも共通methodの比較は、モデル固有のprompt framing感度を分類する基準として利用できる。
