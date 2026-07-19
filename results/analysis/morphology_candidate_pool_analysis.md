# Candidate-pool 全体の形状解析

選択済み20件だけでなく、各モデルの top-100 candidate pool 全体を `assigned_regimes` で展開して集計した。したがって、形状の発生頻度と選択結果を分離して読める。

## カバレッジ

| regime               |   ('candidate_rows', 'gemma_raw') |   ('candidate_rows', 'gpt2') |   ('candidate_rows', 'qwen_raw') |   ('prompts', 'gemma_raw') |   ('prompts', 'gpt2') |   ('prompts', 'qwen_raw') |
|:---------------------|----------------------------------:|-----------------------------:|---------------------------------:|---------------------------:|----------------------:|--------------------------:|
| fragment_heavy       |                               501 |                          501 |                              501 |                         20 |                    20 |                        20 |
| heavy_tail           |                               755 |                          800 |                              800 |                         20 |                    20 |                        20 |
| high_concentration   |                               800 |                          800 |                              800 |                         20 |                    20 |                        20 |
| high_spread          |                               800 |                          800 |                              800 |                         20 |                    20 |                        20 |
| large_gap            |                               800 |                          800 |                              800 |                         20 |                    20 |                        20 |
| list_heavy           |                               501 |                          403 |                              260 |                         20 |                    20 |                        20 |
| low_concentration    |                               800 |                          800 |                              800 |                         20 |                    20 |                        20 |
| low_spread           |                               800 |                          800 |                              800 |                         20 |                    20 |                        20 |
| markup_heavy         |                               134 |                           37 |                               11 |                         18 |                     9 |                         7 |
| mid_concentration    |                               400 |                          400 |                              400 |                         20 |                    20 |                        20 |
| mid_gap              |                               400 |                          400 |                              400 |                         20 |                    20 |                        20 |
| mid_spread           |                               400 |                          400 |                              400 |                         20 |                    20 |                        20 |
| mid_tail             |                               400 |                          400 |                              400 |                         20 |                    20 |                        20 |
| prompt_overlap_heavy |                               501 |                          501 |                              501 |                         20 |                    20 |                        20 |
| recent_overlap_heavy |                               501 |                          501 |                              501 |                         20 |                    20 |                        20 |
| small_gap            |                               800 |                          800 |                              800 |                         20 |                    20 |                        20 |
| thin_tail            |                               845 |                          800 |                              800 |                         20 |                    20 |                        20 |

## 初見

- `markup_heavy` の不足は、選択アルゴリズムだけでなく candidate pool 内で該当 prompt が少ないことが原因である。Qwen raw のtop-100 candidate poolは7 promptしかカバーしないため、top-100 条件のまま20 promptを独立に揃えることはできない（selected候補の件数11とは別の値）。
- 旧step-0 probeでは `recent_text=prompt` を渡していたため、既存CSVの `prompt_overlap_heavy` と `recent_overlap_heavy` は独立な証拠として使えない。実装は修正済みだが、既存CSVは再計算が必要である。
- selected sample の平均が極端に見える形状は、candidate pool 全体の平均と比較しないと、形状の性質と選択バイアスを混同する。

## 根拠

- `morphology_candidate_pool_summary.csv`
- 各モデルの `morphology_stratified_candidate_morphology.csv`
