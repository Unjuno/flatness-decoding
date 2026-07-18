# 未見形状への出力モード分類汎化

形状をgroupとしてLeave-One-Shape-Out評価した。訓練に含まれない形状のprompt出力を、token log特徴だけで分類できるかを検証している。

| model   | target        |   folds |   balanced_accuracy_mean |   balanced_accuracy_std |
|:--------|:--------------|--------:|-------------------------:|------------------------:|
| gemma   | repeat_target |       7 |                 0.637009 |                0.185421 |
| gemma   | format_target |      18 |                 0.552387 |                0.070857 |
| gpt2    | repeat_target |       9 |                 0.708194 |                0.181773 |
| gpt2    | format_target |      17 |                 0.63581  |                0.184784 |
| qwen    | repeat_target |       8 |                 0.574058 |                0.078567 |
| qwen    | format_target |      18 |                 0.590309 |                0.102195 |

## 解釈

- 0.5付近なら、前回のprompt GroupKFold性能は形状固有の相関またはサンプル構成に依存している可能性が高い。
- 0.5を継続的に上回るなら、形状を跨いだ連続的なtoken-state信号がある可能性がある。
- fold内の陽性数が少ない場合があるため、by-shape CSVも確認し、平均値だけで決定的結論を出さない。
