# 形状レジームの不確実性解析

selected candidates の prompt 単位サンプルを bootstrap（5000回）し、各平均の95% CIを推定した。これはモデル間の厳密な paired test ではなく、probe内の平均安定性の診断である。

## CI幅が大きい形状

### gemma_raw
- `markup_heavy` / `successor_gap_1_2`: mean=2.035, 95%CI=[1.178, 3.016], n=20
- `thin_tail` / `successor_gap_1_2`: mean=5.836, 95%CI=[4.909, 6.734], n=20
- `low_spread` / `successor_gap_1_2`: mean=5.925, 95%CI=[5.012, 6.777], n=20
- `large_gap` / `successor_gap_1_2`: mean=6.053, 95%CI=[5.176, 6.881], n=20
- `fragment_heavy` / `successor_gap_1_2`: mean=6.305, 95%CI=[5.416, 7.113], n=20
- `high_concentration` / `successor_gap_1_2`: mean=6.053, 95%CI=[5.192, 6.877], n=20
- `list_heavy` / `successor_gap_1_2`: mean=1.788, 95%CI=[1.180, 2.400], n=20
- `mid_spread` / `successor_gap_1_2`: mean=1.776, 95%CI=[1.179, 2.398], n=20

### gpt2
- `high_spread` / `successor_gap_1_2`: mean=3.350, 95%CI=[2.027, 4.706], n=20
- `thin_tail` / `successor_gap_1_2`: mean=5.166, 95%CI=[4.263, 6.010], n=20
- `high_concentration` / `successor_gap_1_2`: mean=5.345, 95%CI=[4.527, 6.133], n=20
- `low_spread` / `successor_gap_1_2`: mean=2.936, 95%CI=[2.198, 3.642], n=20
- `fragment_heavy` / `successor_gap_1_2`: mean=5.492, 95%CI=[4.844, 6.167], n=20
- `large_gap` / `successor_gap_1_2`: mean=5.492, 95%CI=[4.856, 6.163], n=20
- `list_heavy` / `successor_gap_1_2`: mean=5.492, 95%CI=[4.845, 6.150], n=20
- `mid_spread` / `successor_gap_1_2`: mean=1.361, 95%CI=[0.783, 2.014], n=20

### qwen_raw
- `markup_heavy` / `successor_gap_1_2`: mean=1.962, 95%CI=[0.947, 3.052], n=11
- `thin_tail` / `successor_gap_1_2`: mean=1.434, 95%CI=[0.777, 2.299], n=20
- `high_concentration` / `successor_gap_1_2`: mean=1.718, 95%CI=[1.137, 2.525], n=20
- `large_gap` / `successor_gap_1_2`: mean=1.758, 95%CI=[1.221, 2.559], n=20
- `list_heavy` / `successor_gap_1_2`: mean=1.221, 95%CI=[0.625, 1.940], n=20
- `fragment_heavy` / `successor_gap_1_2`: mean=1.805, 95%CI=[1.217, 2.530], n=20
- `low_spread` / `successor_gap_1_2`: mean=1.229, 95%CI=[0.846, 1.708], n=20
- `mid_concentration` / `successor_gap_1_2`: mean=0.791, 95%CI=[0.509, 1.112], n=20

## CI幅が小さい形状

### gemma_raw
- `fragment_heavy` / `successor_tail_mass_after_top50`: mean=0.001, 95%CI=[0.000, 0.001], n=20
- `large_gap` / `successor_tail_mass_after_top50`: mean=0.001, 95%CI=[0.001, 0.002], n=20
- `high_concentration` / `successor_tail_mass_after_top50`: mean=0.001, 95%CI=[0.001, 0.002], n=20
- `mid_tail` / `successor_tail_mass_after_top50`: mean=0.013, 95%CI=[0.011, 0.014], n=20
- `thin_tail` / `successor_tail_mass_after_top50`: mean=0.002, 95%CI=[0.001, 0.004], n=20
- `low_spread` / `successor_tail_mass_after_top50`: mean=0.003, 95%CI=[0.001, 0.005], n=20
- `markup_heavy` / `successor_tail_mass_after_top50`: mean=0.017, 95%CI=[0.005, 0.033], n=20
- `list_heavy` / `successor_tail_mass_after_top50`: mean=0.031, 95%CI=[0.018, 0.046], n=20

### gpt2
- `mid_concentration` / `successor_top1_prob`: mean=0.196, 95%CI=[0.186, 0.207], n=20
- `mid_tail` / `successor_tail_mass_after_top50`: mean=0.256, 95%CI=[0.245, 0.267], n=20
- `low_concentration` / `successor_top1_prob`: mean=0.096, 95%CI=[0.081, 0.110], n=20
- `small_gap` / `successor_top1_prob`: mean=0.099, 95%CI=[0.077, 0.121], n=20
- `thin_tail` / `successor_tail_mass_after_top50`: mean=0.062, 95%CI=[0.034, 0.092], n=20
- `list_heavy` / `successor_tail_mass_after_top50`: mean=0.061, 95%CI=[0.034, 0.094], n=20
- `large_gap` / `successor_tail_mass_after_top50`: mean=0.061, 95%CI=[0.033, 0.094], n=20
- `fragment_heavy` / `successor_tail_mass_after_top50`: mean=0.061, 95%CI=[0.033, 0.095], n=20

### qwen_raw
- `mid_tail` / `successor_tail_mass_after_top50`: mean=0.131, 95%CI=[0.123, 0.140], n=20
- `mid_spread` / `successor_logit_std`: mean=3.001, 95%CI=[2.991, 3.011], n=20
- `mid_concentration` / `successor_top1_prob`: mean=0.254, 95%CI=[0.244, 0.264], n=20
- `thin_tail` / `successor_tail_mass_after_top50`: mean=0.054, 95%CI=[0.041, 0.066], n=20
- `low_concentration` / `successor_top1_prob`: mean=0.121, 95%CI=[0.095, 0.148], n=20
- `heavy_tail` / `successor_top1_prob`: mean=0.157, 95%CI=[0.123, 0.193], n=20
- `mid_tail` / `successor_top1_prob`: mean=0.232, 95%CI=[0.196, 0.269], n=20
- `high_concentration` / `successor_tail_mass_after_top50`: mean=0.110, 95%CI=[0.072, 0.150], n=20

## 初見

- top1 probability と gap は、極端な形状ではCIが比較的狭く、単峰／競合の差が安定して観測される。
- markup_heavy はQwen rawのn=11のため、他の形状よりCIが広く、現時点で強いモデル差を断定しない。
- 同じ形状名でも、CIの重なりを考えるとモデル間の細かな平均差は過大解釈できない。安定しているのは大きな方向差（尖り vs tail厚）である。
- 形状の効果検定を行うには、同一prompt・同一seedで全形状を paired に生成し、prompt単位の差をbootstrapする必要がある。

