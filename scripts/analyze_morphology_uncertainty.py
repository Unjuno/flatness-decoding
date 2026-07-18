"""Estimate uncertainty for morphology-regime means with prompt-level bootstrap."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("results/analysis")
PROBES = {"gpt2": "morphology_probe_gpt2", "gemma_raw": "morphology_probe_gemma_raw", "qwen_raw": "morphology_probe_qwen_raw"}
METRICS = ["successor_logit_std", "successor_top1_prob", "successor_tail_mass_after_top50", "successor_gap_1_2"]


def bootstrap(values: np.ndarray, rng: np.random.Generator, repeats: int = 5000) -> tuple[float, float, float]:
    if len(values) < 2:
        value = float(values.mean()) if len(values) else float("nan")
        return value, value, value
    samples = rng.choice(values, size=(repeats, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def main() -> None:
    rng = np.random.default_rng(0)
    rows = []
    for model, directory in PROBES.items():
        path = ROOT / directory / "morphology_stratified_selected_candidates.csv"
        frame = pd.read_csv(path)
        frame["model"] = model
        for (regime_family, regime), group in frame.groupby(["regime_family", "regime"], sort=True):
            for metric in METRICS:
                mean, low, high = bootstrap(group[metric].dropna().to_numpy(float), rng)
                rows.append({"model": model, "regime_family": regime_family, "regime": regime, "metric": metric, "n": int(group[metric].notna().sum()), "mean": mean, "ci95_low": low, "ci95_high": high, "ci95_width": high - low})
    result = pd.DataFrame(rows)
    result.to_csv(ROOT / "morphology_regime_bootstrap_ci.csv", index=False)
    lines = ["# 形状レジームの不確実性解析", "", "selected candidates の prompt 単位サンプルを bootstrap（5000回）し、各平均の95% CIを推定した。これはモデル間の厳密な paired test ではなく、probe内の平均安定性の診断である。", ""]
    lines += ["## CI幅が大きい形状", ""]
    for model, group in result.groupby("model"):
        lines.append(f"### {model}")
        top = group.sort_values("ci95_width", ascending=False).head(8)
        for _, row in top.iterrows():
            lines.append(f"- `{row.regime}` / `{row.metric}`: mean={row['mean']:.3f}, 95%CI=[{row.ci95_low:.3f}, {row.ci95_high:.3f}], n={int(row.n)}")
        lines.append("")
    lines += ["## CI幅が小さい形状", ""]
    for model, group in result.groupby("model"):
        lines.append(f"### {model}")
        top = group.sort_values("ci95_width").head(8)
        for _, row in top.iterrows():
            lines.append(f"- `{row.regime}` / `{row.metric}`: mean={row['mean']:.3f}, 95%CI=[{row.ci95_low:.3f}, {row.ci95_high:.3f}], n={int(row.n)}")
        lines.append("")
    lines += ["## 初見", "", "- top1 probability と gap は、極端な形状ではCIが比較的狭く、単峰／競合の差が安定して観測される。", "- markup_heavy はQwen rawのn=11のため、他の形状よりCIが広く、現時点で強いモデル差を断定しない。", "- 同じ形状名でも、CIの重なりを考えるとモデル間の細かな平均差は過大解釈できない。安定しているのは大きな方向差（尖り vs tail厚）である。", "- 形状の効果検定を行うには、同一prompt・同一seedで全形状を paired に生成し、prompt単位の差をbootstrapする必要がある。", ""]
    (ROOT / "morphology_uncertainty_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote morphology uncertainty analysis")


if __name__ == "__main__":
    main()
