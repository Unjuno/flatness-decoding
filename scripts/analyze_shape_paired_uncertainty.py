"""Quantify uncertainty of shape-vs-greedy paired output effects."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]


def bootstrap(values: np.ndarray, rng: np.random.Generator, reps: int = 5000) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    sample = rng.choice(values, size=(reps, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(sample, 0.025)), float(np.quantile(sample, 0.975))


def main() -> None:
    data = pd.read_csv(ROOT / "results/analysis/morphology_shape_paired_deltas.csv")
    rng = np.random.default_rng(20260719)
    rows = []
    for (shape, model), group in data.groupby(["shape", "model"]):
        for metric in ["delta_repeat_rate", "delta_markup_rate", "delta_non_ascii_rate", "delta_output_words"]:
            values = group[metric].dropna().to_numpy(float)
            ci_low, ci_high = bootstrap(values, rng)
            nonzero = values[np.abs(values) > 1e-12]
            if len(nonzero) >= 5 and np.any(nonzero > 0) and np.any(nonzero < 0):
                test = wilcoxon(nonzero, alternative="two-sided", method="auto")
                p = float(test.pvalue)
            elif len(nonzero) >= 5:
                p = 1.0 if np.all(nonzero == 0) else 0.0
            else:
                p = float("nan")
            rows.append({"shape": shape, "model": model, "metric": metric, "n": len(values), "mean_delta": values.mean(), "median_delta": np.median(values), "ci95_low": ci_low, "ci95_high": ci_high, "positive_fraction": float((values > 0).mean()), "wilcoxon_p": p})
    result = pd.DataFrame(rows)
    result.to_csv(ROOT / "results/analysis/morphology_shape_paired_uncertainty.csv", index=False)
    repeat = result[result["metric"] == "delta_repeat_rate"].copy()
    repeat["ci_excludes_zero"] = (repeat["ci95_low"] > 0) | (repeat["ci95_high"] < 0)
    repeat["p_lt_005"] = repeat["wilcoxon_p"] < 0.05
    lines = ["# 形状paired効果の不確実性解析", "", "各形状×モデルについて、同一promptのgreedyとの差分をbootstrap 95% CIとWilcoxon符号順位検定で評価した。20 promptの小標本であり、18形状×複数指標の多重比較補正は行っていないため、探索的結果である。", "", "## 反復率差分でCIが0を跨がないセル", "", repeat[repeat["ci_excludes_zero"]][["shape", "model", "mean_delta", "ci95_low", "ci95_high", "positive_fraction", "wilcoxon_p"]].sort_values("mean_delta", ascending=False).to_markdown(index=False), "", "## 解釈", "", "- CIが0を跨がないセルを、prompt間でも方向が比較的安定した候補とする。", "- p値だけで形状の効果を断定しない。特に差分がほぼ0のセルや陽性数が少ないセルでは検定力が低い。", "- モデル横断で同じ形状がCI除外になるかを見て、共通制御候補とmodel-dependent候補を分ける。"]
    (ROOT / "results/analysis/morphology_shape_paired_uncertainty.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(repeat[repeat["ci_excludes_zero"]][["shape", "model", "mean_delta", "ci95_low", "ci95_high", "wilcoxon_p"]].to_string(index=False))


if __name__ == "__main__":
    main()
