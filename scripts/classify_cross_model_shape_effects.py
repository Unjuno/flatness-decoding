"""Classify morphology effects by cross-model direction in paired deltas."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def direction(values: list[float], threshold: float = 0.005) -> str:
    if len(values) < 3:
        return "insufficient_models"
    if all(v > threshold for v in values):
        return "consistently_increases"
    if all(v < -threshold for v in values):
        return "consistently_decreases"
    if max(values) - min(values) <= threshold * 2:
        return "mostly_neutral"
    return "model_dependent"


def main() -> None:
    data = pd.read_csv(ROOT / "results/analysis/morphology_shape_paired_summary.csv")
    rows = []
    for shape, group in data.groupby("shape"):
        repeat = group["delta_repeat_rate_mean"].tolist()
        markup = group["delta_markup_rate_mean"].tolist()
        words = group["delta_output_words_mean"].tolist()
        record = {"shape": shape, "repeat_effect_class": direction(repeat, 0.005), "markup_effect_class": direction(markup, 0.001), "length_effect_class": direction(words, 2.0)}
        for metric, vals in [("repeat", repeat), ("markup", markup), ("length", words)]:
            for model, value in zip(sorted(group["model"]), vals):
                record[f"{metric}_{model}_delta"] = value
        rows.append(record)
    result = pd.DataFrame(rows).sort_values("shape")
    result.to_csv(ROOT / "results/analysis/morphology_cross_model_effect_classes.csv", index=False)
    lines = ["# 形状のモデル横断効果分類", "", "paired greedy差分を、形状ごとに3モデルの効果方向へ集約した。閾値は記述的な実務分類用で、統計的有意差の判定ではない。", "", result[["shape", "repeat_effect_class", "markup_effect_class", "length_effect_class"]].to_markdown(index=False), "", "## 分類の意味", "", "- `consistently_increases/decreases`: 3モデルで同じ方向に変化。モデル横断の制御候補。", "- `model_dependent`: 効果の向きまたは大きさがモデルで分かれる。model-local calibrationが必要。", "- `mostly_neutral`: 指標上の差が小さい。", "", "## 現時点の結論", "", "- 反復率について、全モデルで一貫して増加／減少する形状は限定的で、形状名だけの普遍的な良し悪しは支持されない。", "- 多くの形状はmodel-dependentであり、モデル内percentileと出力リスクを組み合わせる設計が妥当である。", "- 形式崩れと出力長は反復とは異なる方向を示すことがあり、単一のflatness軸に統合すべきではない。"]
    (ROOT / "results/analysis/morphology_cross_model_effect_classes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(result[["shape", "repeat_effect_class", "markup_effect_class", "length_effect_class"]].to_string(index=False))


if __name__ == "__main__":
    main()
