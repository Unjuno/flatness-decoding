"""Test whether saved token-log features predict surface output failure modes."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import make_scorer, balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["mean_chosen_rank", "mean_chosen_prob", "mean_chosen_flatness", "mean_selection_gain", "fallback_rate"]


def main() -> None:
    data = pd.read_csv(ROOT / "results/analysis/morphology_shape_token_output_link.csv")
    # Rejoin the shape label explicitly because the linked table has it already.
    data["repeat_target"] = (data["repeated_4gram_rate"] >= 0.05).astype(int)
    data["format_target"] = ((data["markup_marker_rate"] >= 0.01) | (data["non_ascii_rate"] >= 0.02)).astype(int)
    rows = []
    lines = ["# token log特徴から本文出力モードを予測する分類解析", "", "保存済みprompt単位データに対して、token logの連続特徴だけで反復・形式崩れを予測できるかを評価した。DecisionTreeは解釈用の浅いモデルで、prompt_idをGroupKFoldのgroupにして同一promptの漏洩を避けた。", ""]
    scorer = make_scorer(balanced_accuracy_score)
    for model, group in data.groupby("model"):
        group = group.dropna(subset=FEATURES).copy()
        if len(group) < 10:
            continue
        cv = GroupKFold(n_splits=5)
        for target in ["repeat_target", "format_target"]:
            y = group[target]
            if y.nunique() < 2:
                continue
            clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=8, class_weight="balanced", random_state=0)
            scores = cross_val_score(clf, group[FEATURES], y, groups=group["prompt_id"], cv=cv, scoring=scorer)
            clf.fit(group[FEATURES], y)
            rows.append({"model": model, "target": target, "n": len(group), "positive_rate": y.mean(), "balanced_accuracy_mean": scores.mean(), "balanced_accuracy_std": scores.std()})
            lines += [f"## {model} / {target}", "", f"- n={len(group)}, positive_rate={y.mean():.3f}", f"- GroupKFold balanced accuracy={scores.mean():.3f} ± {scores.std():.3f}", "", "```text", export_text(clf, feature_names=FEATURES), "```", ""]
    result = pd.DataFrame(rows)
    result.to_csv(ROOT / "results/analysis/morphology_output_feature_classifier.csv", index=False)
    lines += ["## 解釈", "", "- balanced accuracyが0.5付近なら、token log特徴だけでは安定分類できない。", "- 0.5を明確に上回る場合は、形状名より連続的な選択状態を制御変数にできる可能性がある。", "- これは表層分類の検定であり、意味品質や因果効果の検定ではない。"]
    (ROOT / "results/analysis/morphology_output_feature_classifier.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(result.to_string(index=False) if not result.empty else "no eligible targets")


if __name__ == "__main__":
    main()
