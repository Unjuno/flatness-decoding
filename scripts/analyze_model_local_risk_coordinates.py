"""Evaluate model-local normalized token-state coordinates for output risk."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
RAW_FEATURES = ["mean_chosen_rank", "mean_chosen_prob", "mean_chosen_flatness", "mean_selection_gain", "fallback_rate"]


def main() -> None:
    data = pd.read_csv(ROOT / "results/analysis/morphology_shape_token_output_link.csv")
    local = data.copy()
    for feature in RAW_FEATURES:
        local[f"{feature}_z"] = local.groupby("model")[feature].transform(lambda x: (x - x.mean()) / x.std(ddof=0))
        local[f"{feature}_pct"] = local.groupby("model")[feature].rank(pct=True)
    local["repeat_target"] = (local["repeated_4gram_rate"] >= 0.05).astype(int)
    local["format_target"] = ((local["markup_marker_rate"] >= 0.01) | (local["non_ascii_rate"] >= 0.02)).astype(int)
    local.to_csv(ROOT / "results/analysis/morphology_model_local_risk_coordinates.csv", index=False)
    zfeatures = [f"{x}_z" for x in RAW_FEATURES]
    rows = []
    lines = ["# model-local risk coordinates", "", "token log特徴をモデル内z-scoreとpercentileへ変換し、モデル間で共通する出力リスク座標があるかを調べた。評価はprompt_id GroupKFoldで行った。", ""]
    scorer = make_scorer(balanced_accuracy_score)
    for target in ["repeat_target", "format_target"]:
        for model, group in local.groupby("model"):
            group = group.dropna(subset=zfeatures)
            if group[target].nunique() < 2: continue
            clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, random_state=0))
            scores = cross_val_score(clf, group[zfeatures], group[target], groups=group["prompt_id"], cv=GroupKFold(n_splits=5), scoring=scorer)
            clf.fit(group[zfeatures], group[target])
            coefs = clf[-1].coef_[0]
            rows.append({"model": model, "target": target, "n": len(group), "positive_rate": group[target].mean(), "balanced_accuracy": scores.mean(), "balanced_accuracy_std": scores.std(), **{f"coef_{f}": c for f, c in zip(zfeatures, coefs)}})
        lines.append(f"## {target}")
        lines.append("")
        lines.append(pd.DataFrame([r for r in rows if r["target"] == target]).to_markdown(index=False))
        lines.append("")
    result = pd.DataFrame(rows)
    result.to_csv(ROOT / "results/analysis/morphology_model_local_risk_classifier.csv", index=False)
    lines += ["## 解釈", "", "- z-score化しても反復リスクの分類性能が0.5を上回り、係数の向きがモデル間で揃う特徴は、共通の制御座標候補である。", "- 係数が逆向きなら、そのtoken-stateの意味がモデルごとに異なる。", "- これは表層反復・形式崩れの診断であり、意味品質の分類ではない。"]
    (ROOT / "results/analysis/morphology_model_local_risk_coordinates.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
