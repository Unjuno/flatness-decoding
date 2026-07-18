"""Evaluate output-mode prediction on completely unseen morphology shapes."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["mean_chosen_rank", "mean_chosen_prob", "mean_chosen_flatness", "mean_selection_gain", "fallback_rate"]


def main() -> None:
    data = pd.read_csv(ROOT / "results/analysis/morphology_shape_token_output_link.csv")
    data["repeat_target"] = (data["repeated_4gram_rate"] >= 0.05).astype(int)
    data["format_target"] = ((data["markup_marker_rate"] >= 0.01) | (data["non_ascii_rate"] >= 0.02)).astype(int)
    rows = []
    details = []
    for model, frame in data.groupby("model"):
        cv = LeaveOneGroupOut()
        for target in ["repeat_target", "format_target"]:
            scores = []
            for train_idx, test_idx in cv.split(frame, frame[target], groups=frame["shape"]):
                train, test = frame.iloc[train_idx], frame.iloc[test_idx]
                if train[target].nunique() < 2 or test[target].nunique() < 2:
                    continue
                clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=8, class_weight="balanced", random_state=0)
                clf.fit(train[FEATURES], train[target])
                pred = clf.predict(test[FEATURES])
                scores.append(balanced_accuracy_score(test[target], pred))
                details.append({"model": model, "target": target, "held_out_shape": test["shape"].iloc[0], "n_test": len(test), "positive_test": int(test[target].sum()), "balanced_accuracy": scores[-1]})
            rows.append({"model": model, "target": target, "folds": len(scores), "balanced_accuracy_mean": sum(scores) / len(scores) if scores else float("nan"), "balanced_accuracy_std": pd.Series(scores).std(ddof=0) if scores else float("nan")})
    summary = pd.DataFrame(rows)
    summary.to_csv(ROOT / "results/analysis/morphology_shape_holdout_classifier.csv", index=False)
    pd.DataFrame(details).to_csv(ROOT / "results/analysis/morphology_shape_holdout_classifier_by_shape.csv", index=False)
    lines = ["# 未見形状への出力モード分類汎化", "", "形状をgroupとしてLeave-One-Shape-Out評価した。訓練に含まれない形状のprompt出力を、token log特徴だけで分類できるかを検証している。", "", summary.to_markdown(index=False), "", "## 解釈", "", "- 0.5付近なら、前回のprompt GroupKFold性能は形状固有の相関またはサンプル構成に依存している可能性が高い。", "- 0.5を継続的に上回るなら、形状を跨いだ連続的なtoken-state信号がある可能性がある。", "- fold内の陽性数が少ない場合があるため、by-shape CSVも確認し、平均値だけで決定的結論を出さない。"]
    (ROOT / "results/analysis/morphology_shape_holdout_classifier.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
