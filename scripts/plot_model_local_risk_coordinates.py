"""Plot model-local chosen-probability/flatness risk coordinates."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    data = pd.read_csv(ROOT / "results/analysis/morphology_model_local_risk_coordinates.csv")
    data["risk_quadrant"] = ((data["mean_chosen_prob_z"] > 0) & (data["mean_chosen_flatness_z"] < 0)).map({True: "high_prob_low_flatness", False: "other"})
    quad = data.groupby(["model", "risk_quadrant"], as_index=False).agg(n=("prompt_id", "size"), repeat_rate=("repeat_target", "mean"), mean_repeat=("repeated_4gram_rate", "mean"))
    quad.to_csv(ROOT / "results/analysis/morphology_model_local_risk_quadrants.csv", index=False)
    out = ROOT / "results/figures/morphology/model_local_prob_flatness_risk.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, (model, frame) in zip(axes, data.groupby("model")):
        sc = ax.scatter(frame["mean_chosen_prob_z"], frame["mean_chosen_flatness_z"], c=frame["repeated_4gram_rate"], cmap="magma", vmin=0, vmax=max(0.05, data["repeated_4gram_rate"].quantile(0.98)), s=22, alpha=0.75, edgecolors="none")
        ax.axvline(0, color="0.7", lw=0.8)
        ax.axhline(0, color="0.7", lw=0.8)
        ax.axvspan(0, ax.get_xlim()[1], ymin=0, ymax=0.5, color="tab:red", alpha=0.07)
        ax.set_title(model)
        ax.set_xlabel("chosen probability z-score")
        ax.set_ylabel("chosen flatness z-score")
    fig.colorbar(sc, ax=axes, label="4-gram repeat rate")
    fig.suptitle("Model-local output risk: high chosen probability × low flatness")
    fig.savefig(out, dpi=180)
    plt.close(fig)
    lines = ["# model-local risk coordinate figure", "", "![risk coordinate figure](../figures/morphology/model_local_prob_flatness_risk.png)", "", "赤く薄く示した右下領域は、chosen probabilityがモデル平均より高く、chosen flatnessが低い領域である。前回の分類器で反復リスクと共通して関連した2軸を表示した。", "", "## quadrant summary", "", quad.to_markdown(index=False), "", "この図は相関を示すもので、制御閾値の確定には独立prompt・seedでの再検証が必要である。"]
    (ROOT / "results/analysis/morphology_model_local_risk_figure.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
