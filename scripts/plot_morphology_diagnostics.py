"""Create compact figures for cross-model morphology diagnostics."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("results/analysis")
OUT = Path("results/figures/morphology")
MODELS = ["gpt2", "gemma_raw", "qwen_raw"]
METRICS = {
    "mean_successor_logit_std": "successor logit std",
    "mean_successor_top1_prob": "top-1 probability",
    "mean_successor_tail_mass_after_top50": "tail mass after top-50",
    "mean_successor_gap_1_2": "top-1/top-2 gap",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(ROOT / "morphology_deep_summary.csv")
    plt.rcParams.update({"font.size": 9})
    for column, label in METRICS.items():
        pivot = frame.pivot(index="regime", columns="model", values=column)[MODELS]
        fig, ax = plt.subplots(figsize=(12, 8))
        image = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
        fig.colorbar(image, ax=ax, label=label)
        ax.set_xticks(range(len(pivot.columns)), pivot.columns)
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center", color="white")
        ax.set_title(label)
        ax.set_xlabel("model")
        ax.set_ylabel("regime")
        fig.tight_layout()
        fig.savefig(OUT / f"{column}.png", dpi=160)
        plt.close(fig)

    z_rows = []
    for model, group in frame.groupby("model"):
        for column in METRICS:
            values = group[column]
            group = group.copy()
            group[f"z_{column}"] = (values - values.mean()) / values.std(ddof=0)
        z_rows.append(group)
    z_frame = pd.concat(z_rows)
    z_frame.to_csv(ROOT / "morphology_model_local_zscores.csv", index=False)
    pivot = z_frame.pivot(index="regime", columns="model", values="z_mean_successor_top1_prob")[MODELS]
    fig, ax = plt.subplots(figsize=(12, 8))
    image = ax.imshow(pivot.values, aspect="auto", cmap="coolwarm", vmin=-2, vmax=2)
    fig.colorbar(image, ax=ax, label="z-score")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center", color="black")
    ax.set_title("Model-local z-score: successor top-1 probability")
    fig.tight_layout()
    fig.savefig(OUT / "top1_probability_model_local_zscore.png", dpi=160)
    plt.close(fig)
    print(f"wrote figures to {OUT}")


if __name__ == "__main__":
    main()
