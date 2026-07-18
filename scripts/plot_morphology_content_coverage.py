"""Plot content-token composition and 18-shape observation coverage."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("results/analysis")
OUT = Path("results/figures/morphology")
MODELS = ["gpt2", "gemma_raw", "qwen_raw"]


def draw_matrix(frame: pd.DataFrame, index: str, columns: list[str], value: str, title: str, output: Path, fmt: str = ".2f") -> None:
    pivot = frame.pivot(index=index, columns="model", values=value).reindex(columns=MODELS).fillna(0)
    fig, ax = plt.subplots(figsize=(12, max(6, 0.35 * len(pivot))))
    image = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    fig.colorbar(image, ax=ax, label=value)
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, format(pivot.iloc[i, j], fmt), ha="center", va="center", color="white")
    ax.set_title(title)
    ax.set_xlabel("model")
    ax.set_ylabel(index)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    content = pd.read_csv(ROOT / "morphology_18_shape_content_summary.csv")
    content["shape_kind"] = content["shape"] + " / " + content["token_kind"]
    draw_matrix(content, "shape_kind", [], "rate", "Candidate token-kind rate by shape", OUT / "content_token_kind_rate.png")

    coverage = pd.read_csv(ROOT / "morphology_18_shape_summary.csv")
    draw_matrix(coverage, "shape", [], "prompts", "Observed prompt coverage for the 18 shape labels", OUT / "shape_prompt_coverage.png", ".0f")
    draw_matrix(coverage, "shape", [], "rows", "Observed candidate-row coverage for the 18 shape labels", OUT / "shape_candidate_coverage.png", ".0f")
    print(f"wrote content and coverage figures to {OUT}")


if __name__ == "__main__":
    main()
