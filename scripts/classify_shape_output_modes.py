"""Classify saved shape continuations into auditable surface output modes."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {
    "gpt2": "outputs/generations/comparison_gpt2_shapes_20",
    "gemma": "outputs/generations/comparison_gemma3_4b_shapes_20",
    "qwen": "outputs/generations/comparison_qwen25_1_5b_shapes_20",
}
SHAPES = {"low_spread", "mid_spread", "high_spread", "low_concentration", "mid_concentration", "high_concentration", "thin_tail", "mid_tail", "heavy_tail", "small_gap", "mid_gap", "large_gap", "rank_slope_flat", "rank_slope_mid", "rank_slope_steep", "spiky_top", "smooth_top", "multi_peak"}


def classify(text: str) -> dict[str, float | str]:
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    words = re.findall(r"\S+", text)
    lower = text.lower()
    list_lines = sum(bool(re.match(r"^(?:[-*•]|\d+[.)]|#{1,6}\s)", x)) for x in lines)
    meta = bool(re.search(r"\b(?:title|characters?|objectives?|references?|summary|name:|age:|rank:|unit:|prompt:)\b", lower))
    instruction = bool(re.search(r"\b(?:do not|please|you should|write a|continue the following|answer the question)\b", lower))
    narrative = bool(re.search(r"\b(?:the|he|she|they|we|i)\b", lower) and re.search(r"\b(?:was|were|said|felt|looked|walked|stood|ran|turned|saw|heard)\b", lower))
    grams = [" ".join(w.lower() for w in words[i : i + 4]) for i in range(max(0, len(words) - 3))]
    repeat = (len(grams) - len(set(grams))) / max(1, len(grams))
    markup = sum(c in "[]{}<>*#`_" for c in text) / max(1, len(text))
    return {
        "list_mode": float(list_lines >= 2 or (list_lines >= 1 and meta)),
        "meta_mode": float(meta),
        "instruction_mode": float(instruction),
        "narrative_mode": float(narrative and not meta),
        "markup_mode": float(markup >= 0.01),
        "repetition_mode": float(repeat >= 0.05),
        "repeat_rate": repeat,
        "surface_label": "repetition" if repeat >= 0.05 else "list_or_meta" if list_lines >= 2 or meta else "instruction_like" if instruction and not narrative else "narrative_like" if narrative else "fragment_or_other",
    }


def main() -> None:
    rows = []
    for model, directory in TARGETS.items():
        for path in sorted((ROOT / directory).glob("*.jsonl")):
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    shape = str(row.get("method", ""))
                    if shape not in SHAPES:
                        continue
                    rows.append({"model": model, "shape": shape, "prompt_id": row.get("prompt_id"), **classify(str(row.get("continuation_text", "")))})
    data = pd.DataFrame(rows)
    data.to_csv(ROOT / "results/analysis/morphology_shape_output_modes.csv", index=False)
    summary = data.groupby(["shape", "model"], as_index=False).agg(
        prompt_count=("prompt_id", "nunique"),
        **{c: (c, "mean") for c in ["list_mode", "meta_mode", "instruction_mode", "narrative_mode", "markup_mode", "repetition_mode", "repeat_rate"]},
    )
    summary.to_csv(ROOT / "results/analysis/morphology_shape_output_modes_summary.csv", index=False)
    lines = ["# 形状別の出力モード分類", "", "保存済み `shapes_20` continuation を、監査可能な表層ルールで分類した。これは意味品質の判定ではなく、出力形式・反復・物語らしさの診断である。", ""]
    for shape, group in summary.groupby("shape", sort=True):
        best = group.loc[group["narrative_mode"].idxmax()]
        worst = group.loc[group["repetition_mode"].idxmax()]
        lines.append(f"- **{shape}**: narrative_like最大={best['model']} ({best['narrative_mode']:.2f}); repetition最大={worst['model']} ({worst['repetition_mode']:.2f})")
    lines += ["", "## 注意", "", "- regexベースなので、文脈理解や文章品質を保証しない。", "- `list_or_meta` と `instruction_like` は、形状が形式トークンを誘発するかを調べる補助変数である。", "- 形状分類の決定性を上げるには、同一prompt・同一seedで候補形状をpairedにした本文比較が必要である。"]
    (ROOT / "results/analysis/morphology_shape_output_modes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(data)} mode rows")


if __name__ == "__main__":
    main()
