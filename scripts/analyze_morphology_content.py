"""Inspect candidate-token content for the 18 shape labels."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path("results/analysis")
PROBES = {"gpt2": "morphology_probe_gpt2", "gemma_raw": "morphology_probe_gemma_raw", "qwen_raw": "morphology_probe_qwen_raw"}
SHAPE_RULES = {
    "rank_slope_flat": lambda r: r["successor_rank_prob_slope_top50"] >= -0.05,
    "rank_slope_mid": lambda r: -0.20 < r["successor_rank_prob_slope_top50"] < -0.05,
    "rank_slope_steep": lambda r: r["successor_rank_prob_slope_top50"] <= -0.20,
    "spiky_top": lambda r: r["successor_top1_prob"] >= 0.60 and r["successor_gap_1_2"] >= 1.0,
    "smooth_top": lambda r: r["successor_top1_prob"] <= 0.20 and r["successor_gap_1_2"] <= 0.15,
    "multi_peak": lambda r: r["successor_large_gap_count_top50"] >= 2 and r["successor_gap_std_top50"] >= 0.5,
}


def token_kind(token: str) -> str:
    text = str(token).replace("\\n", "\n").replace("\\t", "\t")
    if not text.strip():
        return "whitespace"
    if re.search(r"<|>|</|#|\*|[-–—]", text):
        return "markup_or_bullet"
    if text.isdigit() or any(ch.isdigit() for ch in text):
        return "digit"
    if re.fullmatch(r"[^\w\s]+", text, flags=re.UNICODE):
        return "punctuation"
    if re.search(r"[A-Za-zÀ-ÿぁ-んァ-ヶ一-龯]", text):
        return "word_fragment"
    return "other"


def main() -> None:
    rows = []
    for model, directory in PROBES.items():
        frame = pd.read_csv(ROOT / directory / "morphology_stratified_candidate_morphology.csv")
        frame["model"] = model
        for _, row in frame.iterrows():
            for shape, rule in SHAPE_RULES.items():
                if rule(row):
                    token = "" if pd.isna(row["candidate_token"]) else str(row["candidate_token"])
                    rows.append({"model": model, "shape": shape, "prompt_id": row["prompt_id"], "candidate_rank": row["candidate_rank"], "candidate_token": token, "token_kind": token_kind(token), "token_length": len(token), "candidate_current_prob": row["candidate_current_prob"], "candidate_current_rank": row["candidate_current_rank"]})
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "morphology_18_shape_content_rows.csv", index=False)
    counts = frame.groupby(["model", "shape", "token_kind"], as_index=False).size().rename(columns={"size": "rows"})
    totals = counts.groupby(["model", "shape"])["rows"].transform("sum")
    counts["rate"] = counts["rows"] / totals
    counts.to_csv(ROOT / "morphology_18_shape_content_summary.csv", index=False)

    lines = ["# 18形状の内容的特徴解析", "", "候補 token の文字列を直接調べた。これは生成文の意味評価ではなく、形状ごとにどの種類の token が選ばれやすいかの解析である。", "", "## token kind 比率", "", "| model | shape | word/fragment | punctuation | markup/bullet | digit | whitespace | other |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for (model, shape), group in counts.groupby(["model", "shape"], sort=True):
        values = dict(zip(group.token_kind, group.rate))
        lines.append(f"| {model} | `{shape}` | {values.get('word_fragment', 0):.2f} | {values.get('punctuation', 0):.2f} | {values.get('markup_or_bullet', 0):.2f} | {values.get('digit', 0):.2f} | {values.get('whitespace', 0):.2f} | {values.get('other', 0):.2f} |")

    lines += ["", "## 上位 token の例", ""]
    for (model, shape), group in frame.groupby(["model", "shape"], sort=True):
        top = group["candidate_token"].value_counts().head(8)
        examples = ", ".join(f"{repr(token)}×{count}" for token, count in top.items())
        lines.append(f"- {model} / `{shape}`: {examples}")

    lines += ["", "## 直接的な初見", "", "- これまでの統計解析だけでは内容差を見落としていた。候補 token を見ると、形状差は語彙意味そのものより、word fragment / whitespace / punctuation / markup の選択傾向として現れる可能性が高い。", "- `spiky_top` はモデルによっては特定の一般語 token に集中し、`smooth_top` は whitespace・punctuation を含む競合状態になりやすい。", "- `multi_peak` は内容が多様というより、複数の token 群（語、記号、形式 token）が競合している状態である可能性がある。", "- ただし候補 token だけでは、その token を選んだ後の文章内容までは分からない。形状別の意味的差を主張するには、各候補から continuation を生成して比較する必要がある。", "- 現在のデータには morphology probe の continuation が保存されていないため、「内容的に差がない」と結論するのは不適切であり、今回の解析は内容差の入口に留まる。", ""]
    (ROOT / "morphology_content_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote morphology content analysis")


if __name__ == "__main__":
    main()
