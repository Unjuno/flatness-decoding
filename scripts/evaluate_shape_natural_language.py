"""Evaluate all saved shape continuations with an auditable NLP surface rubric.

This is a reproducible lexical evaluation, not a claim of human semantic judgment.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {"gpt2": "outputs/generations/comparison_gpt2_shapes_20", "gemma": "outputs/generations/comparison_gemma3_4b_shapes_20", "qwen": "outputs/generations/comparison_qwen25_1_5b_shapes_20"}
SHAPES = {"low_spread", "mid_spread", "high_spread", "low_concentration", "mid_concentration", "high_concentration", "thin_tail", "mid_tail", "heavy_tail", "small_gap", "mid_gap", "large_gap", "rank_slope_flat", "rank_slope_mid", "rank_slope_steep", "spiky_top", "smooth_top", "multi_peak"}
STOP = {"the", "and", "that", "this", "with", "from", "your", "you", "were", "have", "into", "about", "they", "their", "what", "when", "where", "which", "would", "could", "there", "then", "than", "been", "will", "story", "prompt"}
ACTION = r"(?:was|were|said|asked|felt|looked|walked|stood|ran|turned|saw|heard|thought|found|opened|closed|went|came|took|made|held|watched|replied|whispered|knew)"
META = r"(?:title|summary|objectives?|references?|characters?|name:|age:|rank:|unit:|equipment|prompt:|instructions?|answer:|analysis)"


def words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z'-]{2,}", text.lower())


def score(text: str, prompt: str) -> dict[str, float]:
    ws = words(text)
    ps = set(words(prompt)) - STOP
    out_set = set(ws)
    grams = [" ".join(ws[i : i + 4]) for i in range(max(0, len(ws) - 3))]
    action_count = len(re.findall(rf"\b{ACTION}\b", text.lower()))
    sentence_count = max(1, len(re.findall(r"[.!?]+", text)))
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    list_count = sum(bool(re.match(r"^(?:[-*•]|\d+[.)]|#{1,6}\s)", x)) for x in lines)
    meta = bool(re.search(rf"\b{META}\b", text.lower()))
    prompt_overlap = len(ps & out_set) / max(1, len(ps))
    return {
        "narrative_score": min(1.0, action_count / max(3, sentence_count * 2)) * float(not meta),
        "prompt_grounding_score": min(1.0, prompt_overlap * 2.0),
        "style_consistency_score": min(1.0, sentence_count / 8) * float(not meta),
        "meta_list_risk": min(1.0, (list_count / max(1, len(lines))) * 3 + float(meta) * 0.7),
        "repetition_risk": (len(grams) - len(set(grams))) / max(1, len(grams)),
        "fragment_risk": float(len(ws) < 30 or sum(len(x) < 3 for x in ws) / max(1, len(ws)) > 0.25),
        "non_ascii_rate": sum(ord(c) > 127 for c in text) / max(1, len(text)),
        "markup_rate": sum(c in "[]{}<>*#`_" for c in text) / max(1, len(text)),
    }


def label(row: pd.Series) -> str:
    if row.repetition_risk >= 0.05:
        return "repetitive_or_collapsed"
    if row.meta_list_risk >= 0.5:
        return "meta_or_list_drift"
    if row.non_ascii_rate >= 0.02 or row.markup_rate >= 0.02 or row.fragment_risk >= 1:
        return "token_or_format_breakdown"
    if row.narrative_score >= 0.35 and row.style_consistency_score >= 0.4:
        return "narrative_continuation"
    return "weak_or_mixed_continuation"


def main() -> None:
    records = []
    for model, directory in TARGETS.items():
        for path in sorted((ROOT / directory).glob("*.jsonl")):
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip(): continue
                    row = json.loads(line)
                    if row.get("method") not in SHAPES: continue
                    text = str(row.get("continuation_text", ""))
                    record = {"model": model, "shape": row["method"], "prompt_id": row.get("prompt_id"), "text": text, **score(text, str(row.get("prompt", "")))}
                    records.append(record)
    data = pd.DataFrame(records)
    data["natural_language_label"] = data.apply(label, axis=1)
    data["snippet"] = data["text"].str.replace(r"\s+", " ", regex=True).str.slice(0, 280)
    data.drop(columns=["text"]).to_csv(ROOT / "results/analysis/morphology_natural_language_rows.csv", index=False)
    summary = data.groupby(["shape", "model"], as_index=False).agg(prompt_count=("prompt_id", "nunique"), **{c: (c, "mean") for c in ["narrative_score", "prompt_grounding_score", "style_consistency_score", "meta_list_risk", "repetition_risk", "fragment_risk", "non_ascii_rate", "markup_rate"]})
    labels = data.groupby(["shape", "model", "natural_language_label"], as_index=False).size().rename(columns={"size": "count"})
    labels.to_csv(ROOT / "results/analysis/morphology_natural_language_labels.csv", index=False)
    summary.to_csv(ROOT / "results/analysis/morphology_natural_language_summary.csv", index=False)
    lines = ["# 18形状の自然言語出力評価", "", "保存済み3モデル×18形状×20 prompt（計1080 continuation）を共通の語彙・文体ルーブリックで評価した。これは人間の意味理解評価ではなく、再現可能な表層NLP診断である。", "", "## 評価軸", "", "- narrative continuation: 動作・出来事・文の連鎖があるか", "- prompt grounding: prompt語彙の接地があるか", "- style consistency: 文章としての文長・文の連続性", "- meta/list drift: タイトル・要約・箇条書き・説明への逸脱", "- repetition / fragment / format breakdown: 反復・断片・記号崩れ", "", "## 形状別の自然言語解釈", ""]
    for shape, group in summary.groupby("shape", sort=True):
        best = group.loc[group.narrative_score.idxmax()]
        risk = group.loc[group.meta_list_risk.add(group.repetition_risk).idxmax()]
        lines.append(f"### `{shape}`")
        lines.append("")
        lines.append(f"- narrative寄り: **{best.model}** (score={best.narrative_score:.2f})")
        lines.append(f"- 逸脱・反復寄り: **{risk.model}** (meta/list={risk.meta_list_risk:.2f}, repeat={risk.repetition_risk:.3f})")
        lines.append(f"- 解釈: {_interpret(shape)}")
        examples = data[data["shape"] == shape].sort_values("narrative_score", ascending=False).head(1)
        counter = data[data["shape"] == shape].sort_values(["meta_list_risk", "repetition_risk"], ascending=False).head(1)
        for kind, frame in [("代表的な継続", examples), ("反例・逸脱例", counter)]:
            if not frame.empty:
                r = frame.iloc[0]
                lines.append(f"- {kind} ({r.model}/{r.prompt_id}): {r.snippet}")
        lines.append("")
    lines += ["## 結論", "", "形状は自然言語の意味を直接表さないが、文章が一つの物語モードに留まるか、反復・メタ説明・形式崩れへ逸脱するかという生成状態には関係する。形状の自然言語的意味はモデル依存であり、18形状名を普遍的な品質ラベルとして扱うことはできない。", "", "## 限界", "", "- prompt接地は語彙重複を使った近似で、意味的な設定維持ではない。", "- narrative判定は動詞・文構造のヒューリスティックで、文脈理解ではない。", "- 代表例は自動抽出であり、人手評価ではない。最終的な意味品質の主張には人手またはLLM評価者との一致検証が必要である。"]
    (ROOT / "results/analysis/morphology_natural_language_evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"evaluated {len(data)} continuations")


def _interpret(shape: str) -> str:
    return {
        "low_spread": "候補が狭く、自然な継続にも反復・形式崩れにも固定される。モデル依存性が強い。",
        "mid_spread": "極端な逸脱を避けやすい中間状態だが、明確な意味品質の保証ではない。",
        "high_spread": "反復を抑えやすく、語彙の探索を許すが、物語の焦点が緩む可能性がある。",
        "low_concentration": "単一候補への固定が弱く、文体が安定する場合と弱くなる場合がある。",
        "mid_concentration": "比較的妥当な継続と探索性の妥協点に近い。",
        "high_concentration": "流れを強く固定する。正しい流れなら流暢だが、誤ると反復・縮退する。",
        "thin_tail": "候補の裾が薄く、既知の強いパターンへ戻りやすい。モデルによって反復が強くなる。",
        "mid_tail": "探索と安定性の中間で、極端な形式逸脱は比較的少ない。",
        "heavy_tail": "反復を抑えやすいが、局所的な意味接地が弱くなり得る。",
        "small_gap": "候補競合が強く、表現の揺れやモード混在が起きやすい。",
        "mid_gap": "候補間の選択が極端でなく、比較的安定した遷移を作る。",
        "large_gap": "一つの流れへのコミットが強く、誤った流れから戻りにくい。",
        "rank_slope_flat": "順位間の確率差が緩く、語彙選択は柔軟だが意味の焦点は保証されない。",
        "rank_slope_mid": "柔軟性と決定性の中間で、モデルによって自然な継続に寄る。",
        "rank_slope_steep": "上位候補への決定性が高く、出力が短絡的に固定される場合がある。",
        "spiky_top": "突出候補に強く依存し、流暢さと反復リスクが同居する。",
        "smooth_top": "複数候補を残し、反復を避けやすいが、モードが弱くなる可能性がある。",
        "multi_peak": "異なる文体・形式候補が競合し、自然文とメタ／記号出力が混ざりやすい。",
    }.get(shape, "形状の挙動はモデル依存である。")


if __name__ == "__main__":
    main()
