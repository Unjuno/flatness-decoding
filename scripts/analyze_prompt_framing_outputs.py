"""Compare saved raw and wrapped prompt outputs for the shared 20 prompts."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {
    "gemma": ("outputs/generations/comparison_gemma3_4b_shapes_20", "outputs/generations/comparison_gemma3_4b_shapes"),
    "qwen": ("outputs/generations/comparison_qwen25_1_5b_shapes_20", "outputs/generations/comparison_qwen25_1_5b_shapes"),
}


def rows(directory: str) -> list[dict]:
    out = []
    for p in sorted((ROOT / directory).glob("*.jsonl")):
        with p.open(encoding="utf-8") as fh:
            out.extend(json.loads(x) for x in fh if x.strip())
    return out


def metrics(text: str) -> dict[str, float]:
    words = re.findall(r"\S+", text)
    normalized = [w.lower() for w in words]
    grams = [" ".join(normalized[i : i + 4]) for i in range(max(0, len(normalized) - 3))]
    return {
        "output_words": len(words),
        "non_ascii_rate": sum(ord(c) > 127 for c in text) / max(1, len(text)),
        "markup_marker_rate": sum(c in "[]{}<>*#`_" for c in text) / max(1, len(text)),
        "repeated_4gram_rate": (len(grams) - len(set(grams))) / max(1, len(grams)),
        "newline_rate": text.count("\n") / max(1, len(text)),
    }


def main() -> None:
    records = []
    for model, (raw_dir, wrapped_dir) in TARGETS.items():
        raw = {(r.get("prompt_id"), r.get("method")): r for r in rows(raw_dir)}
        wrapped = {(r.get("prompt_id"), r.get("method")): r for r in rows(wrapped_dir)}
        for key in sorted(set(raw) & set(wrapped)):
            prompt_id, method = key
            for framing, source in [("raw", raw[key]), ("wrapped", wrapped[key])]:
                records.append({"model": model, "prompt_id": prompt_id, "method": method, "framing": framing, **metrics(str(source.get("continuation_text", "")))})
    data = pd.DataFrame(records)
    data.to_csv(ROOT / "results/analysis/prompt_framing_output_rows.csv", index=False)
    summary = data.groupby(["model", "method", "framing"], as_index=False).agg({c: "mean" for c in ["output_words", "non_ascii_rate", "markup_marker_rate", "repeated_4gram_rate", "newline_rate"]})
    summary.to_csv(ROOT / "results/analysis/prompt_framing_output_summary.csv", index=False)
    pivot = summary.pivot_table(index=["model", "method"], columns="framing", values=["output_words", "non_ascii_rate", "markup_marker_rate", "repeated_4gram_rate"], aggfunc="first")
    lines = ["# raw／wrapped 出力ログ比較", "", "同じ `prompt_id` の保存済み出力を、raw（`shapes_20`）と wrapped（`shapes`）で突合した。raw／wrappedでprompt本文が異なるため、これはプロンプト形式の効果を含む記述比較であり、形状の因果検定ではない。", "", f"- 対象モデル: {data['model'].nunique()}", f"- 共通prompt数: {data['prompt_id'].nunique()}", f"- 共通method数: {data['method'].nunique()}", "", "## 形状methodの解釈上の注意", "", "`shapes_20` の形状methodはraw側にだけ存在する。したがって、raw／wrapped差を形状ごとに直接比較するには、wrapped側にも同じ形状methodを保存したログが必要である。今回の比較は `greedy` と `beam` の共有methodに限定した。", "", "## 観測された差", ""]
    for (model, method), g in summary.groupby(["model", "method"]):
        if set(g["framing"]) != {"raw", "wrapped"}:
            continue
        a = g.set_index("framing")
        lines.append(f"- **{model} / {method}**: raw→wrapped の平均語数 {a.loc['raw','output_words']:.1f}→{a.loc['wrapped','output_words']:.1f}; 反復率 {a.loc['raw','repeated_4gram_rate']:.3f}→{a.loc['wrapped','repeated_4gram_rate']:.3f}; マークアップ率 {a.loc['raw','markup_marker_rate']:.3f}→{a.loc['wrapped','markup_marker_rate']:.3f}")
    lines += ["", "## 結論", "", "1. wrapped条件は出力の長さ・改行・形式を変え得るため、rawの形状ログからwrappedの生成挙動を推定してはいけない。", "2. 現存ログだけでは、18形状それぞれのraw／wrapped対応を完全には検定できない。形状methodをwrapped条件でも保存したtoken logが必要である。", "3. それでも共通methodの比較は、モデル固有のprompt framing感度を分類する基準として利用できる。"]
    (ROOT / "results/analysis/prompt_framing_output_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote prompt framing output analysis")


if __name__ == "__main__":
    main()
