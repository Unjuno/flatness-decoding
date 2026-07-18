"""Analyze one shape method against its saved continuation and token logs."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {
    "gpt2": ("outputs/generations/comparison_gpt2_shapes_20", "outputs/token_logs/comparison_gpt2_shapes_20"),
    "gemma": ("outputs/generations/comparison_gemma3_4b_shapes_20", "outputs/token_logs/comparison_gemma3_4b_shapes_20"),
    "qwen": ("outputs/generations/comparison_qwen25_1_5b_shapes_20", "outputs/token_logs/comparison_qwen25_1_5b_shapes_20"),
}


def read_jsonl(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())
    return rows


def ngrams(text: str, n: int) -> list[str]:
    words = re.findall(r"\S+", text.lower())
    return [" ".join(words[i : i + n]) for i in range(max(0, len(words) - n + 1))]


def output_metrics(text: str, prompt: str) -> dict[str, float]:
    words = re.findall(r"\S+", text)
    grams = ngrams(text, 4)
    repeated = len(grams) - len(set(grams))
    prompt_words = set(re.findall(r"[a-zA-Z]{3,}", prompt.lower()))
    out_words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return {
        "output_chars": len(text),
        "output_words": len(words),
        "non_ascii_rate": sum(ord(c) > 127 for c in text) / max(1, len(text)),
        "whitespace_rate": sum(c.isspace() for c in text) / max(1, len(text)),
        "punctuation_rate": sum(not c.isalnum() and not c.isspace() for c in text) / max(1, len(text)),
        "markup_marker_rate": sum(c in "[]{}<>*#`_" for c in text) / max(1, len(text)),
        "repeated_4gram_rate": repeated / max(1, len(grams)),
        "prompt_word_reuse_rate": sum(w in prompt_words for w in out_words) / max(1, len(out_words)),
        "question_rate": text.count("?") / max(1, len(text)),
        "newline_rate": text.count("\n") / max(1, len(text)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="low_spread")
    args = ap.parse_args()
    rows = []
    token_rows = []
    for model, (gen_rel, tok_rel) in TARGETS.items():
        gens = read_jsonl(sorted((ROOT / gen_rel).glob("*.jsonl")))
        toks = read_jsonl(sorted((ROOT / tok_rel).glob("*.jsonl")))
        gens = [r for r in gens if r.get("method") == args.shape]
        toks = [r for r in toks if r.get("method") == args.shape]
        tok = pd.DataFrame(toks)
        if not tok.empty:
            agg = tok.groupby("prompt_id", as_index=False).agg(
                token_steps=("step", "count"),
                mean_chosen_rank=("chosen_current_rank", "mean"),
                mean_chosen_prob=("chosen_current_prob", "mean"),
                mean_chosen_flatness=("chosen_flatness", "mean"),
                mean_selection_gain=("selection_gain", "mean"),
                fallback_rate=("selection_fallback", "mean"),
            )
        else:
            agg = pd.DataFrame(columns=["prompt_id"])
        for row in gens:
            metrics = output_metrics(str(row.get("continuation_text", "")), str(row.get("prompt", "")))
            record = {"model": model, "shape": args.shape, "prompt_id": row.get("prompt_id"), "method": row.get("method"), **metrics}
            rows.append(record)
        if not agg.empty:
            agg["model"] = model
            token_rows.append(agg)
    out = pd.DataFrame(rows)
    tokout = pd.concat(token_rows, ignore_index=True) if token_rows else pd.DataFrame()
    out_path = ROOT / "results/analysis" / f"shape_{args.shape}_output_rows.csv"
    out.to_csv(out_path, index=False)
    if not tokout.empty:
        tokout.to_csv(ROOT / "results/analysis" / f"shape_{args.shape}_token_rows.csv", index=False)
    lines = [f"# `{args.shape}` 個別出力ログ解析", "", "既存の `shapes_20` 生成ログと token log を、モデル・prompt_id で突合した。再生成は行っていない。", ""]
    lines += [f"- 出力行数: {len(out)}", f"- モデル数: {out['model'].nunique() if not out.empty else 0}", ""]
    if out.empty:
        lines += ["対象ログが見つからない。"]
    else:
        summary = out.groupby("model").agg({c: "mean" for c in out.columns if c not in {"model", "shape", "prompt_id", "method"}}).reset_index()
        lines += ["## モデル別平均", "", summary.to_markdown(index=False), "", "## 暫定分類", ""]
        for _, r in summary.iterrows():
            if r["repeated_4gram_rate"] > 0.05:
                label = "反復・縮退寄り"
            elif r["markup_marker_rate"] > 0.02 or r["non_ascii_rate"] > 0.05:
                label = "形式／トークン崩れ寄り"
            elif r["prompt_word_reuse_rate"] > 0.35:
                label = "プロンプト語彙接地寄り"
            else:
                label = "通常の続き生成寄り"
            lines.append(f"- **{r['model']}**: {label}")
        lines += ["", "この分類は出力ログの記述統計に基づく暫定分類であり、品質の因果判定ではない。"]
    (ROOT / "results/analysis" / f"shape_{args.shape}_output_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
