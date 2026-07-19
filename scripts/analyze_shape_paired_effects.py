"""Compare each saved shape continuation against same-prompt greedy output."""
from __future__ import annotations

import json
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {"gpt2": "outputs/generations/comparison_gpt2_shapes_20", "gemma": "outputs/generations/comparison_gemma3_4b_shapes_20", "qwen": "outputs/generations/comparison_qwen25_1_5b_shapes_20"}
SHAPES = {"low_spread", "mid_spread", "high_spread", "low_concentration", "mid_concentration", "high_concentration", "thin_tail", "mid_tail", "heavy_tail", "small_gap", "mid_gap", "large_gap", "rank_slope_flat", "rank_slope_mid", "rank_slope_steep", "spiky_top", "smooth_top", "multi_peak"}


def metrics(text: str) -> dict[str, float]:
    words = re.findall(r"\S+", text)
    grams = [" ".join(w.lower() for w in words[i : i + 4]) for i in range(max(0, len(words) - 3))]
    return {"output_words": len(words), "repeat_rate": (len(grams) - len(set(grams))) / max(1, len(grams)), "non_ascii_rate": sum(ord(c) > 127 for c in text) / max(1, len(text)), "markup_rate": sum(c in "[]{}<>*#`_" for c in text) / max(1, len(text)), "newline_rate": text.count("\n") / max(1, len(text))}


def main() -> None:
    records = []
    for model, directory in TARGETS.items():
        rows = []
        for path in sorted((ROOT / directory).glob("*.jsonl")):
            with path.open(encoding="utf-8") as fh:
                rows.extend(json.loads(x) for x in fh if x.strip())
        keyed = {(r.get("prompt_id"), r.get("method")): r for r in rows}
        prompts = sorted({k[0] for k in keyed if k[1] == "greedy"})
        for shape in sorted(SHAPES):
            for prompt_id in prompts:
                a = keyed.get((prompt_id, shape)); b = keyed.get((prompt_id, "greedy"))
                if not a or not b: continue
                am, bm = metrics(str(a.get("continuation_text", ""))), metrics(str(b.get("continuation_text", "")))
                records.append({"model": model, "shape": shape, "prompt_id": prompt_id, **{f"delta_{k}": am[k] - bm[k] for k in am}})
    data = pd.DataFrame(records)
    data.to_csv(ROOT / "results/analysis/morphology_shape_paired_deltas.csv", index=False)
    metrics_names = ["delta_output_words", "delta_repeat_rate", "delta_non_ascii_rate", "delta_markup_rate", "delta_newline_rate"]
    summary = data.groupby(["shape", "model"], as_index=False).agg({m: ["mean", "median", lambda s: (s > 0).mean()] for m in metrics_names})
    summary.columns = ["_".join(x).strip("_") if isinstance(x, tuple) else x for x in summary.columns]
    summary.to_csv(ROOT / "results/analysis/morphology_shape_paired_summary.csv", index=False)
    lines = ["# 形状のprompt内paired効果", "", "同じ `model × prompt_id` の `greedy` continuationを基準に、各形状methodの保存済み出力との差分を計算した。これは形状の相対的な出力効果を調べる記述解析である。", "", "## 反復率の増加が大きい形状", ""]
    top = summary.sort_values("delta_repeat_rate_mean", ascending=False).head(15)
    lines.append(top[["shape", "model", "delta_repeat_rate_mean", "delta_repeat_rate_median", "delta_repeat_rate_<lambda_0>"]].to_markdown(index=False))
    lines += ["", "## 解釈", "", "- greedyとの差分なので、モデル全体の出力傾向より形状methodによる相対変化を見やすい。", "- `delta_repeat_rate_mean > 0` でも、同一promptで常に悪化するとは限らない。最後の列は悪化したpromptの割合である。", "- 形状ごとの効果がモデル間で同じ向きなら、モデル横断の制御候補になる。向きが逆なら、model-local calibrationが必要である。"]
    (ROOT / "results/analysis/morphology_shape_paired_effects.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"paired rows={len(data)}")


if __name__ == "__main__":
    main()
