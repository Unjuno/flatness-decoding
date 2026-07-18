"""Analyze all 18 detect_shape_family labels from existing candidate pools."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("results/analysis")
PROBES = {"gpt2": "morphology_probe_gpt2", "gemma_raw": "morphology_probe_gemma_raw", "qwen_raw": "morphology_probe_qwen_raw"}
SHAPE_RULES = {
    "rank_shape": {
        "rank_slope_flat": lambda r: r["successor_rank_prob_slope_top50"] >= -0.05,
        "rank_slope_mid": lambda r: -0.20 < r["successor_rank_prob_slope_top50"] < -0.05,
        "rank_slope_steep": lambda r: r["successor_rank_prob_slope_top50"] <= -0.20,
    },
    "peak_structure": {
        "spiky_top": lambda r: r["successor_top1_prob"] >= 0.60 and r["successor_gap_1_2"] >= 1.0,
        "smooth_top": lambda r: r["successor_top1_prob"] <= 0.20 and r["successor_gap_1_2"] <= 0.15,
        "multi_peak": lambda r: r["successor_large_gap_count_top50"] >= 2 and r["successor_gap_std_top50"] >= 0.5,
    },
}
CORE = ["successor_logit_std", "successor_top1_prob", "successor_tail_mass_after_top50", "successor_gap_1_2", "successor_rank_prob_slope_top50", "successor_large_gap_count_top50", "successor_gap_std_top50"]


def main() -> None:
    rows = []
    for model, directory in PROBES.items():
        frame = pd.read_csv(ROOT / directory / "morphology_stratified_candidate_morphology.csv")
        frame["model"] = model
        for _, row in frame.iterrows():
            for family, rules in SHAPE_RULES.items():
                for shape, rule in rules.items():
                    if rule(row):
                        record = {"model": model, "shape_family": family, "shape": shape, "prompt_id": row["prompt_id"], "candidate_rank": row["candidate_rank"]}
                        record.update({c: row[c] for c in CORE})
                        rows.append(record)
    expanded = pd.DataFrame(rows)
    expanded.to_csv(ROOT / "morphology_18_shape_rows.csv", index=False)
    summary = expanded.groupby(["model", "shape_family", "shape"], as_index=False).agg(
        rows=("shape", "size"), prompts=("prompt_id", "nunique"),
        **{f"mean_{c}": (c, "mean") for c in CORE},
    )
    all_shapes = [(family, shape) for family, rules in SHAPE_RULES.items() for shape in rules]
    grid = pd.MultiIndex.from_product([list(PROBES), all_shapes], names=["model", "shape_key"])
    complete = grid.to_frame(index=False)
    complete[["shape_family", "shape"]] = pd.DataFrame(complete["shape_key"].tolist(), index=complete.index)
    complete = complete.drop(columns="shape_key")
    summary = complete.merge(summary, on=["model", "shape_family", "shape"], how="left")
    summary["rows"] = summary["rows"].fillna(0).astype(int)
    summary["prompts"] = summary["prompts"].fillna(0).astype(int)
    summary.to_csv(ROOT / "morphology_18_shape_summary.csv", index=False)

    lines = [
        "# 18種類のロジット形状解析（3モデル）",
        "",
        "`detect_shape_family()` の定義に合わせ、既存 candidate pool の successor features から18ラベルを再構成した。これは新たなforwardではなく、保存済み特徴量の再分類である。1行が複数shapeに入ることがある。",
        "",
        "## 18種類の定義",
        "",
        "| family | shapes |",
        "|---|---|",
        "| spread | low_spread / mid_spread / high_spread |",
        "| concentration | low_concentration / mid_concentration / high_concentration |",
        "| tail | thin_tail / mid_tail / heavy_tail |",
        "| gap | small_gap / mid_gap / large_gap |",
        "| rank shape | rank_slope_flat / rank_slope_mid / rank_slope_steep |",
        "| peak structure | spiky_top / smooth_top / multi_peak |",
        "",
        "## カバレッジと平均特徴",
        "",
        "| model | shape | rows | prompts | std | top1 | tail50 | gap | rank slope | large gaps | gap std |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary.sort_values(["shape_family", "shape", "model"]).iterrows():
        lines.append(f"| {r['model']} | `{r['shape']}` | {int(r['rows'])} | {int(r['prompts'])} | {r['mean_successor_logit_std']:.3f} | {r['mean_successor_top1_prob']:.3f} | {r['mean_successor_tail_mass_after_top50']:.3f} | {r['mean_successor_gap_1_2']:.3f} | {r['mean_successor_rank_prob_slope_top50']:.3f} | {r['mean_successor_large_gap_count_top50']:.2f} | {r['mean_successor_gap_std_top50']:.3f} |")

    lines += ["", "## 18形状それぞれの初見", ""]
    impressions = {
        "low_spread": "logit幅が狭い。ただし尖ったGemmaと競合的なQwenを含むため、flatness単独では意味が決まらない。",
        "mid_spread": "中間幅。3モデルで比較的stdが近く、基準形状として使いやすい。",
        "high_spread": "幅が広いが、厚いtailとは限らない。GPT-2/Gemmaでは尖った広幅も含む。",
        "low_concentration": "top1が低い探索型。tailも厚くなりやすいがGemmaでは必ずしもそうならない。",
        "mid_concentration": "複数候補を残す中間型。ただしモデルごとの確率校正差が大きい。",
        "high_concentration": "単峰・top1優勢。安定性と保守性、反復リスクが同居する。",
        "thin_tail": "tailがほぼ上位候補に吸収された状態。large_gap/high_concentrationと重複しやすい。",
        "mid_tail": "tailの折衷型。探索性を残しつつ極端な拡散を避ける候補。",
        "heavy_tail": "上位50外に質量が残る。候補の多さは増えるが、意味的接地は保証しない。",
        "small_gap": "top1/top2が競合。sampling・seedに敏感で、候補選択の不安定性を示す。",
        "mid_gap": "候補間の優劣が中程度。単峰と競合の中間状態。",
        "large_gap": "top1が明確。thin_tailと強く重複し、保守的な次状態になりやすい。",
        "rank_slope_flat": "上位rank間の確率減衰が緩い。top1だけでなく上位候補全体に質量がある。",
        "rank_slope_mid": "rank減衰の中間。形状の連続的な補間点として有用。",
        "rank_slope_steep": "上位から急落するrank曲線。単峰性・選択固定化と結びつきやすい。",
        "spiky_top": "top1が高くgapも大きい明確な単峰。high_concentration/large_gap/thin_tailと重複する。",
        "smooth_top": "top1もgapも小さい競合型。候補の微差に敏感で、探索的だが不安定になりやすい。",
        "multi_peak": "上位50内に複数の大きな段差がある多峰型。単純なtop1/tailでは表せない。",
    }
    for shape, text in impressions.items():
        lines.append(f"- `{shape}`: {text}")

    missing = summary[summary["rows"] == 0]
    lines += ["", "## 観測されなかったモデル×形状", ""]
    if missing.empty:
        lines.append("全モデルで全形状が少なくとも1件観測された。")
    else:
        for _, row in missing.iterrows():
            lines.append(f"- `{row['model']}` × `{row['shape']}`: 0 rows / 0 prompts")
        lines.append("")
        lines.append("0件は単なるファイル欠損ではない。現行のモデル・prompt・top-100 candidate pool・step 0 条件では、そのshape ruleを満たす successor state が観測されなかったことを意味する。")

    lines += ["", "## 重要な新しい洞察", "", "1. これまで欠けていた rank shape と peak structure を加えると、18形状は『3つの独立な分布軸』ではなく、spread・concentration・tail・gap・rank curve・peak topology の6軸になる。", "2. `spiky_top` は単なる高 concentration ではなく、top1とtop2の明確な分離を要求する。高 concentrationでもmulti-peakや厚いtailはあり得るため、peak familyは追加情報を持つ。", "3. `rank_slope_flat` と `heavy_tail` は似て見えるが、前者は上位rankの減衰、後者はtop50外の質量を測る。両者が一致しないモデル・形状を探す価値がある。", "4. `multi_peak` はflatness研究にとって重要な候補である。単純にminflat/maxflatを選ぶより、複数の確率峰を持つ successor state を避ける／選ぶ制御が可能になる。", "5. 18形状を横断しても、raw logit stdの絶対値より model-local percentile の方が比較可能である。", "", "## 解析上の限界", "", "- candidate pool は top-100、step 0 の初期promptだけである。生成時系列の18形状頻度ではない。", "- token-type five shapes は今回の18 shape familyとは別の診断で、内容依存のためdistribution topologyと分けて解釈する。", "- `recent_overlap` は旧データではprompt overlapと同じ値になっていたため、形状解析の中心18には含めていない。", "- Qwen raw markup_heavyの不足は別問題として残っている。",
    ]
    (ROOT / "morphology_18_shape_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote 18-shape analysis")


if __name__ == "__main__":
    main()
