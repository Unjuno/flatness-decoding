"""Summarize morphology regimes across the completed three-model probes."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("results/analysis")
PROBES = {"GPT-2": "morphology_probe_gpt2", "Gemma raw": "morphology_probe_gemma_raw", "Qwen raw": "morphology_probe_qwen_raw"}
DESCRIPTIONS = {
    "low_spread": "狭い logits 幅。ただし top1 が高いモデルでは、平坦というより単一候補への集中として現れる。",
    "mid_spread": "中程度の幅。top1・tail・gap が中間になり、最も通常的な遷移候補になりやすい。",
    "high_spread": "広い logits 幅。ただし tail が増えるとは限らず、尖った多峰性も含むため spread 単独では不十分。",
    "low_concentration": "top1 が低く tail が厚い拡散型。候補の選択余地は大きいが、接地性は弱くなりやすい。",
    "mid_concentration": "集中と拡散の中間。複数候補を残しつつ、完全な diffuse 状態を避ける。",
    "high_concentration": "top1 と gap が大きく tail が薄い単峰型。安定するが、反復・保守性のリスクがある。",
    "thin_tail": "上位候補に確率が集まり tail が薄い。large-gap/high-concentration とほぼ同じ状態を別名で検出している。",
    "mid_tail": "上位と tail の折衷。top1 よりも tail の厚みを独立に調整できる領域。",
    "heavy_tail": "上位50以外にも質量が残る拡散型。探索性は高いが、候補の意味的接地が弱い可能性がある。",
    "small_gap": "top1 と top2 の差が小さい競合状態。単一の明確な次語がなく、sampling 感度が高い。",
    "mid_gap": "候補間の優劣が中程度。top1 依存と競合性の中間に位置する。",
    "large_gap": "top1 が明確に優勢。thin-tail/high-concentration と重なり、安定性と保守性を同時に持つ。",
    "markup_heavy": "markup・記号系 token が上位に出やすい状態。モデルの tokenizer と prompt framing の影響が強い。",
    "list_heavy": "箇条書き・列挙 token が上位に出やすい状態。物語継続より構造化出力へ遷移しやすい。",
    "fragment_heavy": "短い token fragment が上位に集中する状態。tokenizer 依存の collapse／不自然な継続の候補。",
    "prompt_overlap_heavy": "prompt の語彙を再利用する状態。接地性にも prompt leakage にもなり得る。",
    "recent_overlap_heavy": "直近出力を再利用する状態。局所反復・ループの予兆になり得る。",
}


def load_probe(directory: str, label: str) -> pd.DataFrame:
    path = ROOT / directory / "morphology_stratified_probe_summary.csv"
    frame = pd.read_csv(path)
    frame["model"] = label
    return frame


def fmt(value: float) -> str:
    return f"{value:.3f}"


def main() -> None:
    summary = pd.concat([load_probe(path, label) for label, path in PROBES.items()], ignore_index=True)
    pivot = summary.pivot(index="regime", columns="model", values=["mean_successor_logit_std", "mean_successor_top1_prob", "mean_successor_tail_mass_after_top50", "mean_successor_gap_1_2"])
    lines = [
        "# 形状レジーム別の特徴解析（3モデル）",
        "",
        "対象は GPT-2、Gemma raw、Qwen raw の各17レジーム×20サンプル。指標は選択された successor distribution の平均であり、品質指標ではない。Qwen raw の markup_heavy は11/20で、それ以外は20/20。",
        "",
        "## 形状別の比較表",
        "",
        "| regime | GPT-2 (std/top1/tail/gap) | Gemma raw | Qwen raw | 共通解釈 |",
        "|---|---|---|---|---|",
    ]
    regimes = list(summary["regime"].drop_duplicates())
    for regime in regimes:
        cells = []
        for model in PROBES:
            row = summary[(summary.model == model) & (summary.regime == regime)].iloc[0]
            cells.append("/".join(fmt(row[c]) for c in ["mean_successor_logit_std", "mean_successor_top1_prob", "mean_successor_tail_mass_after_top50", "mean_successor_gap_1_2"]))
        lines.append(f"| `{regime}` | {cells[0]} | {cells[1]} | {cells[2]} | {DESCRIPTIONS[regime]} |")

    lines += ["", "## 17形状を一つずつ読む", ""]
    for i, regime in enumerate(regimes, 1):
        rows = summary[summary.regime == regime]
        means = rows[["mean_successor_logit_std", "mean_successor_top1_prob", "mean_successor_tail_mass_after_top50", "mean_successor_gap_1_2"]].mean()
        ranges = rows[["mean_successor_logit_std", "mean_successor_top1_prob", "mean_successor_tail_mass_after_top50", "mean_successor_gap_1_2"]].max() - rows[["mean_successor_logit_std", "mean_successor_top1_prob", "mean_successor_tail_mass_after_top50", "mean_successor_gap_1_2"]].min()
        lines.append(f"### {i}. `{regime}`")
        lines.append("")
        lines.append(DESCRIPTIONS[regime])
        lines.append(f"3モデル平均は std={means.iloc[0]:.3f}, top1={means.iloc[1]:.3f}, tail50={means.iloc[2]:.3f}, gap={means.iloc[3]:.3f}。モデル間レンジは std={ranges.iloc[0]:.3f}, top1={ranges.iloc[1]:.3f}, tail50={ranges.iloc[2]:.3f}, gap={ranges.iloc[3]:.3f}。")
        if ranges.iloc[1] > 0.5:
            lines.append("モデル差が大きく、形状名よりもモデル固有の logit スケール／tokenizer の影響が強い。")
        elif ranges.iloc[2] > 0.25:
            lines.append("top1 より tail のモデル差が目立つため、同じ形状でも探索性の意味は一致しない。")
        else:
            lines.append("3モデルで比較的同じ方向の形状として観測される。")
        lines.append("")

    lines += [
        "## 重要な発見",
        "",
        "1. `low_spread` はモデル共通の意味ではない。GPT-2 では top1=.644、Gemma raw では .957、Qwen raw では .395で、同じ spread 条件が尖った分布から競合分布まで変わる。",
        "2. `high_spread` も単純な diffuse ではない。GPT-2 は top1=.525、Gemma raw は .623、Qwen raw は .172で、spread と concentration は別軸として保存すべきである。",
        "3. `high_concentration`、`thin_tail`、`large_gap` は多くの条件で重複した状態を別の検出器が拾っている。18種類を独立な18形状と解釈するより、基礎軸（spread/concentration/tail/gap/token type）の組合せと見るべきである。",
        "4. `prompt_overlap_heavy` と `recent_overlap_heavy` は、GPT-2/Qwen raw ではほぼ同じ統計値であり、実質的に同じ検出状態に落ちている可能性がある。形状として分離するには token 内容と時系列の追加分析が必要である。",
        "5. Gemma raw は top1 と gap が大きく tail が薄い形状が多く、Qwen raw は tail が厚く top1 が低い形状が多い。したがって、同じ selector quantile をモデル間で共有するのは危険である。",
        "",
        "## 解釈上の注意",
        "",
        "- これは各レジームを意図的に選んだ probe であり、自然発生頻度の推定ではない。",
        "- `mean_successor_logit_std` は raw logit の標準偏差で、Shannon entropy ではない。",
        "- regime は閾値ベースの検出器で重複し得る。特に concentration/tail/gap のレジームは独立条件ではない。",
        "- 「18種類」は現行 CSV では17種類。欠落した1種類があるなら、元の run 設定またはラベル生成部を確認する必要がある。",
    ]
    output = ROOT / "morphology_regime_feature_analysis.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
