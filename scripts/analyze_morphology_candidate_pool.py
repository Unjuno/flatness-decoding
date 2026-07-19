"""Analyze all candidate-pool rows, avoiding selected-sample-only bias."""

from pathlib import Path

import pandas as pd


ROOT = Path("results/analysis")
PROBES = {"gpt2": "morphology_probe_gpt2", "gemma_raw": "morphology_probe_gemma_raw", "qwen_raw": "morphology_probe_qwen_raw"}
FEATURES = [
    "successor_logit_std", "successor_top1_prob", "successor_tail_mass_after_top50", "successor_gap_1_2",
    "successor_markup_mass_top20", "successor_subword_fragment_mass_top20", "successor_prompt_overlap_mass_top20",
    "successor_recent_overlap_mass_top20", "candidate_current_rank", "candidate_current_prob",
]


def main() -> None:
    frames = []
    for model, directory in PROBES.items():
        path = ROOT / directory / "morphology_stratified_candidate_morphology.csv"
        frame = pd.read_csv(path)
        frame["model"] = model
        frame["assigned_regimes"] = frame["assigned_regimes"].fillna("")
        frame["assigned_regime"] = frame["assigned_regimes"].str.split(";")
        frame = frame.explode("assigned_regime")
        frame[["regime_family", "regime"]] = frame["assigned_regime"].str.split(":", n=1, expand=True)
        frames.append(frame)
    all_rows = pd.concat(frames, ignore_index=True)
    all_rows = all_rows[all_rows["regime"].notna()].copy()
    summary = all_rows.groupby(["model", "regime_family", "regime"], as_index=False).agg(
        candidate_rows=("regime", "size"),
        prompts=("prompt_id", "nunique"),
        **{f"mean_{c}": (c, "mean") for c in FEATURES},
    )
    summary.to_csv(ROOT / "morphology_candidate_pool_summary.csv", index=False)
    coverage = summary.pivot_table(index="regime", columns="model", values=["candidate_rows", "prompts"], fill_value=0)
    lines = [
        "# Candidate-pool 全体の形状解析",
        "",
        "選択済み20件だけでなく、各モデルの top-100 candidate pool 全体を `assigned_regimes` で展開して集計した。したがって、形状の発生頻度と選択結果を分離して読める。",
        "",
        "## カバレッジ",
        "",
        coverage.round(0).to_markdown(),
        "",
        "## 初見",
        "",
        "- `markup_heavy` の不足は、選択アルゴリズムだけでなく candidate pool 内で該当 prompt が少ないことが原因である。Qwen raw のtop-100 candidate poolは7 promptしかカバーしないため、top-100 条件のまま20 promptを独立に揃えることはできない（selected候補の件数11とは別の値）。",
        "- 旧step-0 probeでは `recent_text=prompt` を渡していたため、既存CSVの `prompt_overlap_heavy` と `recent_overlap_heavy` は独立な証拠として使えない。実装は修正済みだが、既存CSVは再計算が必要である。",
        "- selected sample の平均が極端に見える形状は、candidate pool 全体の平均と比較しないと、形状の性質と選択バイアスを混同する。",
        "",
        "## 根拠",
        "",
        "- `morphology_candidate_pool_summary.csv`",
        "- 各モデルの `morphology_stratified_candidate_morphology.csv`",
    ]
    (ROOT / "morphology_candidate_pool_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote candidate-pool analysis")


if __name__ == "__main__":
    main()
