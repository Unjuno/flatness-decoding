"""Link per-prompt morphology token logs to saved continuation behavior."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    output = pd.concat([pd.read_csv(p) for p in sorted((ROOT / "results/analysis").glob("shape_*_output_rows.csv"))], ignore_index=True)
    token_paths = sorted((ROOT / "results/analysis").glob("shape_*_token_rows.csv"))
    token_frames = []
    for path in token_paths:
        frame = pd.read_csv(path)
        frame["shape"] = path.name.removeprefix("shape_").removesuffix("_token_rows.csv")
        token_frames.append(frame)
    token = pd.concat(token_frames, ignore_index=True)
    key = ["shape", "model", "prompt_id"]
    linked = output.merge(token, on=key, how="inner", validate="one_to_one")
    linked.to_csv(ROOT / "results/analysis/morphology_shape_token_output_link.csv", index=False)
    xcols = ["mean_chosen_rank", "mean_chosen_prob", "mean_chosen_flatness", "mean_selection_gain", "fallback_rate"]
    ycols = ["repeated_4gram_rate", "markup_marker_rate", "non_ascii_rate", "prompt_word_reuse_rate", "output_words"]
    rows = []
    for (model, shape), group in linked.groupby(["model", "shape"]):
        for x in xcols:
            for y in ycols:
                if group[x].nunique() < 3 or group[y].nunique() < 3:
                    rho = float("nan")
                else:
                    rho = group[x].corr(group[y], method="spearman")
                rows.append({"model": model, "shape": shape, "x": x, "y": y, "spearman_rho": rho, "n": len(group)})
    corr = pd.DataFrame(rows)
    corr.to_csv(ROOT / "results/analysis/morphology_shape_token_output_correlations.csv", index=False)
    strongest = corr.dropna().assign(abs_rho=lambda d: d.spearman_rho.abs()).sort_values("abs_rho", ascending=False).head(30)
    lines = ["# ロジット形状と本文出力のprompt単位リンク解析", "", "同じ `shape × model × prompt_id` のtoken logとcontinuation表層分類を結合した。相関は記述的なSpearman相関であり、因果効果ではない。", "", f"- 結合行数: {len(linked)}", f"- 形状数: {linked['shape'].nunique()}", f"- モデル数: {linked['model'].nunique()}", "", "## 最も強い関連（絶対値上位）", "", strongest[["model", "shape", "x", "y", "spearman_rho", "n"]].to_markdown(index=False), "", "## 解釈", "", "- `mean_chosen_rank` と `mean_chosen_prob` は、形状そのものではなく選択候補が元分布のどこにあるかを表す。", "- 反復やmarkupとの相関が形状をまたいで同じ向きになる場合、形状名より連続的なtoken-level coordinateを使う方が分類に適する。", "- prompt単位n=20なので、絶対値の小さな相関は無視し、複数形状・複数モデルで再現する向きだけを仮説として残す。"]
    (ROOT / "results/analysis/morphology_shape_token_output_link.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"linked {len(linked)} rows")


if __name__ == "__main__":
    main()
