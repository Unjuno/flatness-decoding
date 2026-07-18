"""Aggregate the per-shape saved-output analyses into a cross-shape table."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    paths = sorted((ROOT / "results/analysis").glob("shape_*_output_rows.csv"))
    frames = [pd.read_csv(p) for p in paths]
    if not frames:
        raise SystemExit("no per-shape output rows found")
    data = pd.concat(frames, ignore_index=True)
    metrics = ["repeated_4gram_rate", "non_ascii_rate", "markup_marker_rate", "prompt_word_reuse_rate", "output_words"]
    summary = data.groupby(["shape", "model"], as_index=False).agg(
        prompt_count=("prompt_id", "nunique"),
        **{f"mean_{m}": (m, "mean") for m in metrics},
    )
    summary.to_csv(ROOT / "results/analysis/morphology_shape_output_summary.csv", index=False)
    lines = [
        "# 形状別出力ログの横断解析",
        "",
        "保存済み `shapes_20` の continuation と token-log突合結果を形状×モデルで再集計した。各セルは20 promptを目標とし、再生成は行っていない。",
        "",
        "## 読み方",
        "",
        "- `mean_repeated_4gram_rate`: 反復・縮退の粗い指標",
        "- `mean_non_ascii_rate`: tokenizer／形式崩れの粗い指標（多言語文字を含むため単独で品質判定しない）",
        "- `mean_markup_marker_rate`: マークアップ・記号混入の粗い指標",
        "- `mean_prompt_word_reuse_rate`: プロンプト語彙の再利用率。接地と単純反復を分離していない",
        "",
        "## 形状ごとの暫定分類",
        "",
    ]
    for shape, group in summary.groupby("shape", sort=True):
        top_repeat = group.loc[group["mean_repeated_4gram_rate"].idxmax()]
        top_nonascii = group.loc[group["mean_non_ascii_rate"].idxmax()]
        top_reuse = group.loc[group["mean_prompt_word_reuse_rate"].idxmax()]
        lines.append(
            f"- **{shape}**: 反復最大={top_repeat['model']} ({top_repeat['mean_repeated_4gram_rate']:.3f}); "
            f"非ASCII最大={top_nonascii['model']} ({top_nonascii['mean_non_ascii_rate']:.3f}); "
            f"prompt語彙再利用最大={top_reuse['model']} ({top_reuse['mean_prompt_word_reuse_rate']:.3f})"
        )
    lines += [
        "",
        "## 現時点の洞察",
        "",
        "1. 出力の差は形状単独よりもモデルとの交互作用が強い。特に `low_spread`、`thin_tail`、`high_concentration`、`spiky_top` で反復または非ASCII混入がモデル間で大きく分かれる。",
        "2. `large_gap` は3モデルで反復率が比較的高く、上位候補の断絶が続きやすいという仮説を支持する。ただし `low_spread` とは異なり、モデル固有の形式崩れだけでは説明しにくい。",
        "3. `rank_slope_flat` や `smooth_top` は平均的には比較的安定だが、Qwenの prompt語彙再利用や非ASCII混入のように、形状の安定性と内容の健全性は別軸である。",
        "4. この表は表層指標であり、意味的な物語品質・遵守性の決定的分類ではない。次の判定では出力本文の手動サンプル監査または意味カテゴリ分類を追加する。",
    ]
    (ROOT / "results/analysis/morphology_shape_output_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote morphology_shape_output_summary.csv and .md")


if __name__ == "__main__":
    main()
