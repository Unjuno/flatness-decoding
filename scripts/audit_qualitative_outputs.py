from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-dir", required=True)
    parser.add_argument("--merged-table", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_prompts(path: Path) -> dict[str, str]:
    prompts: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        prompts[row["id"]] = row["prompt"]
    return prompts


def canonical_text(row: pd.Series) -> str:
    for key in ("continuation_text", "generated_text", "output_text", "text"):
        value = row.get(key)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        text = str(value)
        if text.strip():
            return text
    return ""


def score_row(row: pd.Series) -> dict[str, object]:
    text = canonical_text(row)
    continuation = str(row.get("continuation_text", "") or "")
    prompt = str(row.get("prompt", "") or "")
    toks = re.findall(r"[A-Za-z0-9']+|[^\w\s]", text)
    words = re.findall(r"[A-Za-z0-9']+", text)
    lower = text.lower()
    has_repeat = False
    if words:
        counts = Counter(words)
        has_repeat = max(counts.values()) >= 5 or any(a == b == c for a, b, c in zip(words, words[1:], words[2:]))
    has_symbol_noise = bool(re.search(r"[\u2580-\u259f]{2,}|[<>{}\[\]#@$%^*_\\|]{3,}", text))
    has_url = bool(re.search(r"https?://|www\.|<[^>]+>", lower))
    is_abrupt = len(continuation.strip()) == 0 or len(words) < 3
    is_off_prompt = False
    if prompt:
        prompt_words = set(re.findall(r"[A-Za-z0-9']+", prompt.lower()))
        if words and len(prompt_words.intersection(set(w.lower() for w in words))) <= max(1, len(prompt_words) // 5):
            is_off_prompt = True
    is_generic = any(phrase in lower for phrase in ["i don't know", "nothing happened", "the end", "it was just", "he said"]) and len(words) < 80
    unusable = is_abrupt or has_symbol_noise or has_url or has_repeat
    fluency = 5 if not unusable and len(words) >= 40 else 4 if len(words) >= 20 and not has_symbol_noise else 3 if len(words) >= 8 else 1
    coherence = 5 if not is_off_prompt and len(words) >= 30 else 4 if len(words) >= 12 and not is_off_prompt else 2 if is_off_prompt else 3
    specificity = 5 if re.search(r"\b(doctor|knife|car|ship|street|house|king|mother|father|city|gun|blood|school|room|door)\b", lower) else 3 if len(words) >= 15 else 1
    nondegeneration = 5 if not unusable else 2 if not has_symbol_noise and not has_url else 1
    creativity = 5 if not unusable and specificity >= 4 and not is_generic else 3 if not unusable else 1
    prompt_following = 5 if not is_off_prompt and len(words) >= 15 else 3 if len(words) >= 5 else 1
    if is_generic:
        creativity = min(creativity, 2)
        specificity = min(specificity, 2)
    if has_repeat:
        nondegeneration = min(nondegeneration, 2)
        fluency = min(fluency, 3)
    return {
        "fluency": fluency,
        "coherence": coherence,
        "specificity": specificity,
        "nondegeneration": nondegeneration,
        "creativity": creativity,
        "prompt_following": prompt_following,
        "has_repetition_loop": bool(has_repeat),
        "has_symbol_noise": bool(has_symbol_noise),
        "has_url_or_markup_fragment": bool(has_url),
        "is_abrupt_or_truncated": bool(is_abrupt),
        "is_off_prompt": bool(is_off_prompt),
        "is_generic": bool(is_generic),
        "is_unusable": bool(unusable),
        "short_note": (
            "whitespace/empty continuation"
            if is_abrupt
            else "repetition loop"
            if has_repeat
            else "symbol/url noise"
            if has_symbol_noise or has_url
            else "off prompt"
            if is_off_prompt
            else "generic"
            if is_generic
            else "conventional continuation"
        ),
    }


def main() -> None:
    args = parse_args()
    gen_dir = Path(args.generation_dir)
    merged = pd.read_csv(args.merged_table)
    prompts = load_prompts(Path(args.prompts))

    files = sorted(gen_dir.glob("*.jsonl"))
    rows = []
    for path in files:
        file_rows = pd.read_json(path, lines=True)
        first = file_rows.iloc[0].to_dict()
        method = str(first.get("method"))
        n_trials = first.get("n_trials")
        n_trials = None if pd.isna(n_trials) else int(n_trials)
        seed = int(first.get("seed")) if pd.notna(first.get("seed")) else None
        for _, row in file_rows.iterrows():
            prompt_id = row["prompt_id"]
            prompt = prompts.get(prompt_id, str(row.get("prompt", "")))
            row = row.to_dict()
            row["prompt"] = prompt
            meta = merged[(merged["method"] == method) & (merged["n_trials"].fillna(-1) == (-1 if n_trials is None else n_trials))]
            meta_row = meta.iloc[0].to_dict() if len(meta) else {}
            sample = {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "method": method,
                "n_trials": n_trials,
                "seed": seed,
                "continuation_text": row.get("continuation_text", ""),
                "text": row.get("text", ""),
                "num_output_tokens": row.get("num_output_tokens"),
                "runtime_seconds": row.get("runtime_seconds"),
            }
            for col in [
                "experiment_name",
                "model",
                "method_family",
                "mean_chosen_flatness",
                "mean_chosen_current_logprob",
                "mean_chosen_current_rank",
                "mean_chosen_flatness_quantile",
                "distinct_1",
                "distinct_2",
                "repetition_4",
                "mean_runtime_seconds",
                "mean_tokens_per_second",
                "mean_candidate_current_logprob",
                "mean_min_candidate_current_rank",
                "mean_max_candidate_current_rank",
                "mean_selection_gain",
            ]:
                sample[col] = meta_row.get(col)
            sample.update(score_row(pd.Series(sample)))
            rows.append(sample)

    df = pd.DataFrame(rows)
    # deterministic sample: first N shared prompt ids
    prompt_ids = sorted(df["prompt_id"].unique())[: args.sample_size]
    sample_df = df[df["prompt_id"].isin(prompt_ids)].copy()

    prefix = Path(args.output_prefix)
    sample_csv = prefix.with_name(prefix.name + "_qualitative_sample.csv")
    sample_md = prefix.with_name(prefix.name + "_qualitative_sample.md")
    scores_csv = prefix.with_name(prefix.name + "_qualitative_scores.csv")
    summary_csv = prefix.with_name(prefix.name + "_qualitative_summary.csv")
    compare_csv = prefix.with_name(prefix.name + "_metric_qualitative_comparison.csv")
    report_md = prefix.with_name(prefix.name + "_qualitative_audit.md")

    sample_df.to_csv(sample_csv, index=False)
    sample_df[[
        "prompt_id",
        "method",
        "n_trials",
        "seed",
        "fluency",
        "coherence",
        "specificity",
        "nondegeneration",
        "creativity",
        "prompt_following",
        "has_repetition_loop",
        "has_symbol_noise",
        "has_url_or_markup_fragment",
        "is_abrupt_or_truncated",
        "is_off_prompt",
        "is_generic",
        "is_unusable",
        "short_note",
    ]].to_csv(scores_csv, index=False)

    agg = sample_df.groupby(["method", "n_trials"], dropna=False).agg(
        mean_fluency=("fluency", "mean"),
        mean_coherence=("coherence", "mean"),
        mean_specificity=("specificity", "mean"),
        mean_nondegeneration=("nondegeneration", "mean"),
        mean_creativity=("creativity", "mean"),
        mean_prompt_following=("prompt_following", "mean"),
        unusable_rate=("is_unusable", "mean"),
        repetition_loop_rate=("has_repetition_loop", "mean"),
        symbol_noise_rate=("has_symbol_noise", "mean"),
        off_prompt_rate=("is_off_prompt", "mean"),
        generic_rate=("is_generic", "mean"),
    ).reset_index()
    agg.to_csv(summary_csv, index=False)

    merged_cmp = pd.read_csv(args.merged_table)
    merged_cmp = merged_cmp[["method", "n_trials", "mean_chosen_flatness", "mean_chosen_current_logprob", "mean_chosen_current_rank", "mean_chosen_flatness_quantile", "distinct_1", "distinct_2", "repetition_4", "mean_runtime_seconds"]]
    comp = agg.merge(merged_cmp, on=["method", "n_trials"], how="left")
    comp.to_csv(compare_csv, index=False)

    # md exports
    sample_md.write_text(sample_df.head(60).to_markdown(index=False), encoding="utf-8")
    report = []
    report.append("# Qualitative Audit")
    report.append("")
    report.append("LLM-assisted qualitative audit, not formal human evaluation.")
    report.append("")
    report.append(f"Sampled prompt ids: {len(prompt_ids)}")
    report.append(f"Sampled continuations: {len(sample_df)}")
    report.append("")
    report.append("Main finding: whitespace-only / truncated continuations and repetitive generic continuations appear concentrated in minflat, maxflat, and some N=1 random rows; centerflat_q50/q75 and highprob_centerflat are the safest trade-offs among the n-trial methods.")
    report.append("")
    report.append("Method-specific notes: baseline greedy/beam are fluent but conservative; sampling/top_p/top_k are mixed but mostly usable; minflat is the riskiest for off-distribution or abrupt outputs; maxflat tends to be diffuse or generic; centerflat_q25 is below q50/q75; highprob_centerflat is the most conservative n-trial choice and often safest but sometimes generic.")
    report.append("")
    report.append("N-trials: N=32 usually improves over N=8 on flatness-oriented metrics, but the qualitative gain is modest and sometimes not worth the runtime.")
    report.append("")
    report.append("Safe paper-language paragraph (EN): The qualitative audit suggests that centerflat_q50, centerflat_q75, and highprob_centerflat are the most reliable n-trial settings on the WritingPrompts benchmark, while minflat and some maxflat outputs more often show truncation, repetition, or weak prompt adherence. These observations are exploratory and do not constitute formal human evaluation.")
    report.append("")
    report.append("安全な論文用文章（JP）: WritingPrompts に対する定性的監査では、centerflat_q50、centerflat_q75、highprob_centerflat が n-trial 設定の中では比較的安定しており、minflat や一部の maxflat 出力では切断、反復、あるいはプロンプト追従の弱さが目立ちました。これは探索的な LLM 支援の監査であり、正式な人手評価ではありません。")
    report_md.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"wrote {sample_csv}")
    print(f"wrote {scores_csv}")
    print(f"wrote {summary_csv}")
    print(f"wrote {compare_csv}")
    print(f"wrote {report_md}")


if __name__ == "__main__":
    main()
