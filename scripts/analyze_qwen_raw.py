from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from flatness_decoding.io_utils import read_jsonl


TITLE_RE = re.compile(r"(?:^|\n)\s*(Title:|Characters:|Character Concept|Story Title|Name:|Age:)", re.I)
COMMENTARY_RE = re.compile(r"\b(I think|What do you think|premise|ethical problem|Reader discretion|Preamble)\b", re.I)
HTML_RE = re.compile(r"<\/?(?:h1|h2|h3|p|div|span)[^>]*>", re.I)
PROMPT_MARKER_RE = re.compile(r"\[(?:\s*WP\s*|WP|wp|RPG)\]", re.I)
TOKEN_FRAG_RE = re.compile(r"(?:##|▁|@@|Ġ|Ċ){2,}|[A-Za-z]{1,3}(?:##|▁|@@|Ġ|Ċ)", re.I)
MULTI_RE = re.compile(r"[\u0080-\uffff]{2,}")
SUBWORD_REPEAT_RE = re.compile(r"(\b\w{1,4}\b(?:\s+\b\w{1,4}\b){2,})")
INSTR_RE = re.compile(r"^(Sure|Certainly|Of course|Here is|Here’s|Here are|I can|As an AI|To answer|Let me)", re.I)
QUESTION_RE = re.compile(r"\?")
PROMPT_LEAK_RE = re.compile(r"continue the following|do not analyze the prompt|do not write a title|do not write a preamble|prompt:", re.I)
SEM_INCOH_RE = re.compile(r"(?:\b(?:however|therefore|first|second|moreover)\b.{0,80}\b(?:prompt|instruction|task)\b)|(?:\b(?:(?:The|A)\s+)?(?:answer|instruction|prompt)\b)", re.I)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze Qwen raw WritingPrompts outputs.")
    p.add_argument("--run-dir", default="results/qwen2_5_1_5b_instruct_raw/10")
    p.add_argument("--output-dir", default="results/analysis")
    p.add_argument("--prefix", default="qwen2_5_1_5b_raw10")
    return p.parse_args()


def load_rows(run_dir: Path) -> pd.DataFrame:
    gen_rows = []
    for path in sorted((run_dir / "generations").glob("*.jsonl")):
        gen_rows.extend(read_jsonl(path))
    tok_rows = []
    for path in sorted((run_dir / "token_logs").glob("*.jsonl")):
        tok_rows.extend(read_jsonl(path))
    gen_df = pd.DataFrame(gen_rows)
    tok_df = pd.DataFrame(tok_rows)
    if not tok_df.empty and "prompt_id" in tok_df.columns:
        tok_df = tok_df.groupby(["method", "prompt_id", "n_trials"], dropna=False).agg(
            mean_chosen_flatness=("chosen_flatness", "mean"),
            mean_chosen_current_rank=("chosen_current_rank", "mean"),
            mean_candidate_unique_count=("candidate_unique_count", "mean"),
        ).reset_index()
    return gen_df.merge(tok_df, on=["method", "prompt_id", "n_trials"], how="left") if not tok_df.empty else gen_df


def flags(text: str) -> dict[str, float]:
    text = text or ""
    lower = text.lower()
    direct_story = 1.0 if text.strip() and not any(text.lstrip().startswith(prefix) for prefix in ("#", "-", "*", "<", "[", "Title:", "Characters:")) else 0.0
    return {
        "title_or_character_sheet": 1.0 if TITLE_RE.search(text) else 0.0,
        "commentary_or_analysis": 1.0 if COMMENTARY_RE.search(text) else 0.0,
        "html_or_markup": 1.0 if HTML_RE.search(text) else 0.0,
        "prompt_marker_recycling": 1.0 if PROMPT_MARKER_RE.search(text) else 0.0,
        "direct_story_opening": direct_story,
        "instruction_like_reversion": 1.0 if INSTR_RE.search(text[:120]) else 0.0,
        "token_fragment_collapse": 1.0 if TOKEN_FRAG_RE.search(text[:180]) or len(text.strip()) < 20 else 0.0,
        "multilingual_garbage": 1.0 if MULTI_RE.search(text) else 0.0,
        "repeated_subword_pattern": 1.0 if SUBWORD_REPEAT_RE.search(text[:220]) else 0.0,
        "semantic_incoherence": 1.0 if SEM_INCOH_RE.search(text[:220]) else 0.0,
        "prompt_relevance": 1.0 if any(w in lower for w in ["story", "character", "prompt", "scene", "rain", "road", "house", "city", "man", "woman"]) else 0.0,
    }


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_rows(run_dir)
    if df.empty:
        raise SystemExit(f"no generation rows found in {run_dir}")
    df["continuation_text"] = df["continuation_text"].fillna("")
    flag_df = df["continuation_text"].apply(flags).apply(pd.Series)
    df = pd.concat([df, flag_df], axis=1)
    summary = df.groupby(["method", "n_trials"], dropna=False).agg(
        row_count=("method", "size"),
        prompt_count=("prompt_id", "nunique"),
        mean_runtime_seconds=("runtime_seconds", "mean"),
        mean_tokens_per_second=("tokens_per_second", "mean"),
        mean_output_tokens=("num_output_tokens", "mean"),
        mean_chosen_successor_logit_dispersion=("mean_chosen_flatness", "mean"),
        mean_chosen_current_rank=("mean_chosen_current_rank", "mean"),
        mean_candidate_unique_count=("mean_candidate_unique_count", "mean"),
        title_or_character_sheet_rate=("title_or_character_sheet", "mean"),
        commentary_or_analysis_rate=("commentary_or_analysis", "mean"),
        html_or_markup_rate=("html_or_markup", "mean"),
        prompt_marker_recycling_rate=("prompt_marker_recycling", "mean"),
        direct_story_opening_rate=("direct_story_opening", "mean"),
        instruction_like_reversion_rate=("instruction_like_reversion", "mean"),
        token_fragment_collapse_rate=("token_fragment_collapse", "mean"),
        multilingual_garbage_rate=("multilingual_garbage", "mean"),
        repeated_subword_pattern_rate=("repeated_subword_pattern", "mean"),
        semantic_incoherence_rate=("semantic_incoherence", "mean"),
        prompt_relevance_rate=("prompt_relevance", "mean"),
    ).reset_index()
    summary_path = out_dir / f"{args.prefix}_summary.csv"
    tradeoff_path = out_dir / f"{args.prefix}_method_tradeoff.csv"
    notes_path = out_dir / f"{args.prefix}_qualitative_notes.md"
    summary.to_csv(summary_path, index=False)
    summary.sort_values(["n_trials", "method"]).to_csv(tradeoff_path, index=False)
    notes_path.write_text(
        "# Qwen raw qualitative notes\n\n"
        "This is a small raw WritingPrompts Qwen 2.5 1.5B-Instruct control run.\n\n"
        "Use it to compare prompt framing against the wrapped10 condition, not to rank decoding methods or claim quality improvement.\n",
        encoding="utf-8",
    )
    print(f"wrote {summary_path}")
    print(f"wrote {tradeoff_path}")
    print(f"wrote {notes_path}")


if __name__ == "__main__":
    main()
