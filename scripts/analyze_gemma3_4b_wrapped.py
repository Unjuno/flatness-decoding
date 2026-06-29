from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from flatness_decoding.io_utils import read_jsonl


TITLE_RE = re.compile(r"(?:^|\n)\s*(Title:|Characters:|Character Concept|Story Title|Name:|Age:)", re.I)
COMMENTARY_RE = re.compile(r"\b(I think|What do you think|premise|ethical problem|Reader discretion|Preamble)\b", re.I)
HTML_RE = re.compile(r"<\/?(?:h1|h2|h3|p|div|span)[^>]*>", re.I)
PROMPT_MARKER_RE = re.compile(r"\[(?:\s*WP\s*|WP|wp|RPG)\]", re.I)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze Gemma wrapped smoke outputs.")
    p.add_argument("--run-dir", default="results/gemma3_4b_it_wrapped/10")
    p.add_argument("--output-dir", default="results/analysis")
    p.add_argument("--prefix", default=None)
    return p.parse_args()


def result_prefix(run_dir: Path, prefix: str | None) -> str:
    if prefix:
        return prefix
    phase = run_dir.name
    return f"gemma3_4b_wrapped{phase}"


def load_rows(run_dir: Path) -> pd.DataFrame:
    gen_dir = run_dir / "generations"
    token_dir = run_dir / "token_logs"
    gen_rows = []
    for path in sorted(gen_dir.glob("*.jsonl")):
        gen_rows.extend(read_jsonl(path))
    token_rows = []
    for path in sorted(token_dir.glob("*.jsonl")):
        token_rows.extend(read_jsonl(path))
    gen_df = pd.DataFrame(gen_rows)
    tok_df = pd.DataFrame(token_rows)
    if not tok_df.empty and "prompt_id" in tok_df.columns:
        tok_df = tok_df.groupby(["method", "prompt_id", "n_trials"], dropna=False).agg(
            mean_chosen_flatness=("chosen_flatness", "mean"),
            mean_chosen_current_rank=("chosen_current_rank", "mean"),
            mean_candidate_unique_count=("candidate_unique_count", "mean"),
        ).reset_index()
    return gen_df.merge(tok_df, on=["method", "prompt_id", "n_trials"], how="left") if not tok_df.empty else gen_df


def artifact_flags(text: str) -> dict[str, float]:
    lower = text.lower()
    title = 1.0 if TITLE_RE.search(text) else 0.0
    commentary = 1.0 if COMMENTARY_RE.search(text) else 0.0
    html = 1.0 if HTML_RE.search(text) else 0.0
    marker = 1.0 if PROMPT_MARKER_RE.search(text) else 0.0
    direct_story = 1.0 if text.strip() and not any(text.lstrip().startswith(prefix) for prefix in ("#", "-", "*", "<", "[", "Title:", "Characters:")) else 0.0
    return {
        "title_or_character_sheet": title,
        "commentary": commentary,
        "html_or_markup": html,
        "prompt_marker": marker,
        "direct_story_opening": direct_story,
    }


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = result_prefix(run_dir, args.prefix)
    df = load_rows(run_dir)
    if df.empty:
        raise SystemExit(f"no generation rows found in {run_dir}")
    df["continuation_text"] = df["continuation_text"].fillna("")
    flags = df["continuation_text"].apply(artifact_flags).apply(pd.Series)
    df = pd.concat([df, flags], axis=1)
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
        commentary_rate=("commentary", "mean"),
        html_or_markup_rate=("html_or_markup", "mean"),
        prompt_marker_rate=("prompt_marker", "mean"),
        direct_story_opening_rate=("direct_story_opening", "mean"),
    ).reset_index()
    summary_path = out_dir / f"{prefix}_summary.csv"
    tradeoff_path = out_dir / f"{prefix}_method_tradeoff.csv"
    notes_path = out_dir / f"{prefix}_qualitative_notes.md"
    summary.to_csv(summary_path, index=False)
    summary.sort_values(["n_trials", "method"]).to_csv(tradeoff_path, index=False)
    notes_path.write_text(
        "# Gemma wrapped qualitative notes\n\n"
        "This is a small wrapped-prompt Gemma 3 4B-it sanity check.\n\n"
        "Compare with the raw smoke runs qualitatively, not as a benchmark.\n"
        "The wrapper should reduce task drift if it works, but the analysis remains exploratory.\n",
        encoding="utf-8",
    )
    print(f"wrote {summary_path}")
    print(f"wrote {tradeoff_path}")
    print(f"wrote {notes_path}")


if __name__ == "__main__":
    main()
