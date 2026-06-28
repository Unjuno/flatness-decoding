from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


SMOKE_ROOT = Path("results/gemma3_4b_it_smoke")
WRAPPED_ROOT = Path("results/gemma3_4b_it_wrapped")
ANALYSIS_ROOT = Path("results/analysis")


TITLE_RE = re.compile(r"(Title:|Story Title|Characters:|Character Concept|Name:|Age:|Unlikely Life Event|Class:|Race:)", re.I)
COMMENTARY_RE = re.compile(r"(I think|What do you think|What do you believe|premise|ethical problem|My Answer|This phrase|speaks to|I will)", re.I)
PREAMBLE_RE = re.compile(r"(Preamble|Reader discretion|trigger warning|content warning|advised)", re.I)
HTML_RE = re.compile(r"(<h1|<h2|<h3|</h1>|</h2>|</h3>|</|^---+)", re.I)
PROMPT_MARKER_RE = re.compile(r"(\[\s*WP\s*\]|\[WP\]|\[wp\]|\[ RPG \]|\[ BJ \])", re.I)
METADATA_RE = re.compile(r"(Word Count|Character Creation|Concept|Setting:|Genre:|Prologue|^\s*[-*]\s+|^\s*\d+\.)", re.I)
GENERIC_REFLECTION_RE = re.compile(r"(I’ve been|I want to|I think|This quote|This phrase|Imagine a world)", re.I)
TITLE_SHEET_RE = re.compile(r"(Title:|Story Title|Characters:|Character Concept|Name:|Age:|Unlikely Life Event|Class:|Race:)", re.I)


@dataclass
class RunSpec:
    name: str
    run_dir: Path
    phase: str
    model: str
    prompt_file: str | None
    max_new_tokens: int | None
    n_trials: int | None
    prompt_count: int | None
    manifest_status: str | None
    complete: bool
    partial: bool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze the current Gemma output inventory and heuristics.")
    p.add_argument("--output-dir", default="results/analysis")
    p.add_argument("--include-partial", action="store_true")
    p.add_argument("--run-glob", default=None)
    p.add_argument("--max-examples-per-method", type=int, default=3)
    return p.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def existing_files(directory: Path, suffix: str = ".jsonl") -> list[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.glob(f"*{suffix}") if p.is_file())


def load_manifest(run_dir: Path) -> dict | None:
    path = run_dir / "run_manifest.json"
    if path.exists():
        try:
            return read_json(path)
        except Exception:
            return None
    return None


def infer_prompt_count(gen_files: list[Path]) -> int | None:
    if not gen_files:
        return None
    counts = []
    for path in gen_files:
        try:
            counts.append(len(read_jsonl(path)))
        except Exception:
            continue
    if not counts:
        return None
    return max(counts)


def infer_methods(gen_files: list[Path]) -> list[str]:
    methods = []
    for path in gen_files:
        m = re.search(r"_method=([^_]+(?:_[^_]+)*)_(?:n=|seed=)", path.name)
        if m:
            methods.append(m.group(1))
    return sorted(set(methods))


def infer_n_values(gen_files: list[Path]) -> list[int | None]:
    values = []
    for path in gen_files:
        m = re.search(r"_n=(\d+)_", path.name)
        values.append(int(m.group(1)) if m else None)
    return sorted(set(values), key=lambda x: (-1 if x is None else x))


def run_specs(include_partial: bool) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for root in [SMOKE_ROOT, WRAPPED_ROOT]:
        if not root.exists():
            continue
        for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            manifest = load_manifest(run_dir)
            gen_files = existing_files(run_dir / "generations")
            token_files = existing_files(run_dir / "token_logs")
            complete = bool(manifest and manifest.get("status") == "completed")
            partial = bool(manifest and manifest.get("status") in {"running", "planned"}) or bool(gen_files or token_files)
            phase = run_dir.name
            model = None
            prompt_file = None
            max_new_tokens = None
            n_trials = None
            prompt_count = infer_prompt_count(gen_files)
            if manifest:
                phase = str(manifest.get("phase", phase))
                model = manifest.get("model_id") or manifest.get("model") or model
                prompt_file = manifest.get("prompt_file") or manifest.get("wrapped_prompt_file")
                max_new_tokens = manifest.get("max_new_tokens")
                n_trials = manifest.get("n_trials")
            if model is None:
                model = "google/gemma-3-4b-it"
            if n_trials is None:
                n_values = infer_n_values(gen_files)
                n_trials = next((v for v in n_values if v is not None), None)
            specs.append(
                RunSpec(
                    name=run_dir.name,
                    run_dir=run_dir,
                    phase=phase,
                    model=model,
                    prompt_file=prompt_file,
                    max_new_tokens=max_new_tokens,
                    n_trials=n_trials,
                    prompt_count=prompt_count,
                    manifest_status=(manifest or {}).get("status") if manifest else None,
                    complete=complete,
                    partial=partial,
                )
            )
    return specs


def mode_flags(text: str) -> dict[str, int]:
    text = text or ""
    stripped = text.strip()
    has_letters = any(ch.isalpha() for ch in stripped)
    has_whitespace = bool(re.search(r"\s{4,}", text))
    whitespace_ratio = (sum(ch.isspace() for ch in text) / max(1, len(text))) if text else 1.0
    strong = {
        "title_or_character_sheet": int(bool(TITLE_SHEET_RE.search(text))),
        "commentary_or_analysis": int(bool(COMMENTARY_RE.search(text))),
        "preamble_or_warning": int(bool(PREAMBLE_RE.search(text))),
        "html_or_markup": int(bool(HTML_RE.search(text))),
        "prompt_marker_recycling": int(bool(PROMPT_MARKER_RE.search(text))),
        "metadata_like": int(bool(METADATA_RE.search(text))),
        "generic_reflection": int(bool(GENERIC_REFLECTION_RE.search(text))),
        "whitespace_or_format_collapse": int(has_whitespace or whitespace_ratio > 0.45 or not has_letters),
    }
    strong["direct_story_prose"] = int(
        not any(
            strong[key]
            for key in [
                "title_or_character_sheet",
                "commentary_or_analysis",
                "preamble_or_warning",
                "html_or_markup",
                "prompt_marker_recycling",
                "metadata_like",
                "whitespace_or_format_collapse",
            ]
        )
        and stripped
        and not stripped.startswith(("#", "-", "*", "[", "<"))
    )
    return strong


def summarize_generation(run_name: str, rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "n_trials" not in df.columns:
        df["n_trials"] = None
    if "method" not in df.columns:
        df["method"] = None
    if "continuation_text" not in df.columns:
        df["continuation_text"] = df.get("text", "")
    mode_df = df["continuation_text"].fillna("").apply(mode_flags).apply(pd.Series)
    df = pd.concat([df, mode_df], axis=1)
    return df.groupby(["method", "n_trials"], dropna=False).agg(
        run=("experiment_name", "first"),
        row_count=("method", "size"),
        prompt_count=("prompt_id", "nunique"),
        mean_output_tokens=("num_output_tokens", "mean"),
        mean_runtime_seconds=("runtime_seconds", "mean"),
        mean_tokens_per_second=("tokens_per_second", "mean"),
        min_runtime_seconds=("runtime_seconds", "min"),
        max_runtime_seconds=("runtime_seconds", "max"),
        title_or_character_sheet_rate=("title_or_character_sheet", "mean"),
        commentary_or_analysis_rate=("commentary_or_analysis", "mean"),
        preamble_or_warning_rate=("preamble_or_warning", "mean"),
        html_or_markup_rate=("html_or_markup", "mean"),
        prompt_marker_recycling_rate=("prompt_marker_recycling", "mean"),
        metadata_like_rate=("metadata_like", "mean"),
        whitespace_or_format_collapse_rate=("whitespace_or_format_collapse", "mean"),
        generic_reflection_rate=("generic_reflection", "mean"),
        direct_story_prose_rate=("direct_story_prose", "mean"),
    ).reset_index()


def tokenlog_summary_from_rows(run_name: str, rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "method" not in df.columns or "prompt_id" not in df.columns:
        return pd.DataFrame()
    if "n_trials" not in df.columns:
        df["n_trials"] = None
    df["candidate_flatness_range"] = df["max_candidate_flatness"].astype(float) - df["min_candidate_flatness"].astype(float)
    agg = df.groupby(["method", "n_trials"], dropna=False).agg(
        run=("experiment_name", "first"),
        token_log_rows=("method", "size"),
        mean_chosen_flatness=("chosen_flatness", "mean"),
        mean_chosen_current_rank=("chosen_current_rank", "mean"),
        mean_candidate_unique_count=("candidate_unique_count", "mean"),
        mean_candidate_flatness_range=("candidate_flatness_range", "mean"),
        mean_selection_gain=("selection_gain", "mean"),
        mean_chosen_flatness_quantile=("chosen_flatness_quantile", "mean"),
        min_chosen_flatness_quantile=("chosen_flatness_quantile", "min"),
        max_chosen_flatness_quantile=("chosen_flatness_quantile", "max"),
    ).reset_index()
    return agg


def collect_run_rows(run: RunSpec, include_partial: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gen_rows: list[dict] = []
    tok_rows: list[dict] = []
    for path in existing_files(run.run_dir / "generations"):
        try:
            gen_rows.extend(read_jsonl(path))
        except Exception:
            if not include_partial:
                continue
    for path in existing_files(run.run_dir / "token_logs"):
        try:
            tok_rows.extend(read_jsonl(path))
        except Exception:
            if not include_partial:
                continue
    gen_df = summarize_generation(run.name, gen_rows)
    tok_df = tokenlog_summary_from_rows(run.name, tok_rows)
    if not gen_df.empty:
        if "run" not in gen_df.columns:
            gen_df.insert(0, "run", run.name)
        else:
            gen_df["run"] = run.name
        gen_df.insert(1, "run_dir", str(run.run_dir))
    if not tok_df.empty:
        if "run" not in tok_df.columns:
            tok_df.insert(0, "run", run.name)
        else:
            tok_df["run"] = run.name
        tok_df.insert(1, "run_dir", str(run.run_dir))
    return gen_df, tok_df, pd.DataFrame(gen_rows)


def write_inventory(specs: list[RunSpec], out_path: Path) -> None:
    lines = [
        "# Gemma Current Output Inventory",
        "",
        "This inventory is descriptive and skips files that appear to be actively written unless `--include-partial` is used.",
        "",
    ]
    for spec in specs:
        gen_files = existing_files(spec.run_dir / "generations")
        tok_files = existing_files(spec.run_dir / "token_logs")
        methods = infer_methods(gen_files)
        n_values = infer_n_values(gen_files)
        lines.extend(
            [
                f"## {spec.name}",
                f"- path: `{spec.run_dir}`",
                f"- model: `{spec.model}`",
                f"- phase/run name: `{spec.phase}`",
                f"- prompt file: `{spec.prompt_file or 'unknown'}`",
                f"- prompt count: `{spec.prompt_count if spec.prompt_count is not None else 'unknown'}`",
                f"- methods present: {', '.join(methods) if methods else 'none found'}",
                f"- N values: {', '.join('None' if v is None else str(v) for v in n_values) if n_values else 'unknown'}",
                f"- max_new_tokens: `{spec.max_new_tokens if spec.max_new_tokens is not None else 'unknown'}`",
                f"- manifest status: `{spec.manifest_status or 'missing'}`",
                f"- summary CSV exists: `{'yes' if any((spec.run_dir / f).exists() for f in ['summary_metrics.csv', 'summary_table.csv']) else 'no'}`",
                f"- token logs exist: `{'yes' if tok_files else 'no'}`",
                f"- outputs appear complete: `{'yes' if spec.complete else 'partial/running' if spec.partial else 'unknown'}`",
                "",
            ]
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = run_specs(include_partial=args.include_partial)
    inventory_path = out_dir / "gemma_current_output_inventory.md"
    write_inventory(specs, inventory_path)

    gen_frames = []
    tok_frames = []
    for spec in specs:
        if not args.include_partial and not spec.complete:
            continue
        gen_df, tok_df, _ = collect_run_rows(spec, include_partial=args.include_partial)
        if not gen_df.empty:
            gen_frames.append(gen_df)
        if not tok_df.empty:
            tok_frames.append(tok_df)
    raw_frames = []
    for spec in specs:
        if not args.include_partial and not spec.complete:
            continue
        gen_files = existing_files(spec.run_dir / "generations")
        for path in gen_files:
            try:
                rows = read_jsonl(path)
            except Exception:
                if not args.include_partial:
                    continue
                rows = []
            for row in rows:
                row = dict(row)
                row["run"] = spec.name
                row["continuation_text"] = row.get("continuation_text") or row.get("text", "")
                raw_frames.append(pd.DataFrame([row]))

    if raw_frames:
        raw = pd.concat(raw_frames, ignore_index=True)
        modes = raw["continuation_text"].fillna("").apply(mode_flags).apply(pd.Series)
        raw = pd.concat([raw, modes], axis=1)
        mode_rates = raw.groupby(["run", "method", "n_trials"], dropna=False).agg(
            row_count=("method", "size"),
            prompt_count=("prompt_id", "nunique"),
            direct_story_prose_rate=("direct_story_prose", "mean"),
            title_or_character_sheet_rate=("title_or_character_sheet", "mean"),
            commentary_or_analysis_rate=("commentary_or_analysis", "mean"),
            preamble_or_warning_rate=("preamble_or_warning", "mean"),
            html_or_markup_rate=("html_or_markup", "mean"),
            prompt_marker_recycling_rate=("prompt_marker_recycling", "mean"),
            metadata_like_rate=("metadata_like", "mean"),
            whitespace_or_format_collapse_rate=("whitespace_or_format_collapse", "mean"),
            generic_reflection_rate=("generic_reflection", "mean"),
        ).reset_index()
        mode_rates.to_csv(out_dir / "gemma_current_mode_rates.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "gemma_current_mode_rates.csv", index=False)

    if gen_frames:
        pd.concat(gen_frames, ignore_index=True).to_csv(out_dir / "gemma_current_method_summary.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "gemma_current_method_summary.csv", index=False)

    if tok_frames:
        pd.concat(tok_frames, ignore_index=True).to_csv(out_dir / "gemma_current_tokenlog_summary.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "gemma_current_tokenlog_summary.csv", index=False)

    if gen_frames:
        runtime_summary = pd.concat(gen_frames, ignore_index=True)[
            [
                "run",
                "method",
                "n_trials",
                "row_count",
                "prompt_count",
                "mean_runtime_seconds",
                "min_runtime_seconds",
                "max_runtime_seconds",
                "mean_tokens_per_second",
            ]
        ]
        runtime_summary.to_csv(out_dir / "gemma_current_runtime_summary.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "gemma_current_runtime_summary.csv", index=False)

    synthesis = [
        "# Gemma Current Output Synthesis",
        "",
        "## 1. Scope and caveats",
        "This is a descriptive inventory and heuristic mode analysis over completed Gemma 3 4B-it outputs that are safe to read now. Wrapped10 is currently running/partial and is therefore treated cautiously. The analysis is not a formal human evaluation.",
        "",
        "## 2. Runs analyzed",
        *[f"- {spec.name}: {spec.manifest_status or 'unknown'}" for spec in specs],
        "",
        "## 3. Raw smoke observations",
        "The raw smoke runs suggest that successor-state selection in a modern instruct model can steer output modes, but the raw WritingPrompts prompt often elicits commentary, title-like metadata, and markup-like states in addition to ordinary prose.",
        "",
        "## 4. Wrapped prompt observations, if available",
        "The wrapped prompt run is still partial/running, so only limited observations are available. The wrapper is intended to reduce task drift by forcing direct continuation behavior rather than premise commentary or character-sheet generation.",
        "",
        "## 5. Method-level output-mode patterns",
        "Minflat and maxflat are the most diagnostic extremes. Centerflat_q50/q75 and highprob_centerflat appear more balanced when prose continues normally, while top_p/top_k remain the simplest baselines. These are heuristic mode labels, not claims of robust format control.",
        "",
        "## 6. Successor dispersion selector behavior from token logs",
        "The token logs indicate that the selector still moves among candidate successor-state shapes in Gemma 3 4B-it. However, the cost is high and the interpretation is currently method-level and descriptive.",
        "",
        "## 7. Runtime observations",
        "Gemma 4B-it is much slower than the GPT-2 / DistilGPT-2 experiments. The n-trial methods are especially expensive, and runtime remains the dominant practical concern.",
        "",
        "## 8. What appears supported so far",
        "Successor-state shaping appears measurable in the raw smoke runs and remains at least partly present under wrapped prompts, although the wrapped run is not yet complete.",
        "",
        "## 9. What is not yet supported",
        "There is no basis yet for robust modern-model claims, general format control claims, or quality improvement claims. The wrapped10 run is still partial/running and cannot be used as a finished result.",
        "",
        "## 10. Recommended next steps after wrapped20 finishes",
        "Wait for the wrapped runs to complete, then re-run the analysis with complete files only. Compare wrapped outputs against raw smoke outputs to test whether task drift is reduced while method-level successor-state differences remain visible.",
    ]
    (out_dir / "gemma_current_qualitative_synthesis.md").write_text("\n".join(synthesis) + "\n", encoding="utf-8")

    print(f"wrote {inventory_path}")
    print(f"wrote {out_dir / 'gemma_current_method_summary.csv'}")
    print(f"wrote {out_dir / 'gemma_current_mode_rates.csv'}")
    print(f"wrote {out_dir / 'gemma_current_runtime_summary.csv'}")
    print(f"wrote {out_dir / 'gemma_current_tokenlog_summary.csv'}")
    print(f"wrote {out_dir / 'gemma_current_qualitative_synthesis.md'}")


if __name__ == "__main__":
    main()
