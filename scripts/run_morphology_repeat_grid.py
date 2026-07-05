from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from analyze_successor_distribution_morphology import CONDITION_SPECS


DEFAULT_METHODS = [
    "random",
    "minflat",
    "maxflat",
    "centerflat_q25",
    "centerflat_q50",
    "centerflat_q75",
    "highprob_centerflat",
]


def parse_csv_like(values: list[str]) -> list[str]:
    parts: list[str] = []
    for value in values:
        parts.extend(item.strip() for item in value.split(",") if item.strip())
    return parts


def stable_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run repeat-backed morphology generation grids and optional analysis.")
    p.add_argument("--config", default=None, help="Optional YAML config describing the repeat-grid run.")
    p.add_argument("--conditions", action="append", default=[], help="Repeatable or comma-separated condition names.")
    p.add_argument("--methods", action="append", default=[], help="Repeatable or comma-separated n-trial methods.")
    p.add_argument("--prompt-limit", type=int, default=20)
    p.add_argument("--repeat-count", type=int, default=20)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--n-values", nargs="+", type=int, default=[1, 4, 8, 16])
    p.add_argument("--max-new-tokens", type=int, default=60)
    p.add_argument("--output-root", default=str(ROOT / "results" / "repeat_generations"))
    p.add_argument("--analysis-output-root", default=str(ROOT / "results" / "analysis"))
    p.add_argument("--generation-root", default=str(ROOT / "results"))
    p.add_argument("--analysis-preset", default="morphology_20x20_top20")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--dry-run", action="store_true", default=True)
    p.add_argument("--estimate-only", action="store_true")
    p.add_argument("--skip-analysis", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--allow-gemma", action="store_true")
    p.add_argument("--confirm-large-run", action="store_true")
    p.add_argument("--model-cache-dir", default=None)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    return p.parse_args()


def load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise SystemExit(f"config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise SystemExit("config must be a YAML mapping")
    return data


def condition_rows(conditions: list[str]) -> list[str]:
    resolved = stable_unique(parse_csv_like(conditions) or ["qwen_raw"])
    for condition in resolved:
        if condition not in CONDITION_SPECS:
            raise SystemExit(f"unknown condition: {condition}")
    return resolved


def generation_dirs_for_condition(condition: str, generation_root: Path) -> tuple[Path, Path]:
    spec = CONDITION_SPECS[condition]
    gen_dir = spec.get("generation_dir")
    tok_dir = spec.get("token_log_dir")
    if gen_dir is None:
        gen_dir = generation_root / condition / "generations"
    else:
        gen_dir = Path(gen_dir)
    if tok_dir is None:
        tok_dir = generation_root / condition / "token_logs"
    else:
        tok_dir = Path(tok_dir)
    return gen_dir, tok_dir


def estimate_rows(max_prompts: int, repeat_count: int, n_values: list[int], methods: list[str]) -> tuple[int, int]:
    generation_rows = max_prompts * repeat_count * len(n_values) * len(methods)
    token_log_rows = generation_rows
    return generation_rows, token_log_rows


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    if cfg:
        if cfg.get("conditions") is not None and not args.conditions:
            args.conditions = [str(v) for v in (cfg.get("conditions") or [])]
        if cfg.get("methods") is not None and not args.methods:
            args.methods = [str(v) for v in (cfg.get("methods") or [])]
        args.prompt_limit = int(cfg.get("prompt_limit", args.prompt_limit))
        args.repeat_count = int(cfg.get("repeat_count", args.repeat_count))
        args.seed_start = int(cfg.get("seed_start", args.seed_start))
        args.n_values = [int(v) for v in cfg.get("n_values", args.n_values)]
        args.max_new_tokens = int(cfg.get("max_new_tokens", args.max_new_tokens))
        args.analysis_preset = str(cfg.get("analysis_preset", args.analysis_preset))
        args.skip_analysis = bool(cfg.get("skip_analysis", args.skip_analysis))
        args.resume = bool(cfg.get("resume", args.resume))
        args.allow_gemma = bool(cfg.get("allow_gemma", args.allow_gemma))
        args.confirm_large_run = bool(cfg.get("confirm_large_run", args.confirm_large_run))
        if cfg.get("model_cache_dir") is not None and args.model_cache_dir is None:
            args.model_cache_dir = str(cfg.get("model_cache_dir"))
        if cfg.get("local_files_only") is not None:
            args.local_files_only = bool(cfg.get("local_files_only"))
        if cfg.get("device") is not None:
            args.device = str(cfg.get("device"))
        if cfg.get("dtype") is not None:
            args.dtype = str(cfg.get("dtype"))

    conditions = condition_rows(args.conditions)
    methods = stable_unique(parse_csv_like(args.methods) or DEFAULT_METHODS)
    generation_root = Path(args.generation_root)
    output_root = Path(args.output_root)
    analysis_output_root = Path(args.analysis_output_root)
    generation_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    analysis_output_root.mkdir(parents=True, exist_ok=True)

    generation_rows, token_log_rows = estimate_rows(args.prompt_limit, args.repeat_count, args.n_values, methods)
    print("conditions:", ", ".join(conditions))
    print("methods:", ", ".join(methods))
    print("prompt_limit:", args.prompt_limit)
    print("repeat_count:", args.repeat_count)
    print("seed_start:", args.seed_start)
    print("n_values:", ", ".join(str(n) for n in args.n_values))
    print("analysis_preset:", args.analysis_preset)
    print("config:", args.config or "none")
    print("estimated_generation_rows_per_condition:", generation_rows)
    print("estimated_token_log_rows_per_condition:", token_log_rows)

    if args.estimate_only or not args.execute:
        return

    env = {**os.environ, "PYTHONPATH": "src"}
    for condition in conditions:
        spec = CONDITION_SPECS[condition]
        if spec.get("requires_allow_gemma") and not args.allow_gemma:
            raise SystemExit(f"refusing {condition} without --allow-gemma")
        if args.prompt_limit > 100 and not args.confirm_large_run:
            raise SystemExit("--prompt-limit > 100 requires --confirm-large-run")
        if args.repeat_count > 20 and not args.confirm_large_run:
            raise SystemExit("--repeat-count > 20 requires --confirm-large-run")
        gen_dir, tok_dir = generation_dirs_for_condition(condition, generation_root)
        gen_dir.parent.mkdir(parents=True, exist_ok=True)
        tok_dir.parent.mkdir(parents=True, exist_ok=True)

        gen_cmd = [
            sys.executable,
            "scripts/run_sweep_n.py",
            "--model",
            spec["model_name"],
            "--prompts",
            str(spec["prompt_path"]),
            "--limit",
            str(args.prompt_limit),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--methods",
            *methods,
            "--n-values",
            *[str(n) for n in args.n_values],
            "--repeat-count",
            str(args.repeat_count),
            "--seed-start",
            str(args.seed_start),
            "--device",
            args.device,
            "--dtype",
            args.dtype,
            "--output-dir",
            str(gen_dir),
            "--token-log-dir",
            str(tok_dir),
        ]
        if args.resume:
            gen_cmd.append("--resume")
        if args.local_files_only:
            gen_cmd.append("--local-files-only")

        print(f"running generation for {condition} -> {gen_dir}")
        subprocess.run(gen_cmd, check=True, cwd=ROOT, env=env)

        if args.skip_analysis:
            continue

        analysis_dir = analysis_output_root / f"morphology_repeat_grid_{condition}"
        analysis_cmd = [
            sys.executable,
            "scripts/analyze_successor_distribution_morphology.py",
            "--preset",
            args.analysis_preset,
            "--device",
            "cpu",
            "--conditions",
            condition,
            "--methods",
            *methods,
            "--max-prompts",
            str(args.prompt_limit),
            "--repeat-count",
            str(args.repeat_count),
            "--seed-start",
            str(args.seed_start),
            "--generation-dir",
            str(gen_dir),
            "--token-log-dir",
            str(tok_dir),
            "--output-dir",
            str(analysis_dir),
            "--execute",
            "--resume",
        ]
        if args.model_cache_dir:
            analysis_cmd.extend(["--model-cache-dir", args.model_cache_dir])
        if args.local_files_only:
            analysis_cmd.append("--local-files-only")
        if args.allow_gemma:
            analysis_cmd.append("--allow-gemma")
        if args.confirm_large_run:
            analysis_cmd.append("--confirm-large-run")

        print(f"running morphology analysis for {condition} -> {analysis_dir}")
        subprocess.run(analysis_cmd, check=True, cwd=ROOT, env=env)

    print("repeat-grid run complete")


if __name__ == "__main__":
    import os

    main()
