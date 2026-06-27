from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


TECHNICAL_DEFAULT_METHODS = ["top_p", "random", "minflat", "maxflat", "centerflat_q50"]
PAPER_DEFAULT_METHODS = [
    "top_p",
    "top_k",
    "random",
    "minflat",
    "maxflat",
    "centerflat_q50",
    "centerflat_q75",
    "highprob_centerflat",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build staged Gemma 3 4B-it smoke commands.")
    parser.add_argument("--model-id", default="google/gemma-3-4b-it")
    parser.add_argument("--phase", choices=["technical", "paper", "optional"], default="technical")
    parser.add_argument("--prompt-file", default="prompts/writingprompts_fixed_100.jsonl")
    parser.add_argument("--num-prompts", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--methods", default=None, help="Comma-separated method list.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-large-n", action="store_true")
    parser.add_argument("--allow-large-run", action="store_true")
    return parser.parse_args()


def defaults_for_phase(phase: str) -> tuple[int, int, int, list[str]]:
    if phase == "technical":
        return 2, 4, 16, TECHNICAL_DEFAULT_METHODS
    if phase == "paper":
        return 5, 8, 32, PAPER_DEFAULT_METHODS
    return 10, 8, 32, PAPER_DEFAULT_METHODS


def manifest_path(output_dir: Path) -> Path:
    return output_dir / "run_manifest.json"


def write_manifest(output_dir: Path, data: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_path(output_dir)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    default_num_prompts, default_n_trials, default_max_new_tokens, default_methods = defaults_for_phase(args.phase)
    num_prompts = args.num_prompts if args.num_prompts is not None else default_num_prompts
    n_trials = args.n_trials if args.n_trials is not None else default_n_trials
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else default_max_new_tokens
    methods = [m.strip() for m in (args.methods.split(",") if args.methods else default_methods) if m.strip()]
    output_dir = Path(args.output_dir or f"results/gemma3_4b_it_smoke/{args.phase}")

    if n_trials > 8 and not args.allow_large_n:
        raise SystemExit("Gemma staged smoke refuses n_trials > 8 unless --allow-large-n is passed.")
    if n_trials >= 32:
        raise SystemExit("Gemma staged smoke always refuses n_trials >= 32.")
    if num_prompts > 20 and not args.allow_large_run:
        raise SystemExit("Gemma staged smoke refuses num_prompts > 20 unless --allow-large-run is passed.")

    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_id": args.model_id,
        "phase": args.phase,
        "num_prompts": num_prompts,
        "n_trials": n_trials,
        "max_new_tokens": max_new_tokens,
        "methods": methods,
        "dtype": args.dtype,
        "device": args.device,
        "prompt_file": args.prompt_file,
        "output_dir": str(output_dir),
        "local_files_only": args.local_files_only,
        "status": "planned" if args.dry_run else "running",
    }
    write_manifest(output_dir, manifest)

    config_path = output_dir / f"gemma3_4b_it_{args.phase}.yaml"
    config = {
        "status": "planned",
        "experiment_role": "modern instruct-model sanity check",
        "model_id": args.model_id,
        "prompt_file": args.prompt_file,
        "num_prompts": num_prompts,
        "n_trials": n_trials,
        "max_new_tokens": max_new_tokens,
        "methods": methods,
        "device": args.device,
        "dtype": args.dtype,
        "local_files_only": args.local_files_only,
        "output_dir": str(output_dir),
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")

    cmd = [
        sys.executable,
        "scripts/run_experiment_parallel.py",
        "--config",
        str(config_path),
        "--num-shards",
        "1",
        "--workers",
        "1",
        "--dry-run",
    ]
    print("planned phase:", args.phase)
    print("manifest:", manifest_path(output_dir))
    print("config:", config_path)
    print("command:", " ".join(shlex.quote(part) for part in cmd))

    if args.dry_run:
        return

    completed = subprocess.run(cmd, cwd=Path.cwd(), env=os.environ.copy(), check=False)
    if completed.returncode != 0:
        manifest["status"] = "failed"
        write_manifest(output_dir, manifest)
        raise SystemExit(completed.returncode)
    manifest["status"] = "completed"
    write_manifest(output_dir, manifest)


if __name__ == "__main__":
    main()
