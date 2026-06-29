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


WRAPPED_10_METHODS = [
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
    p = argparse.ArgumentParser(description="Build and optionally run wrapped Gemma 3 4B-it continuation experiments.")
    p.add_argument("--phase", choices=["wrapped10", "wrapped20"], default="wrapped10")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--candidate-eval-batch-size", type=int, default=1)
    p.add_argument("--inference-mode", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--allow-swap", action="store_true")
    p.add_argument("--base-output-dir", default="results/gemma3_4b_it_wrapped")
    p.add_argument("--wrapped-prompt-file", default="prompts/writingprompts_fixed_100_gemma_wrapped.jsonl")
    p.add_argument("--model-id", default="google/gemma-3-4b-it")
    p.add_argument("--model-cache-dir", default="models/hf")
    return p.parse_args()


def ensure_wrapped_prompts(source: Path, dest: Path) -> None:
    if dest.exists():
        return
    if not source.exists():
        raise SystemExit(f"source prompt file not found: {source}")
    rows = []
    intro = (
        "Continue the following WritingPrompts story prompt as narrative prose.\n\n"
        "Do not analyze the prompt.\n"
        "Do not list characters.\n"
        "Do not write a title.\n"
        "Do not write a preamble.\n"
        "Do not explain the premise.\n"
        "Continue directly in story form.\n\n"
        "Prompt:\n"
    )
    for row in _read_jsonl(source):
        original = dict(row)
        prompt = str(original.get("prompt", ""))
        wrapped = dict(original)
        wrapped["original_prompt"] = prompt
        wrapped["prompt"] = f"{intro}{prompt}\n\nContinuation:"
        rows.append(wrapped)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(dest, rows)


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_config(args: argparse.Namespace, output_dir: Path, prompt_file: str, limit: int, max_new_tokens: int) -> dict[str, object]:
    methods = WRAPPED_10_METHODS
    return {
        "experiment_name": f"gemma3_4b_it_writingprompts_{args.phase}",
        "model_name": args.model_id,
        "model_cache_dir": args.model_cache_dir,
        "device": "auto",
        "dtype": "auto",
        "seeds": [0],
        "generation": {"max_new_tokens": max_new_tokens},
        "prompts": {"path": prompt_file, "limit": limit},
        "baselines": {
            "methods": ["top_p", "top_k"],
            "temperature": 1.2,
            "top_p": 0.9,
            "top_k": 50,
            "num_beams": 4,
        },
        "n_trial_methods": {
            "methods": [m for m in methods if m not in {"top_p", "top_k"}],
            "n_values": [8],
            "proposal_temperature": 1.0,
            "use_kv_cache_lookahead": False,
            "candidate_eval_batch_size": int(args.candidate_eval_batch_size),
            "highprob_logprob_margin": 5.0,
            "highprob_rank_limit": 100,
        },
        "runtime": {
            "inference_mode": bool(args.inference_mode),
            "resume": {"skip_complete": bool(args.resume)},
            "cleanup": {
                "device_cache_after_prompt": False,
                "device_cache_after_group": True,
                "python_gc_after_group": False,
            },
        },
        "output": {
            "generations_dir": str(output_dir / "generations"),
            "token_log_dir": str(output_dir / "token_logs"),
            "summary_csv": str(output_dir / "summary_metrics.csv"),
            "table_csv": str(output_dir / "summary_table.csv"),
            "figures_dir": str(output_dir / "figures"),
        },
        "gemma_wrapped": {
            "status": "planned",
            "experiment_role": "modern instruct-model wrapped continuation sanity check",
            "prompt_mode": "explicit_story_continuation_wrapper",
            "n_policy": "N=8 only; N=16/N=32 excluded",
        },
    }


def phase_spec(phase: str) -> tuple[str, int, int]:
    if phase == "wrapped10":
        return "10", 10, 48
    return "20", 20, 64


def main() -> None:
    args = parse_args()
    phase_dir, prompt_limit, max_new_tokens = phase_spec(args.phase)
    output_dir = Path(args.base_output_dir) / phase_dir
    wrapped_prompt_path = Path(args.wrapped_prompt_file)
    ensure_wrapped_prompts(Path("prompts/writingprompts_fixed_100.jsonl"), wrapped_prompt_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = build_config(args, output_dir, str(wrapped_prompt_path), prompt_limit, max_new_tokens)
    config_path = output_dir / f"gemma3_4b_it_{args.phase}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")

    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "phase": args.phase,
        "config": str(config_path),
        "output_dir": str(output_dir),
        "wrapped_prompt_file": str(wrapped_prompt_path),
        "status": "planned" if args.dry_run or not args.execute else "running",
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    cmd = [
        sys.executable,
        "scripts/run_experiment_parallel.py",
        "--config",
        str(config_path),
        "--num-shards",
        str(args.num_shards),
        "--workers",
        str(args.workers),
        "--candidate-eval-batch-size",
        str(args.candidate_eval_batch_size),
    ]
    if args.inference_mode:
        cmd.append("--inference-mode")
    if args.resume:
        cmd.append("--resume")
    if args.allow_swap:
        cmd.append("--allow-swap")
    if not args.execute:
        cmd.append("--dry-run")

    print("phase:", args.phase)
    print("config:", config_path)
    print("wrapped_prompt_file:", wrapped_prompt_path)
    print("command:", " ".join(shlex.quote(x) for x in cmd))

    if args.dry_run or not args.execute:
        return

    completed = subprocess.run(cmd, cwd=Path.cwd(), env=os.environ.copy(), check=False)
    if completed.returncode != 0:
        manifest["status"] = "failed"
        (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        raise SystemExit(completed.returncode)
    manifest["status"] = "completed"
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
