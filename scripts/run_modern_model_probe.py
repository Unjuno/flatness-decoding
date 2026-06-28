from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


CONFIGS = {
    "gemma2-2b": "configs/gemma2_2b_writingprompts_wrapped_10.yaml",
    "qwen2.5-1.5b-instruct": "configs/qwen2_5_1_5b_instruct_writingprompts_wrapped_10.yaml",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plan or run small modern-model probes.")
    p.add_argument("--model-key", choices=sorted(CONFIGS), required=True)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = Path(CONFIGS[args.model_key])
    if not config.exists():
        raise SystemExit(f"config not found: {config}")
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    n_values = cfg.get("n_trial_methods", {}).get("n_values", [])
    if any(int(n) > 8 for n in n_values):
        raise SystemExit("modern-model probes refuse n_values > 8")
    if any(int(n) in {16, 32} for n in n_values):
        raise SystemExit("modern-model probes refuse N=16 or N=32")
    cmd = [
        sys.executable,
        "scripts/run_experiment_parallel.py",
        "--config",
        str(config),
        "--num-shards",
        str(args.num_shards),
        "--workers",
        str(args.workers),
    ]
    if args.resume:
        cmd.append("--resume")
    if not args.execute:
        cmd.append("--dry-run")
    print("model_key:", args.model_key)
    print("config:", config)
    print("command:", " ".join(shlex.quote(p) for p in cmd))
    if args.dry_run or not args.execute:
        return
    completed = subprocess.run(cmd, cwd=Path.cwd(), check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
