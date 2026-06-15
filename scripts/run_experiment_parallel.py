from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from flatness_decoding.io_utils import read_jsonl
from flatness_decoding.parallelism import detect_parallel_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-shards", default="auto", help="Integer or 'auto'.")
    parser.add_argument("--workers", default="auto", help="Integer or 'auto'.")
    parser.add_argument(
        "--devices",
        nargs="*",
        default=None,
        help=(
            "Optional CUDA_VISIBLE_DEVICES values assigned round-robin, e.g. '0 1'. "
            "Use only for CUDA multi-GPU runs. Omit for auto/default device behavior."
        ),
    )
    parser.add_argument("--candidate-eval-batch-size", type=int, default=None, help="Override candidate lookahead micro-batch size.")
    parser.add_argument("--inference-mode", action="store_true", help="Run shard generation calls inside torch.inference_mode().")
    parser.add_argument("--resume", action="store_true", help="Skip complete output groups when possible.")
    parser.add_argument("--no-resume", action="store_true", help="Disable complete-output skipping even if config enables it.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true", help="Print resolved plan without running shards.")
    return parser.parse_args()


def prompt_count_from_config(config_path: str | Path) -> int:
    with Path(config_path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    prompts = read_jsonl(cfg["prompts"]["path"])
    limit = cfg.get("prompts", {}).get("limit")
    if limit is not None:
        prompts = prompts[: int(limit)]
    return len(prompts)


def run_shard(
    python: str,
    config: str,
    num_shards: int,
    devices: list[str] | None,
    shard_index: int,
    candidate_eval_batch_size: int | None,
    inference_mode: bool,
    resume: bool,
    no_resume: bool,
) -> tuple[int, int]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src" + os.pathsep + env.get("PYTHONPATH", "")
    if devices:
        env["CUDA_VISIBLE_DEVICES"] = str(devices[shard_index % len(devices)])

    cmd = [
        python,
        "scripts/run_experiment_set.py",
        "--config",
        config,
        "--num-shards",
        str(num_shards),
        "--shard-index",
        str(shard_index),
    ]
    if candidate_eval_batch_size is not None:
        cmd.extend(["--candidate-eval-batch-size", str(candidate_eval_batch_size)])
    if inference_mode:
        cmd.append("--inference-mode")
    if resume:
        cmd.append("--resume")
    if no_resume:
        cmd.append("--no-resume")

    print("+ " + " ".join(cmd))
    completed = subprocess.run(cmd, env=env)
    return shard_index, completed.returncode


def main() -> None:
    args = parse_args()
    if not Path(args.config).exists():
        raise SystemExit(f"config not found: {args.config}")
    if args.resume and args.no_resume:
        raise SystemExit("--resume and --no-resume are mutually exclusive")

    plan = detect_parallel_plan(
        requested_num_shards=args.num_shards,
        requested_workers=args.workers,
    )
    prompt_count = prompt_count_from_config(args.config)
    num_shards = plan.recommended_num_shards
    if args.num_shards == "auto" and prompt_count > 0:
        num_shards = min(num_shards, prompt_count)
    workers = min(plan.recommended_workers, num_shards) if num_shards > 0 else 1
    devices = args.devices if args.devices is not None else plan.recommended_devices

    print("Resolved parallel plan")
    print(f"backend: {plan.backend}")
    print(f"prompt_count: {prompt_count}")
    print(f"workers: {workers}")
    print(f"num_shards: {num_shards}")
    print(f"devices: {devices}")
    print(f"candidate_eval_batch_size_override: {args.candidate_eval_batch_size}")
    print(f"inference_mode_override: {args.inference_mode}")
    print(f"resume_override: {args.resume}")
    print(f"no_resume_override: {args.no_resume}")
    print(f"caution: {plan.caution}")

    if prompt_count == 0:
        raise SystemExit("no prompts found for this config")

    if args.dry_run:
        return

    failed: list[tuple[int, int]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                run_shard,
                args.python,
                args.config,
                num_shards,
                devices,
                shard_idx,
                args.candidate_eval_batch_size,
                args.inference_mode,
                args.resume,
                args.no_resume,
            )
            for shard_idx in range(num_shards)
        ]
        for future in as_completed(futures):
            shard_idx, code = future.result()
            if code != 0:
                failed.append((shard_idx, code))
                print(f"shard {shard_idx} failed with code {code}")
            else:
                print(f"shard {shard_idx} completed")

    if failed:
        raise SystemExit(f"failed shards: {failed}")

    print("all shards completed")


if __name__ == "__main__":
    main()
