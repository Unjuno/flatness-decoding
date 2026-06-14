from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--devices",
        nargs="*",
        default=None,
        help=(
            "Optional CUDA_VISIBLE_DEVICES values assigned round-robin, e.g. '0 1'. "
            "Use only for CUDA multi-GPU runs. Omit for CPU/MPS/default device behavior."
        ),
    )
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def run_shard(args: argparse.Namespace, shard_index: int) -> tuple[int, int]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src" + os.pathsep + env.get("PYTHONPATH", "")
    if args.devices:
        env["CUDA_VISIBLE_DEVICES"] = str(args.devices[shard_index % len(args.devices)])

    cmd = [
        args.python,
        "scripts/run_experiment_set.py",
        "--config",
        args.config,
        "--num-shards",
        str(args.num_shards),
        "--shard-index",
        str(shard_index),
    ]
    print("+ " + " ".join(cmd))
    completed = subprocess.run(cmd, env=env)
    return shard_index, completed.returncode


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if not Path(args.config).exists():
        raise SystemExit(f"config not found: {args.config}")

    failed: list[tuple[int, int]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_shard, args, shard_idx) for shard_idx in range(args.num_shards)]
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
