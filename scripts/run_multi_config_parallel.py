from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import yaml


# Model parameter counts (approximate) for memory estimation.
MODEL_PARAMS: dict[str, int] = {
    "distilgpt2": 82_000_000,
    "gpt2": 124_000_000,
    "gpt2-medium": 355_000_000,
    "Qwen/Qwen2.5-0.5B": 500_000_000,
    "Qwen/Qwen2.5-1.5B-Instruct": 1_500_000_000,
    "google/gemma-2-2b": 2_000_000_000,
    "google/gemma-3-4b-it": 4_000_000_000,
}

# Safety multiplier on top of raw weight memory (accounts for KV cache,
# activations, optimizer states during eval, etc.).
MEMORY_SAFETY_FACTOR = 2.0


def safe_filename(value: str) -> str:
    """Convert an arbitrary identifier into a filesystem-safe name."""
    return value.replace("/", "__").replace("\\", "__")


def estimate_model_memory_bytes(model_name: str, dtype: str = "auto") -> int:
    """Estimate peak memory for a model in bytes."""
    params = MODEL_PARAMS.get(model_name)
    if params is None:
        # Unknown model: assume 1B params as fallback.
        params = 1_000_000_000
    bytes_per_param = 2 if dtype in ("float16", "bfloat16") else 4
    return int(params * bytes_per_param * MEMORY_SAFETY_FACTOR)


def get_available_memory_bytes() -> int:
    """Get available system memory in bytes (macOS + Linux)."""
    import platform
    if platform.system() == "Darwin":
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True,
        )
        total = int(result.stdout.strip())
        # On macOS, estimate ~60% available for model loading
        # (OS + other apps use the rest).
        return int(total * 0.6)
    else:
        # Linux: try /proc/meminfo
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            for line in meminfo.read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
        return 0


def get_available_mps_memory_bytes() -> int:
    """Get available MPS memory (macOS unified memory)."""
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # MPS shares system memory; return available system memory.
            return get_available_memory_bytes()
    except ImportError:
        pass
    return 0


@dataclass
class ConfigJob:
    config_path: str
    model_name: str
    estimated_bytes: int
    status: str = "pending"
    returncode: int | None = None
    elapsed_seconds: float = 0.0
    error_msg: str = ""


@dataclass
class ScheduleResult:
    batch: list[ConfigJob]
    total_estimated_bytes: int


def schedule_batches(
    jobs: list[ConfigJob],
    available_bytes: int,
) -> list[list[ConfigJob]]:
    """Greedy bin-packing: fill batches up to available memory."""
    # Sort by estimated size descending (largest first).
    sorted_jobs = sorted(jobs, key=lambda j: j.estimated_bytes, reverse=True)
    batches: list[list[ConfigJob]] = []
    current_batch: list[ConfigJob] = []
    current_used = 0

    for job in sorted_jobs:
        if current_used + job.estimated_bytes <= available_bytes and current_batch:
            current_batch.append(job)
            current_used += job.estimated_bytes
        elif current_used + job.estimated_bytes <= available_bytes and not current_batch:
            # Single job fits on its own.
            current_batch.append(job)
            current_used += job.estimated_bytes
        else:
            # Doesn't fit; start a new batch.
            if current_batch:
                batches.append(current_batch)
            current_batch = [job]
            current_used = job.estimated_bytes

    if current_batch:
        batches.append(current_batch)

    return batches


def run_config(args: tuple[str, str, bool, float | None]) -> tuple[str, int, float, str]:
    """Run a single config as a subprocess. Returns (config_path, returncode, elapsed, error)."""
    config_path, python_bin, resume, timeout_seconds = args
    start = time.monotonic()
    env = os.environ.copy()
    env["PYTHONPATH"] = "src" + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        python_bin,
        "scripts/run_pipeline.py",
        "--config", config_path,
    ]

    try:
        run_kwargs: dict[str, object] = {
            "env": env,
            "capture_output": True,
            "text": True,
        }
        if timeout_seconds is not None and timeout_seconds > 0:
            run_kwargs["timeout"] = timeout_seconds
        completed = subprocess.run(cmd, **run_kwargs)
        elapsed = time.monotonic() - start
        if completed.returncode != 0:
            err_lines = (completed.stderr or "").strip().splitlines()
            error_msg = "\n".join(err_lines[-10:]) if err_lines else "unknown error"
            return config_path, completed.returncode, elapsed, error_msg
        return config_path, 0, elapsed, ""
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        return config_path, -1, elapsed, f"TIMEOUT ({timeout_seconds:.0f}s)" if timeout_seconds else "TIMEOUT"
    except Exception as e:
        elapsed = time.monotonic() - start
        return config_path, -1, elapsed, str(e)


def wait_for_memory(seconds: float = 30) -> None:
    """Wait for memory to free up after OOM."""
    import gc
    gc.collect()
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    time.sleep(seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple model configs in parallel, memory-aware.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="Config YAML files to run.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip completed outputs (default: True).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable skip-complete behavior.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print schedule without executing.",
    )
    parser.add_argument(
        "--max-memory-gb",
        type=float,
        default=None,
        help="Override available memory (GB). If not set, auto-detect.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max automatic retries for failed configs (default: 3).",
    )
    parser.add_argument(
        "--retry-wait",
        type=float,
        default=30.0,
        help="Seconds to wait before retrying a failed config (default: 30).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=0.0,
        help="Per-config timeout in seconds. 0 disables the timeout (default: 0).",
    )
    parser.add_argument(
        "--log-dir",
        default="outputs/multi_config_logs",
        help="Directory for failure logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resume = not args.no_resume

    # Load configs and estimate memory.
    jobs: list[ConfigJob] = []
    for config_path in args.configs:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        model_name = cfg.get("model_name", "unknown")
        dtype = cfg.get("dtype", "auto")
        est = estimate_model_memory_bytes(model_name, dtype)
        jobs.append(ConfigJob(
            config_path=config_path,
            model_name=model_name,
            estimated_bytes=est,
        ))

    # Detect available memory.
    if args.max_memory_gb is not None:
        available = int(args.max_memory_gb * 1e9)
    else:
        available = get_available_memory_bytes()
    if available == 0:
        # Fallback: be conservative, run one at a time.
        available = max(j.estimated_bytes for j in jobs)

    print(f"Available memory (estimated for model loading): {available / 1e9:.1f} GB")
    print(f"Configs to run: {len(jobs)}")
    print()

    for j in jobs:
        print(f"  {j.config_path}")
        print(f"    model: {j.model_name}")
        print(f"    estimated memory: {j.estimated_bytes / 1e9:.2f} GB")
    print()

    # Schedule into batches.
    batches = schedule_batches(jobs, available)
    print(f"Scheduled {len(batches)} batch(es):")
    for i, batch in enumerate(batches):
        total = sum(j.estimated_bytes for j in batch)
        names = [j.model_name for j in batch]
        print(f"  batch {i}: {names} ({total / 1e9:.2f} GB)")
    print()

    if args.dry_run:
        return

    # Create log directory.
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Execute batches sequentially; within each batch, run configs in parallel.
    # Failed configs are automatically retried up to max_retries times.
    all_results: list[tuple[str, int, float, str]] = []
    failed_configs: list[str] = []

    # Build retry queue: list of (config_path, remaining_retries).
    retry_queue: list[tuple[str, int]] = [(j.config_path, args.max_retries) for j in jobs]

    attempt = 0
    while retry_queue:
        attempt += 1
        # Re-estimate memory for remaining configs.
        pending_jobs: list[ConfigJob] = []
        for config_path, retries_left in retry_queue:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            model_name = cfg.get("model_name", "unknown")
            dtype = cfg.get("dtype", "auto")
            est = estimate_model_memory_bytes(model_name, dtype)
            pending_jobs.append(ConfigJob(
                config_path=config_path,
                model_name=model_name,
                estimated_bytes=est,
            ))

        batches = schedule_batches(pending_jobs, available)
        print(f"--- Attempt {attempt} ({len(retry_queue)} configs, {len(batches)} batch(es)) ---")
        for i, batch in enumerate(batches):
            total = sum(j.estimated_bytes for j in batch)
            names = [j.model_name for j in batch]
            print(f"  batch {i}: {names} ({total / 1e9:.2f} GB)")
        print()

        next_retry: list[tuple[str, int]] = []

        for batch_idx, batch in enumerate(batches):
            print(f"  === Batch {batch_idx + 1}/{len(batches)} ({len(batch)} configs) ===")
            batch_start = time.monotonic()

            if len(batch) == 1:
                job = batch[0]
                config_path, code, elapsed, error = run_config(
                    (job.config_path, args.python, resume, args.timeout_seconds or None)
                )
                job.returncode = code
                job.elapsed_seconds = elapsed
                job.error_msg = error
                retries_left = next(r for p, r in retry_queue if p == job.config_path)
                if code == 0:
                    job.status = "completed"
                    print(f"    [OK] {job.model_name} ({elapsed:.1f}s)")
                    all_results.append((config_path, 0, elapsed, ""))
                else:
                    job.status = "failed"
                    print(f"    [FAIL] {job.model_name} (code={code}, {elapsed:.1f}s)")
                    if error:
                        print(f"      error: {error[:200]}")
                    log_path = log_dir / f"{safe_filename(job.model_name)}_failure.log"
                    log_path.write_text(
                        f"config: {job.config_path}\n"
                        f"model: {job.model_name}\n"
                        f"returncode: {code}\n"
                        f"elapsed: {elapsed:.1f}s\n"
                        f"attempt: {attempt}\n"
                        f"\nerror:\n{error}\n"
                    )
                    if retries_left > 1:
                        print(f"    waiting {args.retry_wait}s then retrying ({retries_left - 1} left)...")
                        wait_for_memory(args.retry_wait)
                        next_retry.append((config_path, retries_left - 1))
                    else:
                        failed_configs.append(config_path)
                        all_results.append((config_path, code, elapsed, error))
            else:
                with ProcessPoolExecutor(max_workers=len(batch)) as executor:
                    futures = {
                        executor.submit(
                            run_config,
                            (job.config_path, args.python, resume, args.timeout_seconds or None),
                        ): job
                        for job in batch
                    }
                    for future in as_completed(futures):
                        job = futures[future]
                        config_path, code, elapsed, error = future.result()
                        job.returncode = code
                        job.elapsed_seconds = elapsed
                        job.error_msg = error
                        retries_left = next(r for p, r in retry_queue if p == job.config_path)
                        if code == 0:
                            job.status = "completed"
                            print(f"    [OK] {job.model_name} ({elapsed:.1f}s)")
                            all_results.append((config_path, 0, elapsed, ""))
                        else:
                            job.status = "failed"
                            print(f"    [FAIL] {job.model_name} (code={code}, {elapsed:.1f}s)")
                            if error:
                                print(f"      error: {error[:200]}")
                            log_path = log_dir / f"{safe_filename(job.model_name)}_failure.log"
                            log_path.write_text(
                                f"config: {job.config_path}\n"
                                f"model: {job.model_name}\n"
                                f"returncode: {code}\n"
                                f"elapsed: {elapsed:.1f}s\n"
                                f"attempt: {attempt}\n"
                                f"\nerror:\n{error}\n"
                            )
                            if retries_left > 1:
                                print(f"    waiting {args.retry_wait}s then retrying ({retries_left - 1} left)...")
                                next_retry.append((config_path, retries_left - 1))
                            else:
                                failed_configs.append(config_path)
                                all_results.append((config_path, code, elapsed, error))

            batch_elapsed = time.monotonic() - batch_start
            print(f"  batch {batch_idx + 1} finished in {batch_elapsed:.1f}s")
            print()

        retry_queue = next_retry

    # Summary.
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_elapsed = sum(r[2] for r in all_results)
    succeeded = sum(1 for r in all_results if r[1] == 0)
    failed = sum(1 for r in all_results if r[1] != 0)
    print(f"Total configs: {len(all_results)}")
    print(f"Succeeded: {succeeded}")
    print(f"Failed: {failed}")
    print(f"Total wall time: {total_elapsed:.1f}s")
    print()

    if failed_configs:
        print("Failed configs (exhausted all retries):")
        for p in failed_configs:
            print(f"  {p}")
        print(f"\nFailure logs in: {log_dir}/")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
