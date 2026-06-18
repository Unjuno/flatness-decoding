from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
PROMPTS_PATH = ROOT / "prompts" / "writingprompts_fixed_100.jsonl"
LOG_DIR = ROOT / "logs"

EXPECTED_PROMPT_FIELDS = {
    "prompt",
    "source_dataset",
    "source_split",
    "source_row_index",
    "prompt_field",
    "reference_field",
    "reference",
}


@dataclass
class RunRecord:
    stage: str
    command: list[str]
    returncode: int
    stdout: str
    stderr: str


class AutoRunnerError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-build-prompts", action="store_true")
    parser.add_argument("--skip-tiny", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--skip-distilgpt2", action="store_true")
    parser.add_argument("--skip-gpt2", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-swap", action="store_true")
    parser.add_argument("--min-free-ram-gb", type=float, default=2.0)
    parser.add_argument("--min-free-vram-gb", type=float, default=1.0)
    parser.add_argument("--yes", action="store_true")
    return parser.parse_args()


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def line(msg: str) -> None:
    print(msg, flush=True)


def fmt_cmd(cmd: Sequence[str], env: dict[str, str] | None = None) -> str:
    parts = []
    if env:
        for key in ("PYTHONPATH",):
            if key in env:
                parts.append(f"{key}={env[key]}")
    parts.extend(cmd)
    return " ".join(parts)


def run_command(
    stage: str,
    cmd: list[str],
    log: list[RunRecord],
    env: dict[str, str] | None = None,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str] | None:
    line(f"\n== {stage} ==")
    line(f"$ {fmt_cmd(cmd, env)}")
    if dry_run:
        return None
    try:
        completed = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True, check=True)
    except subprocess.CalledProcessError as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if stdout:
            print(stdout, end="" if stdout.endswith("\n") else "\n")
        if stderr:
            print(stderr, end="" if stderr.endswith("\n") else "\n", file=sys.stderr)
        log.append(
            RunRecord(
                stage=stage,
                command=list(cmd),
                returncode=exc.returncode,
                stdout=stdout,
                stderr=stderr,
            )
        )
        raise
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    if completed.stderr:
        print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n", file=sys.stderr)
    log.append(
        RunRecord(stage=stage, command=list(cmd), returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)
    )
    return completed


def write_log(path: Path, start: datetime, end: datetime, log: list[RunRecord], status: str, failed_stage: str | None = None, failed_command: list[str] | None = None, failed_code: int | None = None) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"start_time: {start.isoformat()}\n")
        f.write(f"end_time: {end.isoformat()}\n")
        f.write(f"final_status: {status}\n")
        if failed_stage is not None:
            f.write(f"failed_stage: {failed_stage}\n")
        if failed_command is not None:
            f.write(f"failed_command: {json.dumps(failed_command)}\n")
        if failed_code is not None:
            f.write(f"failed_exit_code: {failed_code}\n")
        for record in log:
            f.write("\n---\n")
            f.write(f"stage: {record.stage}\n")
            f.write(f"command: {json.dumps(record.command)}\n")
            f.write(f"exit_code: {record.returncode}\n")
            f.write("stdout:\n")
            f.write(record.stdout)
            if record.stdout and not record.stdout.endswith("\n"):
                f.write("\n")
            f.write("stderr:\n")
            f.write(record.stderr)
            if record.stderr and not record.stderr.endswith("\n"):
                f.write("\n")


def ensure_log_dir() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR / f"writingprompts_auto_{now_stamp()}.log"


def read_non_empty_lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_jsonl_object(line_text: str, path: Path, index: int) -> dict[str, object]:
    try:
        value = json.loads(line_text)
    except json.JSONDecodeError as exc:
        raise AutoRunnerError(f"invalid JSON on line {index} in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AutoRunnerError(f"line {index} in {path} is not a JSON object")
    return value


def check_prompt_artifact(skip_build_prompts: bool, dry_run: bool, log: list[RunRecord]) -> None:
    if not PROMPTS_PATH.exists():
        if skip_build_prompts:
            raise AutoRunnerError(f"missing prompt artifact: {PROMPTS_PATH}")
        run_command("build prompts", ["make", "build-prompts"], log, dry_run=dry_run)
    if dry_run and not PROMPTS_PATH.exists():
        return
    if not PROMPTS_PATH.exists():
        raise AutoRunnerError(f"missing prompt artifact after build: {PROMPTS_PATH}")
    lines = read_non_empty_lines(PROMPTS_PATH)
    if len(lines) != 100:
        raise AutoRunnerError(f"expected 100 non-empty prompt lines in {PROMPTS_PATH}, found {len(lines)}")
    for idx, raw in enumerate(lines[:5], start=1):
        row = parse_jsonl_object(raw, PROMPTS_PATH, idx)
        missing = EXPECTED_PROMPT_FIELDS - set(row)
        if missing:
            raise AutoRunnerError(f"prompt row {idx} missing fields: {sorted(missing)}")
        if row.get("source_dataset") != "euclaise/writingprompts":
            raise AutoRunnerError(f"prompt row {idx} has unexpected source_dataset: {row.get('source_dataset')!r}")
        if row.get("source_split") != "validation":
            raise AutoRunnerError(f"prompt row {idx} has unexpected source_split: {row.get('source_split')!r}")
        if row.get("prompt_field") != "prompt":
            raise AutoRunnerError(f"prompt row {idx} has unexpected prompt_field: {row.get('prompt_field')!r}")
        if row.get("reference_field") != "story":
            raise AutoRunnerError(f"prompt row {idx} has unexpected reference_field: {row.get('reference_field')!r}")


def import_resource_checker():
    try:
        from flatness_decoding.resources import check_runtime_resources, format_bytes

        return check_runtime_resources, format_bytes
    except Exception:
        return None, None


def preflight_resources(min_free_ram_gb: float, min_free_vram_gb: float, allow_swap: bool) -> None:
    checker, formatter = import_resource_checker()
    if checker is None or formatter is None:
        line("warning: resource checker import failed; using lightweight fallback")
        return
    result = checker()
    line("resource preflight")
    line(f"available_ram: {formatter(result.available_ram_bytes)}")
    line(f"swap_used: {formatter(result.swap_used_bytes)}")
    line(f"cuda_device_index: {result.cuda_device_index}")
    line(f"free_vram: {formatter(result.free_vram_bytes)}")
    if result.swap_used_bytes and result.swap_used_bytes > 0 and not allow_swap:
        raise AutoRunnerError("swap is in use; rerun with --allow-swap only if you accept the risk")
    if result.available_ram_bytes is not None and result.available_ram_bytes < int(min_free_ram_gb * 1024**3):
        raise AutoRunnerError(f"available RAM below threshold: {formatter(result.available_ram_bytes)} < {min_free_ram_gb} GiB")
    if result.free_vram_bytes is not None and result.free_vram_bytes < int(min_free_vram_gb * 1024**3):
        raise AutoRunnerError(f"free VRAM below threshold: {formatter(result.free_vram_bytes)} < {min_free_vram_gb} GiB")


def check_target_exists(target: str) -> None:
    completed = subprocess.run(["make", "-n", target], cwd=ROOT, text=True, capture_output=True)
    if completed.returncode != 0:
        raise AutoRunnerError(f"missing or invalid make target: {target}\n{completed.stderr.strip()}")


def check_static_targets() -> None:
    for target in [
        "comparison-tiny",
        "comparison-smoke",
        "postprocess-smoke",
        "comparison-distilgpt2-main",
        "postprocess-distilgpt2-main",
        "comparison-main",
        "postprocess-main",
    ]:
        check_target_exists(target)


def planned_commands() -> list[tuple[str, list[str], dict[str, str] | None]]:
    return [
        (
            "planning dry-run",
            [
                sys.executable,
                "scripts/run_experiment_parallel.py",
                "--config",
                "configs/comparison_tiny.yaml",
                "--num-shards",
                "1",
                "--workers",
                "1",
                "--min-free-ram-gb",
                "0",
                "--min-free-vram-gb",
                "0",
                "--dry-run",
            ],
            {"PYTHONPATH": "src"},
        ),
        ("tiny: comparison", ["make", "comparison-tiny"], None),
        ("tiny: postprocess", ["make", "postprocess-tiny"], None),
        ("smoke: comparison", ["make", "comparison-smoke"], None),
        ("smoke: postprocess", ["make", "postprocess-smoke"], None),
        ("distilgpt2: comparison", ["make", "comparison-distilgpt2-main"], None),
        ("distilgpt2: postprocess", ["make", "postprocess-distilgpt2-main"], None),
        ("gpt2: comparison", ["make", "comparison-main"], None),
        ("gpt2: postprocess", ["make", "postprocess-main"], None),
    ]


def print_full_plan(include_tiny: bool, include_smoke: bool, include_distilgpt2: bool, include_gpt2: bool) -> None:
    line("full plan")
    for stage, cmd, env in planned_commands():
        if stage.startswith("tiny") and not include_tiny:
            continue
        if stage.startswith("smoke") and not include_smoke:
            continue
        if stage.startswith("distilgpt2") and not include_distilgpt2:
            continue
        if stage.startswith("gpt2") and not include_gpt2:
            continue
        line(f"  {stage}: {fmt_cmd(cmd, env)}")


def check_output_dir(path: Path) -> None:
    if not path.exists() or not path.is_dir():
        raise AutoRunnerError(f"missing output directory: {path}")
    if not any(path.iterdir()):
        raise AutoRunnerError(f"empty output directory: {path}")


def check_file(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise AutoRunnerError(f"missing or empty file: {path}")


def snapshot_pollution(paths: list[Path]) -> dict[Path, tuple[bool, float | None, int | None]]:
    snap: dict[Path, tuple[bool, float | None, int | None]] = {}
    for path in paths:
        if path.exists():
            stat = path.stat()
            snap[path] = (True, stat.st_mtime, stat.st_size)
        else:
            snap[path] = (False, None, None)
    return snap


def detect_pollution(before: dict[Path, tuple[bool, float | None, int | None]], paths: list[Path]) -> list[str]:
    problems: list[str] = []
    for path in paths:
        existed, old_mtime, old_size = before[path]
        if not path.exists():
            continue
        stat = path.stat()
        if not existed:
            problems.append(f"created unexpected path: {path}")
        elif old_mtime != stat.st_mtime or old_size != stat.st_size:
            problems.append(f"updated unexpected path: {path}")
    return problems


def main() -> None:
    args = parse_args()
    start = datetime.now()
    log_path = ensure_log_dir()
    records: list[RunRecord] = []
    stages_run: list[str] = []
    stages_skipped: list[str] = []
    pollution_paths = [
        ROOT / "outputs" / "generations" / "comparison_main",
        ROOT / "outputs" / "token_logs" / "comparison_main",
    ]
    pollution_snapshot = snapshot_pollution(pollution_paths)
    status = "FAILED"
    failed_stage: str | None = None
    failed_command: list[str] | None = None
    failed_code: int | None = None
    try:
        line(f"log: {log_path}")
        line(f"start time: {start.isoformat()}")
        line("== preflight checks ==")
        check_static_targets()
        preflight_resources(args.min_free_ram_gb, args.min_free_vram_gb, args.allow_swap)
        if args.dry_run:
            line("dry-run: no commands will be executed")
        check_prompt_artifact(args.skip_build_prompts, args.dry_run, records)
        print_full_plan(not args.skip_tiny, not args.skip_smoke, not args.skip_distilgpt2, not args.skip_gpt2)
        if not args.yes and not args.dry_run:
            line("yes flag not provided; stopping after preflight/static checks")
            status = "CHECKS-ONLY"
            end = datetime.now()
            write_log(log_path, start, end, records, status)
            return
        if args.dry_run:
            status = "DRY-RUN"
            end = datetime.now()
            write_log(log_path, start, end, records, status)
            line("dry-run complete: no commands executed")
            return

        plan_stage, plan_cmd, plan_env = planned_commands()[0]
        run_command(plan_stage, plan_cmd, records, env=plan_env, dry_run=False)

        if not args.skip_tiny:
            preflight_resources(args.min_free_ram_gb, args.min_free_vram_gb, args.allow_swap)
            run_command("tiny: comparison", ["make", "comparison-tiny"], records, dry_run=args.dry_run)
            stages_run.append("tiny: comparison")
            if not args.dry_run:
                check_output_dir(ROOT / "outputs" / "generations" / "comparison_tiny")
                check_output_dir(ROOT / "outputs" / "token_logs" / "comparison_tiny")
            if subprocess.run(["make", "-n", "postprocess-tiny"], cwd=ROOT, text=True, capture_output=True).returncode == 0:
                run_command("tiny: postprocess", ["make", "postprocess-tiny"], records, dry_run=args.dry_run)
                stages_run.append("tiny: postprocess")
            else:
                stages_skipped.append("tiny: postprocess (missing target)")
        else:
            stages_skipped.append("tiny")

        if not args.skip_smoke:
            preflight_resources(args.min_free_ram_gb, args.min_free_vram_gb, args.allow_swap)
            run_command("smoke: comparison", ["make", "comparison-smoke"], records, dry_run=args.dry_run)
            stages_run.append("smoke: comparison")
            if not args.dry_run:
                check_output_dir(ROOT / "outputs" / "generations" / "comparison_smoke")
                check_output_dir(ROOT / "outputs" / "token_logs" / "comparison_smoke")
            run_command("smoke: postprocess", ["make", "postprocess-smoke"], records, dry_run=args.dry_run)
            stages_run.append("smoke: postprocess")
            if not args.dry_run:
                check_file(ROOT / "results" / "summary" / "comparison_smoke_metrics.csv")
        else:
            stages_skipped.append("smoke")

        if not args.skip_distilgpt2:
            preflight_resources(args.min_free_ram_gb, args.min_free_vram_gb, args.allow_swap)
            run_command("distilgpt2: comparison", ["make", "comparison-distilgpt2-main"], records, dry_run=args.dry_run)
            stages_run.append("distilgpt2: comparison")
            if not args.dry_run:
                check_output_dir(ROOT / "outputs" / "generations" / "comparison_distilgpt2_writingprompts")
                check_output_dir(ROOT / "outputs" / "token_logs" / "comparison_distilgpt2_writingprompts")
            run_command("distilgpt2: postprocess", ["make", "postprocess-distilgpt2-main"], records, dry_run=args.dry_run)
            stages_run.append("distilgpt2: postprocess")
            if not args.dry_run:
                check_file(ROOT / "results" / "summary" / "comparison_distilgpt2_writingprompts_metrics.csv")
        else:
            stages_skipped.append("distilgpt2")

        if not args.skip_gpt2:
            preflight_resources(args.min_free_ram_gb, args.min_free_vram_gb, args.allow_swap)
            run_command("gpt2: comparison", ["make", "comparison-main"], records, dry_run=args.dry_run)
            stages_run.append("gpt2: comparison")
            if not args.dry_run:
                check_output_dir(ROOT / "outputs" / "generations" / "comparison_gpt2_writingprompts")
                check_output_dir(ROOT / "outputs" / "token_logs" / "comparison_gpt2_writingprompts")
            run_command("gpt2: postprocess", ["make", "postprocess-main"], records, dry_run=args.dry_run)
            stages_run.append("gpt2: postprocess")
            if not args.dry_run:
                check_file(ROOT / "results" / "summary" / "comparison_gpt2_writingprompts_metrics.csv")
        else:
            stages_skipped.append("gpt2")

        pollution_problems = detect_pollution(pollution_snapshot, pollution_paths)
        if pollution_problems:
            raise AutoRunnerError("; ".join(pollution_problems))

        status = "COMPLETE"
        end = datetime.now()
        write_log(log_path, start, end, records, status)
        line("\nAUTO RUN COMPLETE")
        line(f"stages run: {', '.join(stages_run) if stages_run else 'none'}")
        line(f"stages skipped: {', '.join(stages_skipped) if stages_skipped else 'none'}")
        line(f"log path: {log_path}")
        line("key output directories:")
        line("  outputs/generations/comparison_smoke")
        line("  outputs/token_logs/comparison_smoke")
        line("  outputs/generations/comparison_distilgpt2_writingprompts")
        line("  outputs/token_logs/comparison_distilgpt2_writingprompts")
        line("  outputs/generations/comparison_gpt2_writingprompts")
        line("  outputs/token_logs/comparison_gpt2_writingprompts")
        line("key summary files:")
        line("  results/summary/comparison_smoke_metrics.csv")
        line("  results/summary/comparison_distilgpt2_writingprompts_metrics.csv")
        line("  results/summary/comparison_gpt2_writingprompts_metrics.csv")
        line("next recommended analysis commands:")
        line("  make export-smoke-table")
        line("  PYTHONPATH=src .venv/bin/python scripts/build_results_table.py --help")
        return
    except subprocess.CalledProcessError as exc:
        failed_stage = records[-1].stage if records else "unknown"
        failed_command = exc.cmd if isinstance(exc.cmd, list) else [str(exc.cmd)]
        failed_code = exc.returncode
        status = "FAILED"
        end = datetime.now()
        write_log(log_path, start, end, records, status, failed_stage, failed_command, failed_code)
        line("\nAUTO RUN FAILED")
        line(f"failed stage: {failed_stage}")
        line(f"failed command: {fmt_cmd(failed_command)}")
        line(f"exit code: {failed_code}")
        line(f"log path: {log_path}")
        line("recommended next check: inspect the log and rerun only after fixing the failure")
        raise SystemExit(failed_code)
    except Exception as exc:
        status = "FAILED"
        end = datetime.now()
        write_log(log_path, start, end, records, status, failed_stage, failed_command, failed_code)
        line("\nAUTO RUN FAILED")
        line(f"failed stage: {failed_stage or 'preflight'}")
        line(f"failed command: {fmt_cmd(failed_command or []) if failed_command else 'n/a'}")
        line("exit code: n/a")
        line(f"log path: {log_path}")
        line(f"recommended next check: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
