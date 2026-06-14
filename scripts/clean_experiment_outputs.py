from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, help="Experiment name, e.g. comparison_smoke")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually delete files. Without this flag, only print what would be removed.",
    )
    return parser.parse_args()


def candidate_paths(experiment: str) -> list[Path]:
    return [
        Path("outputs/generations") / experiment,
        Path("outputs/token_logs") / experiment,
        Path("results/figures") / experiment,
        Path("results/samples") / f"{experiment}_samples.md",
        Path("paper/tables") / f"{experiment}_table.md",
        *Path("results/summary").glob(f"{experiment}*.csv"),
        *Path("results/summary").glob(f"{experiment}*.md"),
    ]


def remove_path(path: Path, yes: bool) -> None:
    if not path.exists():
        return
    if not yes:
        print(f"would remove: {path}")
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    print(f"removed: {path}")


def main() -> None:
    args = parse_args()
    paths = candidate_paths(args.experiment)
    for path in paths:
        remove_path(path, args.yes)
    if not args.yes:
        print("dry run only. Re-run with --yes to delete these paths.")


if __name__ == "__main__":
    main()
