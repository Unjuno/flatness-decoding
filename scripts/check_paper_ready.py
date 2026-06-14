from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manuscript", default="paper/manuscript.md")
    parser.add_argument("--fail", action="store_true")
    return parser.parse_args()


def run(cmd: list[str]) -> int:
    print("+ " + " ".join(cmd))
    completed = subprocess.run(cmd)
    return completed.returncode


def main() -> None:
    args = parse_args()
    checks = []

    manuscript = Path(args.manuscript)
    if not manuscript.exists():
        print(f"missing manuscript: {manuscript}")
        raise SystemExit(1 if args.fail else 0)

    checks.append(
        run(
            [
                sys.executable,
                "scripts/check_manuscript_placeholders.py",
                "--path",
                str(manuscript),
                *( ["--fail"] if args.fail else [] ),
            ]
        )
    )

    required_files = [
        "paper/README.md",
        "paper/references.bib",
        "paper/citation_notes.md",
        "paper/thesis_blueprint_final.md",
        "paper/quality_evaluation_plan.md",
        "paper/motivation_biodiversity_analogy.md",
        "paper/figures/method_positioning.mmd",
        "paper/figures/flatness_n_algorithm.mmd",
        "notes/evaluation_matrix.md",
    ]
    missing = [path for path in required_files if not Path(path).exists()]
    if missing:
        print("missing paper support files:")
        for path in missing:
            print(f"- {path}")
        checks.append(1)
    else:
        print("paper support files present")
        checks.append(0)

    if args.fail and any(code != 0 for code in checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
