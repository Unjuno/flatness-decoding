from __future__ import annotations

import argparse
from pathlib import Path

MARKERS = [
    "[TODO",
    "[RESULT TBD",
    "[VERIFY",
    "[CITATION NEEDED",
    "[FIGURE TBD",
    "[TABLE TBD",
    "[ALGORITHM TBD",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="paper/manuscript.md")
    parser.add_argument("--fail", action="store_true", help="Exit nonzero if unresolved markers remain.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.path)
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    found: list[tuple[int, str, str]] = []
    for line_no, line in enumerate(lines, start=1):
        for marker in MARKERS:
            if marker in line:
                found.append((line_no, marker, line.strip()))
                break

    if not found:
        print(f"No unresolved markers found in {path}")
        return

    print(f"Found {len(found)} unresolved markers in {path}:")
    for line_no, marker, line in found:
        print(f"{line_no}: {marker} :: {line}")

    if args.fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
