from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from flatness_decoding.io_utils import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", default="results/samples/qualitative_samples.md")
    parser.add_argument("--prompt-limit", type=int, default=5)
    parser.add_argument("--max-chars", type=int, default=1200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grouped: dict[str, list[dict]] = defaultdict(list)

    for path in sorted(Path(args.input_dir).glob("*.jsonl")):
        for row in read_jsonl(path):
            prompt_id = str(row.get("prompt_id", "unknown"))
            grouped[prompt_id].append(row)

    lines: list[str] = []
    lines.append("# Qualitative sample comparison\n")
    lines.append("This file groups outputs by prompt ID so that methods can be compared side-by-side.\n")

    for idx, (prompt_id, rows) in enumerate(sorted(grouped.items())):
        if idx >= args.prompt_limit:
            break
        rows = sorted(rows, key=lambda r: (str(r.get("method")), str(r.get("n_trials"))))
        prompt = rows[0].get("prompt", "") if rows else ""
        lines.append(f"## Prompt {prompt_id}\n")
        lines.append(f"**Prompt:** {prompt}\n")
        for row in rows:
            method = row.get("method")
            n_trials = row.get("n_trials")
            seed = row.get("seed")
            title = f"method={method}, N={n_trials}, seed={seed}"
            text = str(row.get("text", ""))[: args.max_chars]
            lines.append(f"### {title}\n")
            lines.append("```text")
            lines.append(text)
            lines.append("```\n")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
