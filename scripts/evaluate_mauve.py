from __future__ import annotations

import argparse
from pathlib import Path

from flatness_decoding.io_utils import read_jsonl, write_csv
from flatness_decoding.metrics import analysis_text_from_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Directory containing generation JSONL files.")
    parser.add_argument("--reference-prompts", required=True, help="JSONL containing reference human texts.")
    parser.add_argument("--reference-field", default="reference")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--device-id", type=int, default=-1, help="-1 for CPU, GPU id otherwise.")
    parser.add_argument("--max-texts", type=int, default=1000)
    parser.add_argument("--output", default="results/summary/mauve_metrics.csv")
    return parser.parse_args()


def load_references(path: str | Path, id_field: str, reference_field: str) -> dict[str, str]:
    rows = read_jsonl(path)
    refs: dict[str, str] = {}
    for row in rows:
        if id_field in row and reference_field in row:
            refs[str(row[id_field])] = str(row[reference_field])
    return refs


def metadata(rows: list[dict]) -> dict:
    first = rows[0] if rows else {}
    keys = [
        "experiment_name",
        "model",
        "method",
        "method_family",
        "n_trials",
        "proposal_temperature",
        "baseline_temperature",
        "top_p",
        "top_k",
        "num_beams",
        "seed",
    ]
    return {key: first.get(key) for key in keys}


def main() -> None:
    try:
        import mauve
    except ImportError as exc:
        raise SystemExit("Install optional dependency first: pip install -r requirements-eval.txt") from exc

    args = parse_args()
    refs = load_references(args.reference_prompts, args.id_field, args.reference_field)
    rows_out: list[dict] = []

    for path in sorted(Path(args.input_dir).glob("*.jsonl")):
        rows = read_jsonl(path)
        generated: list[str] = []
        reference: list[str] = []
        for row in rows:
            prompt_id = str(row.get("prompt_id", ""))
            if prompt_id not in refs:
                continue
            generated.append(analysis_text_from_row(row))
            reference.append(refs[prompt_id])
            if len(generated) >= args.max_texts:
                break

        result = {
            "file": str(path),
            **metadata(rows),
            "scored_text": "continuation",
            "num_scored": len(generated),
        }
        if generated:
            out = mauve.compute_mauve(
                p_text=reference,
                q_text=generated,
                device_id=args.device_id,
                max_text_length=256,
                verbose=False,
            )
            result["mauve"] = float(out.mauve)
        rows_out.append(result)

    write_csv(args.output, rows_out)
    print(f"wrote {args.output} with {len(rows_out)} rows")


if __name__ == "__main__":
    main()
