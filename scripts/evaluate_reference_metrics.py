from __future__ import annotations

import argparse
from pathlib import Path

import evaluate

from flatness_decoding.io_utils import read_jsonl, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Directory containing generation JSONL files.")
    parser.add_argument("--reference-prompts", required=True, help="JSONL containing id/prompt/reference fields.")
    parser.add_argument("--reference-field", default="reference")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--output", default="results/summary/reference_metrics.csv")
    parser.add_argument("--metrics", nargs="+", default=["bleu", "rouge", "bertscore"])
    parser.add_argument("--bertscore-model", default="distilbert-base-uncased")
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
        "seed",
    ]
    return {key: first.get(key) for key in keys}


def main() -> None:
    args = parse_args()
    refs = load_references(args.reference_prompts, args.id_field, args.reference_field)
    metric_objs = {name: evaluate.load(name) for name in args.metrics}
    rows_out: list[dict] = []

    for path in sorted(Path(args.input_dir).glob("*.jsonl")):
        rows = read_jsonl(path)
        preds: list[str] = []
        references: list[str] = []
        for row in rows:
            prompt_id = str(row.get("prompt_id", ""))
            if prompt_id not in refs:
                continue
            preds.append(str(row.get("text", "")))
            references.append(refs[prompt_id])

        result = {"file": str(path), **metadata(rows), "num_scored": len(preds)}
        if preds:
            if "bleu" in metric_objs:
                bleu = metric_objs["bleu"].compute(predictions=preds, references=[[ref] for ref in references])
                result["bleu"] = bleu.get("bleu")
            if "rouge" in metric_objs:
                rouge = metric_objs["rouge"].compute(predictions=preds, references=references)
                for key, value in rouge.items():
                    result[key] = value
            if "bertscore" in metric_objs:
                bert = metric_objs["bertscore"].compute(
                    predictions=preds,
                    references=references,
                    model_type=args.bertscore_model,
                    lang="en",
                )
                result["bertscore_precision"] = sum(bert["precision"]) / len(bert["precision"])
                result["bertscore_recall"] = sum(bert["recall"]) / len(bert["recall"])
                result["bertscore_f1"] = sum(bert["f1"]) / len(bert["f1"])
        rows_out.append(result)

    write_csv(args.output, rows_out)
    print(f"wrote {args.output} with {len(rows_out)} rows")


if __name__ == "__main__":
    main()
