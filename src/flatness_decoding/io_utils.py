from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable
import os
import tempfile


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    p = ensure_parent(path)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl_atomic(path: str | Path, rows: Iterable[dict]) -> None:
    """Write JSONL atomically to avoid losing progress on interruption."""
    p = ensure_parent(path)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(p.parent), delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            for row in rows:
                tmp.write(json.dumps(row, ensure_ascii=False) + "\n")
            tmp.flush()
            os.fsync(tmp.fileno())
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
    os.replace(tmp_path, p)


def append_jsonl(path: str | Path, row: dict) -> None:
    p = ensure_parent(path)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    rows: list[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: str | Path, rows: list[dict]) -> None:
    p = ensure_parent(path)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
