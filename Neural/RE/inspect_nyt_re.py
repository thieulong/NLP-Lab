#!/usr/bin/env python3
"""
Inspect Kaggle NYT RE dataset schema and stats.

Handles:
- JSON (single object / list)
- JSONL (one JSON object per line)

Run:
  python Neural/RE/inspect_nyt_re.py
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[2]  # project root
DATA_DIR = ROOT / "Data" / "NYT RE"
FILES = {
    "train": DATA_DIR / "train.json",
    "valid": DATA_DIR / "valid.json",
    "test":  DATA_DIR / "test.json",
}


def load_records(path: Path, max_records: int | None = None) -> List[Dict[str, Any]]:
    """
    Try to load:
      1) standard JSON (list/dict)
      2) JSON Lines (JSONL): one object per line
    Returns list[dict] records.
    """
    text = path.read_text(encoding="utf-8", errors="replace")

    # 1) Try normal JSON first
    try:
        obj = json.loads(text)
        return normalize_to_records(obj, max_records=max_records)
    except json.JSONDecodeError:
        pass  # fall back to JSONL

    # 2) JSONL fallback
    records: List[Dict[str, Any]] = []
    for i, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            # show problematic line for debugging, then raise
            raise ValueError(
                f"Failed parsing JSONL line {i+1} in {path.name}. "
                f"Line starts with: {line[:120]!r}"
            )
        if not isinstance(rec, dict):
            raise ValueError(f"JSONL line {i+1} is not an object/dict.")
        records.append(rec)
        if max_records is not None and len(records) >= max_records:
            break
    return records


def normalize_to_records(obj: Any, max_records: int | None = None) -> List[Dict[str, Any]]:
    """
    Convert common JSON formats into list[dict].
    """
    if isinstance(obj, list):
        records = [r for r in obj if isinstance(r, dict)]
        if len(records) != len(obj):
            raise ValueError("JSON list contains non-dict items.")
        return records[:max_records] if max_records else records

    if isinstance(obj, dict):
        # common containers
        for k in ["data", "examples", "records", "items", "sentences"]:
            if k in obj and isinstance(obj[k], list) and obj[k] and isinstance(obj[k][0], dict):
                records = obj[k]
                return records[:max_records] if max_records else records

        # dict-of-records
        vals = list(obj.values())
        if vals and isinstance(vals[0], dict):
            records = vals  # type: ignore
            return records[:max_records] if max_records else records

    raise ValueError("Could not normalize JSON into list[dict] records.")


def summarize_split(name: str, records: List[Dict[str, Any]], n_preview: int = 2) -> None:
    print("\n" + "=" * 90)
    print(f"SPLIT: {name}  |  num_records={len(records)}")

    key_counter = Counter()
    for r in records[: min(len(records), 2000)]:
        key_counter.update(r.keys())

    print("\nTop keys (from first up to 2000 records):")
    for k, c in key_counter.most_common(40):
        print(f"  {k}: {c}")

    print("\nSample records (truncated):")
    for i in range(min(n_preview, len(records))):
        r = records[i]
        r2 = {}
        for k, v in r.items():
            if isinstance(v, str) and len(v) > 200:
                r2[k] = v[:200] + "...(truncated)"
            else:
                r2[k] = v
        print(f"\n--- record[{i}] ---")
        print(json.dumps(r2, ensure_ascii=False, indent=2))

    labelish = [k for k in key_counter.keys() if any(t in k.lower() for t in ["label", "relation", "rel", "predicate"])]
    if labelish:
        print("\nPossible label/relation keys:", sorted(labelish))


def main() -> None:
    print("Project root:", ROOT)
    print("Data dir:", DATA_DIR)

    for split, path in FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        # load all records for counts, but only preview a few in printing
        records = load_records(path, max_records=None)
        summarize_split(split, records, n_preview=2)

    print("\n" + "=" * 90)

if __name__ == "__main__":
    main()