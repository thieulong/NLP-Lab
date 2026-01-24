#!/usr/bin/env python3
"""
preprocess_nyt_re.py (POSITIVES ONLY)

Converts NYT RE JSON (train/valid/test) into JSONL for SpanBERT-style relation classification.

- Uses ONLY the gold relationMentions present in the dataset.
- DOES NOT generate negative pairs.
- DOES NOT add a no_relation class.

Output files (in out_dir):
  - nyt_re_train.jsonl
  - nyt_re_valid.jsonl
  - nyt_re_test.jsonl
  - label2id.json
  - id2label.json

Each JSONL row:
  {
    "text": "[E1] ... [/E1] ... [E2] ... [/E2]",
    "relation": "/location/location/contains",
    "label_id": 11,
    "head": "Annandale-on-Hudson",
    "tail": "Bard College",
    "split": "train",
    "meta": {...}
  }
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -------------------------
# IO utils
# -------------------------

def read_json(path: Path) -> Any:
    # NYT RE files in your repo are JSONL (one JSON per line), even if named .json.
    # So we must read line-by-line safely.
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -------------------------
# NYT field helpers
# -------------------------

def sentence_text(ex: Dict[str, Any]) -> str:
    # NYT uses "sentText"
    return (ex.get("sentText") or "").strip()


def extract_positive_mentions(ex: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """
    Returns list of (em1Text, em2Text, label) from relationMentions
    """
    out: List[Tuple[str, str, str]] = []
    rms = ex.get("relationMentions") or []
    if not isinstance(rms, list):
        return out
    for rm in rms:
        if not isinstance(rm, dict):
            continue
        h = (rm.get("em1Text") or "").strip()
        t = (rm.get("em2Text") or "").strip()
        rel = (rm.get("label") or "").strip()
        if h and t and rel:
            out.append((h, t, rel))
    return out


def find_first_span(text: str, needle: str) -> Optional[Tuple[int, int]]:
    """
    Find first occurrence span of 'needle' in 'text'. Returns (start,end) or None.
    """
    if not needle:
        return None
    idx = text.find(needle)
    if idx < 0:
        return None
    return (idx, idx + len(needle))


def mark_entities(sent: str, head: str, tail: str) -> Optional[str]:
    """
    Insert [E1]...[/E1] and [E2]...[/E2] around the first mention of head/tail.
    If either mention not found, return None.
    """
    h_span = find_first_span(sent, head)
    t_span = find_first_span(sent, tail)
    if h_span is None or t_span is None:
        return None

    hs, he = h_span
    ts, te = t_span

    # Ensure deterministic marker placement (avoid shifting indexes incorrectly).
    # Place markers from rightmost span first.
    if hs <= ts:
        # head first, tail second
        out = sent
        out = out[:te] + " [/E2]" + out[te:]
        out = out[:ts] + "[E2] " + out[ts:]
        out = out[:he] + " [/E1]" + out[he:]
        out = out[:hs] + "[E1] " + out[hs:]
    else:
        # tail first, head second
        out = sent
        out = out[:he] + " [/E1]" + out[he:]
        out = out[:hs] + "[E1] " + out[hs:]
        out = out[:te] + " [/E2]" + out[te:]
        out = out[:ts] + "[E2] " + out[ts:]

    return out


# -------------------------
# Label maps (positives only)
# -------------------------

def build_label_maps(train_examples: List[Dict[str, Any]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    labels = set()
    for ex in train_examples:
        if not isinstance(ex, dict):
            continue
        for _, _, rel in extract_positive_mentions(ex):
            labels.add(rel)

    sorted_labels = sorted(labels)  # stable
    label2id = {lab: i for i, lab in enumerate(sorted_labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    return label2id, id2label


# -------------------------
# Split preprocessing (positives only)
# -------------------------

def preprocess_split(
    split_name: str,
    in_path: Path,
    out_path: Path,
    label2id: Dict[str, int],
) -> Dict[str, Any]:
    data = read_json(in_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected list from JSONL in {in_path}")

    rows: List[Dict[str, Any]] = []
    rel_counter = Counter()

    for ex in data:
        if not isinstance(ex, dict):
            continue
        sent = sentence_text(ex)
        if not sent:
            continue

        positives = extract_positive_mentions(ex)
        for h, t, rel in positives:
            if rel not in label2id:
                continue
            marked = mark_entities(sent, h, t)
            if marked is None:
                continue

            row = {
                "text": marked,
                "relation": rel,
                "label_id": int(label2id[rel]),
                "head": h,
                "tail": t,
                "split": split_name,
                "meta": {
                    "sentId": ex.get("sentId"),
                    "articleId": ex.get("articleId"),
                },
            }
            rows.append(row)
            rel_counter[rel] += 1

    write_jsonl(out_path, rows)
    return {
        "split": split_name,
        "input": str(in_path),
        "output": str(out_path),
        "rows": len(rows),
        "label_counts": dict(rel_counter),
    }


# -------------------------
# CLI
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="Data/NYT RE")
    ap.add_argument("--out_dir", type=str, default="Neural/RE/processed_no_norel")

    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.json"
    valid_path = data_dir / "valid.json"
    test_path = data_dir / "test.json"

    for p in (train_path, valid_path, test_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing dataset file: {p}")

    train_data = read_json(train_path)
    if not isinstance(train_data, list):
        raise ValueError(f"Expected JSONL list from {train_path}")

    label2id, id2label = build_label_maps(train_data)

    write_json(out_dir / "label2id.json", label2id)
    # write id2label with string keys (HF-friendly)
    write_json(out_dir / "id2label.json", {str(k): v for k, v in id2label.items()})

    print(f"Labels (positives only): {len(label2id)}")

    stats_all = []
    stats_all.append(preprocess_split("train", train_path, out_dir / "nyt_re_train.jsonl", label2id))
    stats_all.append(preprocess_split("valid", valid_path, out_dir / "nyt_re_valid.jsonl", label2id))
    stats_all.append(preprocess_split("test",  test_path,  out_dir / "nyt_re_test.jsonl",  label2id))

    print("\nSplit stats:")
    for s in stats_all:
        print(f"- {s['split']}: rows={s['rows']}")

    write_json(out_dir / "preprocess_stats.json", stats_all)
    print(f"\nWrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()