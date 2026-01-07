#!/usr/bin/env python3
"""
Preprocess Kaggle NYT RE dataset into SpanBERT/BERT relation classification format.

Input:
  Data/NYT RE/{train,valid,test}.json  (JSONL)

Output:
  Neural/RE/processed/nyt_re_{split}.jsonl
  Neural/RE/processed/label2id.json
  Neural/RE/processed/id2label.json

Each output row:
{
  "text": "... [E1] head [/E1] ... [E2] tail [/E2] ...",
  "relation": "/people/person/nationality",
  "label_id": 12,
  "head": "Bobby Fischer",
  "tail": "Iceland",
  "meta": {...}
}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import Counter, defaultdict


ROOT = Path(__file__).resolve().parents[2]
IN_DIR = ROOT / "Data" / "NYT RE"
OUT_DIR = Path(__file__).resolve().parent / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "valid", "test"]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error in {path.name} line {line_no}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"{path.name} line {line_no} is not a JSON object")
            records.append(obj)
    return records


def mark_entities(sentence: str, head: str, tail: str) -> str | None:
    """
    Insert entity markers around the first occurrence of head and tail strings.
    If either entity string is not found, return None.

    Note: This dataset uses mention texts (em1Text/em2Text) but does not provide character offsets.
    We use string matching for a baseline; later we can improve alignment.
    """
    h_idx = sentence.find(head)
    t_idx = sentence.find(tail)

    if h_idx == -1 or t_idx == -1:
        return None

    # If head and tail overlap (rare), skip
    h_end = h_idx + len(head)
    t_end = t_idx + len(tail)
    if not (h_end <= t_idx or t_end <= h_idx):
        return None

    # Insert markers in reverse order so indices stay valid
    if h_idx < t_idx:
        s = sentence
        s = s[:t_idx] + f"[E2] {tail} [/E2]" + s[t_end:]
        s = s[:h_idx] + f"[E1] {head} [/E1]" + s[h_end:]
        return s
    else:
        s = sentence
        s = s[:h_idx] + f"[E1] {head} [/E1]" + s[h_end:]
        s = s[:t_idx] + f"[E2] {tail} [/E2]" + s[t_end:]
        return s


def build_label_maps(train_records: List[Dict[str, Any]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    labels = set()
    for r in train_records:
        for rm in r.get("relationMentions", []):
            if isinstance(rm, dict) and "label" in rm:
                labels.add(rm["label"])
    labels = sorted(labels)
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    return label2id, id2label


def preprocess_split(
    split: str,
    records: List[Dict[str, Any]],
    label2id: Dict[str, int],
) -> Tuple[List[Dict[str, Any]], Counter]:
    out_rows: List[Dict[str, Any]] = []
    stats = Counter()

    for r in records:
        sent = r.get("sentText", "")
        if not isinstance(sent, str) or not sent.strip():
            stats["skip_empty_sent"] += 1
            continue

        rel_mentions = r.get("relationMentions", [])
        if not isinstance(rel_mentions, list) or not rel_mentions:
            stats["skip_no_relations"] += 1
            continue

        for rm in rel_mentions:
            if not isinstance(rm, dict):
                stats["skip_bad_relation_mention"] += 1
                continue

            head = rm.get("em1Text")
            tail = rm.get("em2Text")
            rel = rm.get("label")

            if not (isinstance(head, str) and isinstance(tail, str) and isinstance(rel, str)):
                stats["skip_missing_fields"] += 1
                continue

            if rel not in label2id:
                stats["skip_unknown_label"] += 1
                continue

            marked = mark_entities(sent, head, tail)
            if marked is None:
                stats["skip_entity_not_found_or_overlap"] += 1
                continue

            out_rows.append({
                "text": marked,
                "relation": rel,
                "label_id": label2id[rel],
                "head": head,
                "tail": tail,
                "meta": {
                    "articleId": r.get("articleId"),
                    "sentId": r.get("sentId"),
                }
            })
            stats["kept"] += 1

    stats["total_sentences"] = len(records)
    stats["total_relation_mentions"] = sum(len(r.get("relationMentions", [])) for r in records if isinstance(r.get("relationMentions", None), list))
    return out_rows, stats


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    print("Input dir:", IN_DIR)
    print("Output dir:", OUT_DIR)

    # Load train first to build label maps
    train_path = IN_DIR / "train.json"
    train_records = load_jsonl(train_path)

    label2id, id2label = build_label_maps(train_records)

    (OUT_DIR / "label2id.json").write_text(json.dumps(label2id, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "id2label.json").write_text(json.dumps(id2label, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Built label maps: num_labels={len(label2id)}")

    # Process each split
    for split in SPLITS:
        in_path = IN_DIR / f"{split}.json"
        records = load_jsonl(in_path)

        rows, stats = preprocess_split(split, records, label2id)

        out_path = OUT_DIR / f"nyt_re_{split}.jsonl"
        write_jsonl(out_path, rows)

        print("\n" + "=" * 90)
        print(f"SPLIT: {split}")
        for k, v in stats.most_common():
            print(f"{k:30s} {v}")
        print(f"Saved: {out_path}  (rows={len(rows)})")

if __name__ == "__main__":
    main()