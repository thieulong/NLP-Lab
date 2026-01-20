#!/usr/bin/env python3
"""
preprocess_nyt_re.py

Preprocess NYT relation extraction dataset into a JSONL format for
SpanBERT/BERT relation classification.

This script can optionally add a `no_relation` class by generating negative
entity pairs within each sentence that are not annotated by any relation.

Input files (defaults):
  Data/NYT RE/train.json
  Data/NYT RE/valid.json
  Data/NYT RE/test.json

Output files (defaults):
  Neural/RE/processed/nyt_re_train.jsonl
  Neural/RE/processed/nyt_re_valid.jsonl
  Neural/RE/processed/nyt_re_test.jsonl
  Neural/RE/processed/label2id.json
  Neural/RE/processed/id2label.json

Each output row looks like:
{
  "text": " ... [E1] head [/E1] ... [E2] tail [/E2] ... ",
  "relation": "/some/relation" or "no_relation",
  "label_id": 3,
  "head": "Entity 1 text",
  "tail": "Entity 2 text",
  "split": "train|valid|test",
  "meta": {...}
}

Notes:
- NYT RE data is expected to be a list[dict], each dict containing sentence text and
  relation mentions similar to the common NYT-FB format:
    ex["sentText"] or ex["text"]
    ex["relationMentions"] (list) with keys like em1Text/em2Text/label
    optionally ex["entityMentions"] (list) with key "text"
- If your JSON keys differ, adjust sentence_text(), extract_positive_mentions(),
  and extract_entities() below.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -------------------------
# IO helpers
# -------------------------

def read_json(path: Path) -> Any:
    """
    Read either:
      - standard JSON (single object/array)
      - JSONL (one JSON object per line)
    """
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    # Heuristic: JSONL usually starts with '{' and has multiple lines
    # Standard JSON array usually starts with '['
    if raw[0] == "[":
        return json.loads(raw)

    # Try normal JSON object first
    if raw[0] == "{":
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Fall back to JSONL
            pass

    # JSONL fallback
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {e}") from e
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -------------------------
# Dataset parsing helpers
# -------------------------

def sentence_text(ex: Dict[str, Any]) -> str:
    # Common keys: sentText (NYT-FB), text (some exports)
    if "sentText" in ex and isinstance(ex["sentText"], str):
        return ex["sentText"]
    if "text" in ex and isinstance(ex["text"], str):
        return ex["text"]
    return ""


def extract_positive_mentions(ex: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """
    Returns a list of (head_text, tail_text, relation_label) for this sentence.
    """
    out: List[Tuple[str, str, str]] = []

    rms = ex.get("relationMentions", None)
    if isinstance(rms, list):
        for rm in rms:
            if not isinstance(rm, dict):
                continue
            rel = rm.get("label") or rm.get("relation") or rm.get("rel")
            h = rm.get("em1Text") or rm.get("head") or rm.get("e1") or rm.get("arg1")
            t = rm.get("em2Text") or rm.get("tail") or rm.get("e2") or rm.get("arg2")
            if isinstance(rel, str) and isinstance(h, str) and isinstance(t, str):
                out.append((h.strip(), t.strip(), rel.strip()))
        return out

    # Fallback: some formats store one relation per example
    rel = ex.get("relation")
    h = ex.get("head")
    t = ex.get("tail")
    if isinstance(rel, str) and isinstance(h, str) and isinstance(t, str):
        out.append((h.strip(), t.strip(), rel.strip()))
    return out


def extract_entities(ex: Dict[str, Any]) -> List[str]:
    """
    Extract entity surface forms for negative sampling.

    Priority:
    1) entityMentions[].text if available
    2) unique entity strings from relationMentions (em1Text/em2Text)
    """
    ents: List[str] = []

    ems = ex.get("entityMentions", None)
    if isinstance(ems, list):
        for em in ems:
            if not isinstance(em, dict):
                continue
            t = em.get("text")
            if isinstance(t, str):
                ents.append(t.strip())

    if ents:
        # uniq, keep order
        seen = set()
        uniq = []
        for e in ents:
            if e and e not in seen:
                seen.add(e)
                uniq.append(e)
        return uniq

    # fallback from positives
    pos = extract_positive_mentions(ex)
    seen = set()
    for h, t, _ in pos:
        if h and h not in seen:
            seen.add(h)
            ents.append(h)
        if t and t not in seen:
            seen.add(t)
            ents.append(t)
    return ents


# -------------------------
# Marker insertion
# -------------------------

def _find_span_non_overlapping(text: str, needle: str, occupied: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    """
    Find a span (start,end) for needle in text that does not overlap occupied.
    Uses a simple left-to-right search.
    """
    if not needle:
        return None
    pattern = re.escape(needle)
    for m in re.finditer(pattern, text):
        s, e = m.start(), m.end()
        overlap = False
        for os, oe in occupied:
            if not (e <= os or s >= oe):
                overlap = True
                break
        if not overlap:
            return (s, e)
    return None


def mark_entities(text: str, head: str, tail: str) -> Optional[Tuple[str, Tuple[int, int], Tuple[int, int]]]:
    """
    Insert [E1] head [/E1] and [E2] tail [/E2] around first non-overlapping
    occurrences. Returns (marked_text, head_span, tail_span) where spans are
    offsets in original text.
    """
    occupied: List[Tuple[int, int]] = []
    h_span = _find_span_non_overlapping(text, head, occupied)
    if h_span is None:
        return None
    occupied.append(h_span)

    t_span = _find_span_non_overlapping(text, tail, occupied)
    if t_span is None:
        return None

    (hs, he) = h_span
    (ts, te) = t_span

    # Insert from end to keep indices stable
    out = text
    if hs < ts:
        # tail first (later in string)
        out = out[:te] + " [/E2]" + out[te:]
        out = out[:ts] + "[E2] " + out[ts:]
        out = out[:he] + " [/E1]" + out[he:]
        out = out[:hs] + "[E1] " + out[hs:]
    else:
        # head occurs later than tail
        out = out[:he] + " [/E1]" + out[he:]
        out = out[:hs] + "[E1] " + out[hs:]
        out = out[:te] + " [/E2]" + out[te:]
        out = out[:ts] + "[E2] " + out[ts:]

    return out, h_span, t_span


# -------------------------
# Label maps
# -------------------------

def build_label_maps(
    train_examples: List[Dict[str, Any]],
    add_no_relation: bool,
    no_relation_label: str,
    no_relation_id_first: bool,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build label2id/id2label from training split relation labels.
    If add_no_relation: include no_relation_label as well.

    If no_relation_id_first:
      - no_relation_label gets id 0
      - all other labels get shifted by +1
    """
    labels = set()
    for ex in train_examples:
        if not isinstance(ex, dict):
            continue
        for _, _, rel in extract_positive_mentions(ex):
            labels.add(rel)

    # Stable order
    sorted_labels = sorted(labels)

    label2id: Dict[str, int] = {}
    id2label: Dict[int, str] = {}

    if add_no_relation:
        if no_relation_id_first:
            label2id[no_relation_label] = 0
            id2label[0] = no_relation_label
            start = 1
        else:
            start = 0

        for i, lab in enumerate(sorted_labels, start=start):
            label2id[lab] = i
            id2label[i] = lab

        if not no_relation_id_first:
            # append no_relation at the end
            nr_id = len(label2id)
            label2id[no_relation_label] = nr_id
            id2label[nr_id] = no_relation_label
    else:
        for i, lab in enumerate(sorted_labels):
            label2id[lab] = i
            id2label[i] = lab

    return label2id, id2label


# -------------------------
# Example generation
# -------------------------

def generate_examples_for_sentence(
    ex: Dict[str, Any],
    label2id: Dict[str, int],
    split_name: str,
    add_no_relation: bool,
    no_relation_label: str,
    neg_ratio: float,
    max_negs_per_sentence: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """
    For one sentence example:
      - create positives for each relation mention
      - optionally create sampled negatives (no_relation)
    """
    sent = sentence_text(ex).strip()
    if not sent:
        return []

    positives = extract_positive_mentions(ex)
    rows: List[Dict[str, Any]] = []

    # Track which ordered pairs are positive (so we can avoid sampling them as negatives)
    pos_pairs = set()
    for h, t, rel in positives:
        if not h or not t or not rel:
            continue
        pos_pairs.add((h, t))
        marked = mark_entities(sent, h, t)
        if marked is None:
            continue
        marked_text, _, _ = marked

        if rel not in label2id:
            # If this happens, train label map was built from different data.
            # Skip to avoid crashes.
            continue

        rows.append(
            {
                "text": marked_text,
                "relation": rel,
                "label_id": int(label2id[rel]),
                "head": h,
                "tail": t,
                "split": split_name,
                "meta": ex.get("meta", {}),
            }
        )

    if not add_no_relation:
        return rows

    # Generate candidate negatives from entities in this sentence
    ents = extract_entities(ex)
    if len(ents) < 2:
        return rows

    # All ordered pairs (h,t), h!=t
    all_pairs = [(h, t) for h in ents for t in ents if h != t]
    # Remove positives (for any relation)
    neg_pairs = [(h, t) for (h, t) in all_pairs if (h, t) not in pos_pairs]

    if not neg_pairs:
        return rows

    # How many negatives to sample?
    # Approx: neg_ratio * num_positive_mentions for this sentence, capped.
    num_pos = max(1, len(rows))  # if marking failed for all positives, still allow some negs
    target_negs = int(round(neg_ratio * num_pos))
    target_negs = max(0, min(target_negs, max_negs_per_sentence))
    if target_negs <= 0:
        return rows

    sampled = rng.sample(neg_pairs, k=min(target_negs, len(neg_pairs)))

    for h, t in sampled:
        marked = mark_entities(sent, h, t)
        if marked is None:
            continue
        marked_text, _, _ = marked
        if no_relation_label not in label2id:
            # should not happen if label2id built correctly
            continue
        rows.append(
            {
                "text": marked_text,
                "relation": no_relation_label,
                "label_id": int(label2id[no_relation_label]),
                "head": h,
                "tail": t,
                "split": split_name,
                "meta": ex.get("meta", {}),
            }
        )

    return rows


def preprocess_split(
    split_name: str,
    in_path: Path,
    out_path: Path,
    label2id: Dict[str, int],
    add_no_relation: bool,
    no_relation_label: str,
    neg_ratio: float,
    max_negs_per_sentence: int,
    seed: int,
) -> Dict[str, Any]:
    data = read_json(in_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {in_path}, got {type(data)}")

    rng = random.Random(seed)

    rows: List[Dict[str, Any]] = []
    rel_counter = Counter()

    for ex in data:
        if not isinstance(ex, dict):
            continue
        ex_rows = generate_examples_for_sentence(
            ex,
            label2id=label2id,
            split_name=split_name,
            add_no_relation=add_no_relation,
            no_relation_label=no_relation_label,
            neg_ratio=neg_ratio,
            max_negs_per_sentence=max_negs_per_sentence,
            rng=rng,
        )
        for r in ex_rows:
            rel_counter[r["relation"]] += 1
        rows.extend(ex_rows)

    write_jsonl(out_path, rows)

    stats = {
        "split": split_name,
        "input": str(in_path),
        "output": str(out_path),
        "rows": len(rows),
        "label_counts": dict(rel_counter),
    }
    return stats


# -------------------------
# CLI
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", type=str, default="Data/NYT RE")
    ap.add_argument("--out_dir", type=str, default="Neural/RE/processed")

    ap.add_argument("--add_no_relation", action="store_true", help="Generate negative pairs and add 'no_relation' label")
    ap.add_argument("--no_relation_label", type=str, default="no_relation")
    ap.add_argument(
        "--no_relation_id_first",
        action="store_true",
        help="Assign label_id=0 to no_relation and shift other labels by +1",
    )
    ap.add_argument("--neg_ratio", type=float, default=1.0, help="Negatives per positive mention (approx).")
    ap.add_argument("--max_negs_per_sentence", type=int, default=6)
    ap.add_argument("--seed", type=int, default=13)

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
        raise ValueError(f"Expected list in {train_path}, got {type(train_data)}")

    label2id, id2label = build_label_maps(
        train_examples=train_data,
        add_no_relation=bool(args.add_no_relation),
        no_relation_label=args.no_relation_label,
        no_relation_id_first=bool(args.no_relation_id_first),
    )

    write_json(out_dir / "label2id.json", label2id)
    # id2label keys must be strings for HF compatibility in some cases
    id2label_str_keys = {str(k): v for k, v in id2label.items()}
    write_json(out_dir / "id2label.json", id2label_str_keys)

    print(f"Labels: {len(label2id)}")
    if args.add_no_relation:
        print(f"Added no_relation class: '{args.no_relation_label}' (id={label2id[args.no_relation_label]})")

    stats_all: List[Dict[str, Any]] = []

    stats_all.append(
        preprocess_split(
            "train",
            in_path=train_path,
            out_path=out_dir / "nyt_re_train.jsonl",
            label2id=label2id,
            add_no_relation=bool(args.add_no_relation),
            no_relation_label=args.no_relation_label,
            neg_ratio=float(args.neg_ratio),
            max_negs_per_sentence=int(args.max_negs_per_sentence),
            seed=int(args.seed),
        )
    )

    stats_all.append(
        preprocess_split(
            "valid",
            in_path=valid_path,
            out_path=out_dir / "nyt_re_valid.jsonl",
            label2id=label2id,
            add_no_relation=bool(args.add_no_relation),
            no_relation_label=args.no_relation_label,
            neg_ratio=float(args.neg_ratio),
            max_negs_per_sentence=int(args.max_negs_per_sentence),
            seed=int(args.seed) + 1,
        )
    )

    stats_all.append(
        preprocess_split(
            "test",
            in_path=test_path,
            out_path=out_dir / "nyt_re_test.jsonl",
            label2id=label2id,
            add_no_relation=bool(args.add_no_relation),
            no_relation_label=args.no_relation_label,
            neg_ratio=float(args.neg_ratio),
            max_negs_per_sentence=int(args.max_negs_per_sentence),
            seed=int(args.seed) + 2,
        )
    )

    print("\nSplit stats:")
    for st in stats_all:
        print(f"- {st['split']}: rows={st['rows']}")

    if args.add_no_relation:
        print("\nNext steps:")
        print("1) Retrain the SpanBERT classifier with the new processed JSONL.")
        print("2) Ensure train_spanbert_nyt_re.py reads label2id/id2label from out_dir.")


if __name__ == "__main__":
    main()