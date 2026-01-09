#!/usr/bin/env python3
"""
Predict relations with a fine-tuned SpanBERT (sequence classification) model.

Works in two modes:
1) Test-file mode (recommended): reads processed NYT JSONL rows that already contain entity markers.
2) Custom-text mode: you provide a sentence + head/tail strings; we will insert markers and predict.

Expected training artifacts (from your earlier scripts):
- Model dir:   Neural/RE/models/spanbert_nyt_re
- Processed:   Neural/RE/processed/nyt_re_test.jsonl
- Label map:   Neural/RE/processed/label2id.json (optional if model config already has id2label)
"""

import argparse
import json
import os
import random
from typing import Dict, List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Markers used in preprocessing (match your preprocess script)
E1_START = "[E1]"
E1_END   = "[/E1]"
E2_START = "[E2]"
E2_END   = "[/E2]"


def read_jsonl(path: str, limit: Optional[int] = None) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def load_label_maps(model, processed_dir: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Prefer model.config.id2label/label2id (saved with the model).
    Fallback to processed/label2id.json.
    """
    label2id = getattr(model.config, "label2id", None)
    id2label = getattr(model.config, "id2label", None)

    # HF sometimes stores keys as strings for id2label
    if isinstance(id2label, dict) and len(id2label) > 0:
        id2label_int = {}
        for k, v in id2label.items():
            try:
                id2label_int[int(k)] = v
            except Exception:
                pass
        if id2label_int:
            id2label = id2label_int

    if isinstance(label2id, dict) and isinstance(id2label, dict) and len(label2id) > 0 and len(id2label) > 0:
        return label2id, id2label

    # Fallback
    label2id_path = os.path.join(processed_dir, "label2id.json")
    if not os.path.exists(label2id_path):
        raise FileNotFoundError(
            "Could not find label maps in model config, and label2id.json not found at: "
            f"{label2id_path}"
        )

    with open(label2id_path, "r", encoding="utf-8") as f:
        label2id = json.load(f)
    id2label = {int(v): k for k, v in label2id.items()}
    return label2id, id2label


def find_no_relation_labels(label2id: Dict[str, int]) -> List[str]:
    """
    NYT-style datasets often include NA/no_relation variants. We try to detect them.
    """
    candidates = []
    for lab in label2id.keys():
        low = lab.lower()
        if low in {"na", "no_relation", "n/a", "none"} or "no_relation" in low or low.endswith("/na"):
            candidates.append(lab)
    return candidates


def predict_one(
    text: str,
    tokenizer,
    model,
    id2label: Dict[int, str],
    device: torch.device,
    max_length: int = 256,
) -> Tuple[str, float, List[Tuple[str, float]]]:
    """
    Returns:
      - best_label
      - best_confidence
      - topk list [(label, prob)]
    """
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc)
        logits = out.logits[0]
        probs = torch.softmax(logits, dim=-1)

    best_id = int(torch.argmax(probs).item())
    best_prob = float(probs[best_id].item())
    best_label = id2label.get(best_id, str(best_id))

    # Top-5
    topk = min(5, probs.shape[-1])
    vals, idxs = torch.topk(probs, k=topk)
    top_list = [(id2label.get(int(i.item()), str(int(i.item()))), float(v.item())) for v, i in zip(vals, idxs)]

    return best_label, best_prob, top_list


def insert_entity_markers(sentence: str, head: str, tail: str) -> str:
    """
    Simple marker insertion by FIRST occurrence of head and tail spans.
    This is just for quick custom tests.
    For real NYT-style inference, prefer using processed JSONL rows (they're already marked).
    """
    s = sentence

    h_pos = s.lower().find(head.lower())
    t_pos = s.lower().find(tail.lower())

    if h_pos == -1 or t_pos == -1:
        raise ValueError("Could not find head/tail strings inside the sentence (case-insensitive search).")

    # Ensure non-overlap and stable insertion: insert from right-to-left
    spans = []
    spans.append(("H", h_pos, h_pos + len(head)))
    spans.append(("T", t_pos, t_pos + len(tail)))
    spans.sort(key=lambda x: x[1], reverse=True)

    for tag, a, b in spans:
        seg = s[a:b]
        if tag == "H":
            s = s[:a] + f"{E1_START}{seg}{E1_END}" + s[b:]
        else:
            s = s[:a] + f"{E2_START}{seg}{E2_END}" + s[b:]

    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Path to trained model dir, e.g. Neural/RE/models/spanbert_nyt_re")
    ap.add_argument("--processed_dir", required=True, help="Path to processed dir, e.g. Neural/RE/processed")
    ap.add_argument("--test_jsonl", default="", help="Path to processed test jsonl, e.g. Neural/RE/processed/nyt_re_test.jsonl")
    ap.add_argument("--n", type=int, default=10, help="How many random examples to sample from test_jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_length", type=int, default=256)

    # Custom mode
    ap.add_argument("--text", default="", help="Custom sentence (optional)")
    ap.add_argument("--head", default="", help="Head entity text inside --text (optional)")
    ap.add_argument("--tail", default="", help="Tail entity text inside --text (optional)")

    # Filtering
    ap.add_argument("--filter_no_relation", action="store_true", help="Hide predictions that are NA/no_relation (best effort)")
    ap.add_argument("--min_conf", type=float, default=0.0, help="Only print predictions with prob >= this threshold")

    args = ap.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    label2id, id2label = load_label_maps(model, args.processed_dir)
    no_rel_labels = set(find_no_relation_labels(label2id))

    print(f"Loaded labels: {len(label2id)}")
    if no_rel_labels:
        print(f"Detected possible no-relation labels: {sorted(list(no_rel_labels))}")

    # --------------------------
    # Mode 1: custom sentence
    # --------------------------
    if args.text.strip():
        if not args.head.strip() or not args.tail.strip():
            raise SystemExit("Custom mode requires --text, --head, --tail")

        marked = insert_entity_markers(args.text.strip(), args.head.strip(), args.tail.strip())
        print("\n" + "=" * 100)
        print("CUSTOM INPUT")
        print("Raw:   ", args.text.strip())
        print("Marked:", marked)

        pred, conf, top = predict_one(
            marked, tokenizer, model, id2label, device, max_length=args.max_length
        )

        if args.filter_no_relation and pred in no_rel_labels:
            print(f"\nPrediction filtered (no-relation): {pred}  conf={conf:.4f}")
            return

        if conf < args.min_conf:
            print(f"\nPrediction below min_conf ({args.min_conf}): {pred}  conf={conf:.4f}")
            return

        print(f"\nPRED: {pred}  conf={conf:.4f}")
        print("Top-5:")
        for lab, p in top:
            print(f"  - {lab:40s} {p:.4f}")
        return

    # --------------------------
    # Mode 2: processed test jsonl
    # --------------------------
    if not args.test_jsonl:
        raise SystemExit("Provide --test_jsonl for file mode, or use --text/--head/--tail for custom mode.")

    rows = read_jsonl(args.test_jsonl)
    if not rows:
        raise SystemExit(f"No rows loaded from: {args.test_jsonl}")

    random.seed(args.seed)
    sample = rows if args.n <= 0 else random.sample(rows, k=min(args.n, len(rows)))

    print("\n" + "=" * 100)
    print(f"Sampling {len(sample)} rows from: {args.test_jsonl}")

    shown = 0
    for i, r in enumerate(sample):
        text = r.get("text", "")
        gold = r.get("relation", "")
        head = r.get("head", "")
        tail = r.get("tail", "")
        meta = r.get("meta", {})

        pred, conf, top = predict_one(text, tokenizer, model, id2label, device, max_length=args.max_length)

        if args.filter_no_relation and pred in no_rel_labels:
            continue
        if conf < args.min_conf:
            continue

        print("\n" + "-" * 100)
        print(f"[{i}] head={head!r}  tail={tail!r}")
        if isinstance(meta, dict) and meta:
            # Keep it short
            sent = meta.get("sentText")
            if sent:
                print(f"sentText: {sent}")
        print(f"gold: {gold}")
        print(f"pred: {pred}  conf={conf:.4f}")
        print("Top-3:")
        for lab, p in top[:3]:
            print(f"  - {lab:40s} {p:.4f}")

        shown += 1

    print("\n" + "=" * 100)
    print(f"Done. Printed {shown} predictions (after filtering).")


if __name__ == "__main__":
    main()
