#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def batched(iterable, batch_size: int):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]


def top_confusions(cm: np.ndarray, labels: List[str], k: int = 5) -> List[Tuple[str, str, int]]:
    # return top off-diagonal confusions by count
    confs = []
    n = cm.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            c = int(cm[i, j])
            if c > 0:
                confs.append((labels[i], labels[j], c))
    confs.sort(key=lambda x: x[2], reverse=True)
    return confs[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default="Neural/RE/models/spanbert_nyt_re")
    ap.add_argument("--test_jsonl", type=str, default="Neural/RE/processed/nyt_re_test.jsonl")
    ap.add_argument("--id2label", type=str, default="Neural/RE/processed/id2label.json")
    ap.add_argument("--out_dir", type=str, default="Neural/RE/benchmarks/spanbert")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=256)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    test_path = Path(args.test_jsonl)
    id2label_path = Path(args.id2label)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    id2label = load_json(id2label_path)
    # id2label may have string keys. normalize to list by id order.
    id2label_map = {int(k): v for k, v in id2label.items()}
    labels = [id2label_map[i] for i in range(len(id2label_map))]
    label2id = {v: k for k, v in id2label_map.items()}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model dir: {model_dir}")
    print(f"Test: {test_path}")
    print(f"num_labels: {len(labels)}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    rows = read_jsonl(test_path)

    y_true = []
    y_pred = []

    for batch in batched(rows, args.batch_size):
        texts = [r["text"] for r in batch]
        gold = [r["relation"] for r in batch]

        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits
            pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy().tolist()

        preds = [labels[i] for i in pred_ids]
        y_true.extend(gold)
        y_pred.extend(preds)

    # Ensure consistent label ordering for metrics
    report = classification_report(
        y_true, y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0
    )

    df = pd.DataFrame(report).transpose()
    df.to_csv(out_dir / "per_label_report.csv", index=True)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    np.save(out_dir / "confusion_matrix.npy", cm)

    top5 = top_confusions(cm, labels, k=5)
    (out_dir / "top5_confusions.json").write_text(
        json.dumps([{"gold": a, "pred": b, "count": c} for a, b, c in top5], indent=2),
        encoding="utf-8"
    )

    summary = {
        "accuracy": report.get("accuracy", None),
        "macro_f1": report.get("macro avg", {}).get("f1-score", None),
        "micro_f1": report.get("micro avg", {}).get("f1-score", None),
        "num_examples": len(y_true),
        "top5_confusions": [{"gold": a, "pred": b, "count": c} for a, b, c in top5],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSaved:")
    print(f"  {out_dir / 'per_label_report.csv'}")
    print(f"  {out_dir / 'confusion_matrix.npy'}")
    print(f"  {out_dir / 'top5_confusions.json'}")
    print(f"  {out_dir / 'summary.json'}")
    print("\nTop-5 confusions:")
    for a, b, c in top5:
        print(f"  {a}  ->  {b}   ({c})")


if __name__ == "__main__":
    main()
