#!/usr/bin/env python3
"""
Fit per-relation confidence thresholds on the dev/valid set.

Input:
  - processed/nyt_re_valid.jsonl with fields:
      text, label_id, relation (optional), head, tail (optional)
  - trained model directory with correct id2label/label2id

Output:
  - thresholds.json (per relation)
  - report.json (metrics per relation at chosen threshold)

Typical run:
  python Neural/RE/fit_thresholds.py \
    --proc_dir Neural/RE/processed \
    --valid_file nyt_re_valid.jsonl \
    --model_dir Neural/RE/models/spanbert_nyt_re_norel \
    --out_json Neural/RE/processed/thresholds.json \
    --metric f1 \
    --min_margin 0.15
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def device_auto() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore
        return torch.device("mps")
    return torch.device("cpu")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def id_to_label(model, pred_id: int) -> str:
    id2label = model.config.id2label
    if isinstance(id2label, dict):
        if pred_id in id2label:
            return id2label[pred_id]
        s = str(pred_id)
        if s in id2label:
            return id2label[s]
        for k, v in id2label.items():
            try:
                if int(k) == pred_id:
                    return v
            except Exception:
                pass
        raise KeyError(f"pred_id={pred_id} not found in id2label")
    return id2label[pred_id]


@torch.no_grad()
def predict(texts: List[str], model, tokenizer, device: torch.device, max_length: int, temperature: float = 1.0):
    enc = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc)
    logits = out.logits / float(temperature)
    probs = torch.softmax(logits, dim=-1)  # [B, C]

    top2_probs, top2_ids = torch.topk(probs, k=2, dim=-1)
    pred_ids = torch.argmax(probs, dim=-1)

    conf = top2_probs[:, 0].detach().cpu().tolist()
    margin = (top2_probs[:, 0] - top2_probs[:, 1]).detach().cpu().tolist()
    pred_ids = pred_ids.detach().cpu().tolist()
    top2_ids = top2_ids.detach().cpu().tolist()
    top2_probs = top2_probs.detach().cpu().tolist()

    return pred_ids, conf, margin, top2_ids, top2_probs


def f1_from_counts(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return prec, rec, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc_dir", type=str, default="Neural/RE/processed")
    ap.add_argument("--valid_file", type=str, default="nyt_re_valid.jsonl")
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--out_json", type=str, default="Neural/RE/processed/thresholds.json")
    ap.add_argument("--report_json", type=str, default="Neural/RE/processed/threshold_report.json")

    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=64)

    ap.add_argument("--min_margin", type=float, default=0.0, help="Optional: require top1-top2 >= min_margin for acceptance.")
    ap.add_argument("--temperature", type=float, default=1.0, help="Optional: apply fixed temperature at scoring time.")

    ap.add_argument("--metric", type=str, choices=["f1", "precision_at"], default="f1")
    ap.add_argument("--precision_target", type=float, default=0.95)

    ap.add_argument("--no_relation_label", type=str, default="no_relation")
    args = ap.parse_args()

    proc_dir = Path(args.proc_dir)
    valid_path = proc_dir / args.valid_file
    model_dir = Path(args.model_dir)

    if not valid_path.exists():
        raise FileNotFoundError(f"valid file not found: {valid_path}")
    if not model_dir.exists():
        raise FileNotFoundError(f"model_dir not found: {model_dir}")

    device = device_auto()
    print(f"Device: {device}")
    print(f"Loading model: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    rows = read_jsonl(valid_path)
    print(f"Loaded valid rows: {len(rows)}")

    # Prepare gold labels (string)
    gold_labels: List[str] = []
    texts: List[str] = []
    for r in rows:
        texts.append(r["text"])
        # prefer explicit relation label if present, else map label_id via model
        if "relation" in r and isinstance(r["relation"], str):
            gold_labels.append(r["relation"])
        else:
            gold_labels.append(id_to_label(model, int(r["label_id"])))

    # Run predictions in batches
    pred_labels: List[str] = []
    confs: List[float] = []
    margins: List[float] = []

    bs = args.batch_size
    for i in range(0, len(texts), bs):
        batch_texts = texts[i:i+bs]
        pred_ids, conf, margin, _, _ = predict(
            batch_texts, model, tokenizer, device,
            max_length=args.max_length,
            temperature=args.temperature,
        )
        for pid in pred_ids:
            pred_labels.append(id_to_label(model, int(pid)))
        confs.extend([float(x) for x in conf])
        margins.extend([float(x) for x in margin])

    assert len(pred_labels) == len(gold_labels) == len(confs)

    # Collect examples per predicted relation for threshold sweeping
    # We decide acceptance based on: pred_rel == R, conf >= thr, margin >= min_margin
    per_rel_idxs: Dict[str, List[int]] = defaultdict(list)
    for idx, pr in enumerate(pred_labels):
        if pr == args.no_relation_label:
            continue
        per_rel_idxs[pr].append(idx)

    # Sweep thresholds for each relation using only candidates predicted as that relation.
    thresholds: Dict[str, float] = {}
    report: Dict[str, Any] = {}

    # Candidate thresholds: use observed confidences for that relation
    for rel, idxs in sorted(per_rel_idxs.items(), key=lambda kv: len(kv[1]), reverse=True):
        # If relation is very rare in preds, still handle it
        rel_confs = sorted({confs[i] for i in idxs})
        if not rel_confs:
            continue

        best_thr = 1.0
        best_score = -1.0
        best_stats = None

        # Add endpoints
        candidates = [0.0] + rel_confs + [1.0]

        for thr in candidates:
            tp = fp = fn = 0

            for i in idxs:
                if margins[i] < args.min_margin:
                    continue
                accept = confs[i] >= thr
                if not accept:
                    continue

                gold = gold_labels[i]
                if gold == rel:
                    tp += 1
                else:
                    fp += 1

            # FN counts: gold is rel but we did not accept it
            # For FN, we need to consider all rows where gold == rel, not just those predicted as rel.
            for i, g in enumerate(gold_labels):
                if g != rel:
                    continue
                # accepted only if predicted rel and passes gates
                accepted = (pred_labels[i] == rel) and (confs[i] >= thr) and (margins[i] >= args.min_margin)
                if not accepted:
                    fn += 1

            prec, rec, f1 = f1_from_counts(tp, fp, fn)

            if args.metric == "f1":
                score = f1
            else:
                # precision_at: only consider thresholds that meet precision target, maximize recall
                if prec + 1e-12 < args.precision_target:
                    score = -1.0
                else:
                    score = rec

            if score > best_score:
                best_score = score
                best_thr = float(thr)
                best_stats = {
                    "tp": tp, "fp": fp, "fn": fn,
                    "precision": prec, "recall": rec, "f1": f1,
                    "support_gold": sum(1 for g in gold_labels if g == rel),
                    "support_pred": len(idxs),
                }

        thresholds[rel] = best_thr
        report[rel] = {
            "threshold": best_thr,
            "metric": args.metric,
            "min_margin": args.min_margin,
            "temperature": args.temperature,
            **(best_stats or {}),
        }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")

    rep_json = Path(args.report_json)
    rep_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote thresholds: {out_json}")
    print(f"Wrote report:     {rep_json}")
    print(f"Relations fitted: {len(thresholds)}")


if __name__ == "__main__":
    main()

'''
python Neural/RE/fit_thresholds.py \
  --proc_dir Neural/RE/processed \
  --valid_file nyt_re_valid.jsonl \
  --model_dir Neural/RE/models/spanbert_nyt_re_norel \
  --out_json Neural/RE/processed/thresholds.json \
  --report_json Neural/RE/processed/threshold_report.json \
  --min_margin 0.15
'''