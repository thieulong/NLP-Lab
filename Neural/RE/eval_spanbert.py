#!/usr/bin/env python3
"""
Evaluate a SpanBERT relation extraction classifier on NYT-RE processed JSONL.

JSONL format per line:
{
  "text": "... [E1] head [/E1] ... [E2] tail [/E2] ...",
  "label_id": 11,
  "relation": "/location/location/contains",
  "head": "...",
  "tail": "...",
  ...
}

Reports:
- accuracy
- micro F1
- macro F1 (all classes)
- macro F1 excluding no_relation
- top confusion pairs
- optional per-label precision/recall/F1 table

Also supports:
- saving reports to a folder
- fitting per-relation confidence thresholds on a dev/valid split

Example:
  python Neural/RE/eval_spanbert.py \
    --model_dir Neural/RE/models/spanbert_nyt_re_norel \
    --proc_dir Neural/RE/processed \
    --split valid \
    --batch_size 32 \
    --show_per_label \
    --save_reports Neural/RE/benchmarks/spanbert_norel_eval \
    --fit_thresholds \
    --threshold_out Neural/RE/benchmarks/spanbert_norel_eval/thresholds.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ----------------------------
# Helpers
# ----------------------------

def device_auto() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore
        return torch.device("mps")
    return torch.device("cpu")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path, limit: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def id_to_label(pred_id: int, id2label: Any) -> str:
    # HF configs sometimes store id2label with str keys
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
        return f"__UNK_ID_{pred_id}__"
    try:
        return id2label[pred_id]
    except Exception:
        return f"__UNK_ID_{pred_id}__"


def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


def compute_prf(
    y_true: List[int],
    y_pred: List[int],
    num_labels: int,
) -> Tuple[float, float, float, List[Tuple[float, float, float, int]]]:
    """
    Returns:
      acc, micro_f1, macro_f1, per_label[(p,r,f1,support)]
    """
    assert len(y_true) == len(y_pred)

    cm = [[0 for _ in range(num_labels)] for _ in range(num_labels)]
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_labels and 0 <= p < num_labels:
            cm[t][p] += 1

    total = len(y_true)
    correct = sum(cm[i][i] for i in range(num_labels))
    acc = safe_div(correct, total)

    TP = correct
    FP = sum(cm[t][p] for t in range(num_labels) for p in range(num_labels) if t != p)
    FN = FP
    micro_p = safe_div(TP, TP + FP)
    micro_r = safe_div(TP, TP + FN)
    micro_f1 = safe_div(2 * micro_p * micro_r, micro_p + micro_r)

    per_label: List[Tuple[float, float, float, int]] = []
    f1s: List[float] = []
    for k in range(num_labels):
        tp = cm[k][k]
        fp = sum(cm[t][k] for t in range(num_labels) if t != k)
        fn = sum(cm[k][p] for p in range(num_labels) if p != k)
        supp = sum(cm[k][p] for p in range(num_labels))
        p = safe_div(tp, tp + fp)
        r = safe_div(tp, tp + fn)
        f1 = safe_div(2 * p * r, p + r)
        per_label.append((p, r, f1, supp))
        f1s.append(f1)

    macro_f1 = sum(f1s) / num_labels if num_labels > 0 else 0.0
    return acc, micro_f1, macro_f1, per_label


def top_confusions(
    y_true: List[int],
    y_pred: List[int],
    num_labels: int,
    top_n: int = 15,
) -> List[Tuple[int, int, int]]:
    cm = [[0 for _ in range(num_labels)] for _ in range(num_labels)]
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_labels and 0 <= p < num_labels:
            cm[t][p] += 1

    pairs: List[Tuple[int, int, int]] = []
    for t in range(num_labels):
        for p in range(num_labels):
            if t == p:
                continue
            c = cm[t][p]
            if c > 0:
                pairs.append((c, t, p))
    pairs.sort(reverse=True)
    return [(t, p, c) for (c, t, p) in pairs[:top_n]]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, header: List[str], rows: List[List[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ----------------------------
# Dataset
# ----------------------------

class NYTREJsonlDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        r = self.rows[idx]
        text = r.get("text", "")
        label_id = r.get("label_id", None)
        if label_id is None:
            raise ValueError("Row is missing 'label_id'.")
        return text, int(label_id)


def collate_fn(batch, tokenizer, max_length: int):
    texts = [x[0] for x in batch]
    labels = torch.tensor([x[1] for x in batch], dtype=torch.long)
    enc = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    return enc, labels


# ----------------------------
# Threshold fitting
# ----------------------------

def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    p = safe_div(tp, tp + fp)
    r = safe_div(tp, tp + fn)
    return safe_div(2 * p * r, p + r)


def fit_thresholds_per_relation(
    y_true: List[int],
    y_pred: List[int],
    y_conf: List[float],
    id2label_map: Any,
    no_rel_id: Optional[int],
    min_support_predicted: int = 50,
    grid_step: float = 0.005,
) -> Dict[str, float]:
    """
    For each relation label L (excluding no_relation), we consider only examples where model predicted L.
    We choose a threshold on confidence that maximizes F1 for accepting L (higher threshold reduces FP).

    This is designed for KG extraction where you want fewer false positives per relation.
    """
    num = len(y_true)
    assert num == len(y_pred) == len(y_conf)

    thresholds: Dict[str, float] = {}
    label_ids = sorted(set(y_true) | set(y_pred))

    for lab_id in label_ids:
        if no_rel_id is not None and lab_id == no_rel_id:
            continue

        # indices where model predicted this label
        idxs = [i for i in range(num) if y_pred[i] == lab_id]
        if len(idxs) < min_support_predicted:
            continue

        # Precompute (conf, is_correct)
        items = [(y_conf[i], 1 if y_true[i] == lab_id else 0) for i in idxs]
        items.sort(key=lambda x: x[0])

        # Also compute FN for this label that threshold cannot recover:
        # true label is lab_id but model did NOT predict lab_id.
        base_fn = sum(1 for i in range(num) if y_true[i] == lab_id and y_pred[i] != lab_id)

        best_thr = 0.5
        best_f1 = -1.0
        best_prec = -1.0

        # grid search thresholds
        thr = 0.0
        while thr <= 1.000001:
            tp = 0
            fp = 0
            for conf, correct in items:
                if conf >= thr:
                    if correct:
                        tp += 1
                    else:
                        fp += 1
            fn = base_fn + (sum(1 for conf, correct in items if correct and conf < thr))
            f1 = f1_from_counts(tp, fp, fn)
            prec = safe_div(tp, tp + fp)

            # tie-break: higher precision if F1 same
            if (f1 > best_f1) or (abs(f1 - best_f1) < 1e-9 and prec > best_prec):
                best_f1 = f1
                best_prec = prec
                best_thr = thr

            thr += grid_step

        lab = id_to_label(lab_id, id2label_map)
        thresholds[lab] = float(round(best_thr, 6))

    return thresholds


# ----------------------------
# Main
# ----------------------------

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_dir", type=str, required=True)

    # input selection
    ap.add_argument("--proc_dir", type=str, default="", help="If set, use --split to pick nyt_re_{split}.jsonl from here.")
    ap.add_argument("--split", type=str, choices=["train", "valid", "test"], default="test")
    ap.add_argument("--test_file", type=str, default="", help="Direct path to a JSONL file (overrides proc_dir/split).")

    # eval config
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--show_per_label", action="store_true")
    ap.add_argument("--top_confusions", type=int, default=15)

    # no_relation config
    ap.add_argument("--no_relation_label", type=str, default="no_relation")

    # reports
    ap.add_argument("--save_reports", type=str, default="", help="Folder to save metrics/per-label/confusions/thresholds.")

    # threshold fitting
    ap.add_argument("--fit_thresholds", action="store_true", help="Fit per-relation thresholds on this split.")
    ap.add_argument("--threshold_out", type=str, default="", help="Path to write thresholds.json (default: <save_reports>/thresholds.json)")
    ap.add_argument("--thr_grid_step", type=float, default=0.005, help="Threshold search grid step.")
    ap.add_argument("--thr_min_support_pred", type=int, default=50, help="Min predicted count to fit a threshold for a label.")

    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"model_dir not found: {model_dir}")

    # resolve file
    if args.test_file:
        data_file = Path(args.test_file)
    else:
        if not args.proc_dir:
            raise ValueError("Provide --test_file or set --proc_dir and --split.")
        proc_dir = Path(args.proc_dir)
        data_file = proc_dir / f"nyt_re_{args.split}.jsonl"

    if not data_file.exists():
        raise FileNotFoundError(f"data_file not found: {data_file}")

    # load label maps
    id2label_path = model_dir / "id2label.json"
    label2id_path = model_dir / "label2id.json"
    if not id2label_path.exists():
        id2label_path = data_file.parent / "id2label.json"
    if not label2id_path.exists():
        label2id_path = data_file.parent / "label2id.json"
    if not id2label_path.exists() or not label2id_path.exists():
        raise FileNotFoundError("Could not find id2label.json/label2id.json in model_dir or alongside data file.")

    id2label_map = load_json(id2label_path)
    label2id = load_json(label2id_path)
    num_labels = len(label2id)

    no_rel_id = label2id.get(args.no_relation_label, None)
    if no_rel_id is None:
        # tolerate case mismatch
        for k, v in label2id.items():
            if str(k).lower() == args.no_relation_label.lower():
                no_rel_id = int(v)
                break

    print(f"Model:   {model_dir}")
    print(f"Data:    {data_file}")
    print(f"Split:   {args.split}")
    print(f"Labels:  {num_labels}")
    print(f"no_rel:  {args.no_relation_label} -> {no_rel_id}")

    device = device_auto()
    print(f"Device:  {device}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    rows = read_jsonl(data_file, limit=args.limit)
    ds = NYTREJsonlDataset(rows)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_length),
    )

    y_true: List[int] = []
    y_pred: List[int] = []
    y_conf: List[float] = []

    for enc, labels in dl:
        enc = {k: v.to(device) for k, v in enc.items()}
        labels = labels.to(device)

        out = model(**enc)
        probs = torch.softmax(out.logits, dim=-1)
        conf, pred_ids = torch.max(probs, dim=-1)

        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(pred_ids.detach().cpu().tolist())
        y_conf.extend(conf.detach().cpu().tolist())

    acc, micro_f1, macro_f1, per_label = compute_prf(y_true, y_pred, num_labels)

    macro_excl = None
    if no_rel_id is not None and 0 <= no_rel_id < num_labels:
        f1s_excl = [per_label[i][2] for i in range(num_labels) if i != no_rel_id]
        macro_excl = sum(f1s_excl) / len(f1s_excl) if f1s_excl else 0.0

    print("\n====================")
    print("EVAL METRICS")
    print("====================")
    print(f"Samples:   {len(y_true)}")
    print(f"Accuracy:  {acc:.6f}")
    print(f"Micro-F1:  {micro_f1:.6f}")
    print(f"Macro-F1:  {macro_f1:.6f}")
    if macro_excl is not None:
        print(f"Macro-F1 (exclude {args.no_relation_label}): {macro_excl:.6f}")

    print("\n====================")
    print("TOP CONFUSIONS")
    print("====================")
    confs = top_confusions(y_true, y_pred, num_labels, top_n=args.top_confusions)
    for t, p, c in confs:
        t_lab = id_to_label(t, id2label_map)
        p_lab = id_to_label(p, id2label_map)
        print(f"{c:5d}  true={t:>2d}:{t_lab}  ->  pred={p:>2d}:{p_lab}")

    if args.show_per_label:
        print("\n====================")
        print("PER-LABEL PRF")
        print("====================")
        items = []
        for i in range(num_labels):
            p, r, f1, supp = per_label[i]
            lab = id_to_label(i, id2label_map)
            items.append((supp, i, lab, p, r, f1))
        items.sort(reverse=True)

        print(f"{'id':>3s}  {'support':>7s}  {'P':>7s}  {'R':>7s}  {'F1':>7s}  label")
        for supp, i, lab, p, r, f1 in items:
            flag = ""
            if no_rel_id is not None and i == no_rel_id:
                flag = "  <no_relation>"
            print(f"{i:3d}  {supp:7d}  {p:7.4f}  {r:7.4f}  {f1:7.4f}  {lab}{flag}")

    # threshold fitting
    thresholds: Dict[str, float] = {}
    if args.fit_thresholds:
        print("\n====================")
        print("FITTING PER-RELATION THRESHOLDS")
        print("====================")
        thresholds = fit_thresholds_per_relation(
            y_true=y_true,
            y_pred=y_pred,
            y_conf=y_conf,
            id2label_map=id2label_map,
            no_rel_id=no_rel_id,
            min_support_predicted=args.thr_min_support_pred,
            grid_step=args.thr_grid_step,
        )
        print(f"Fitted thresholds for {len(thresholds)} labels (min predicted support={args.thr_min_support_pred}).")

    # save reports
    if args.save_reports:
        out_dir = Path(args.save_reports)
        out_dir.mkdir(parents=True, exist_ok=True)

        metrics = {
            "model_dir": str(model_dir),
            "data_file": str(data_file),
            "split": args.split,
            "samples": len(y_true),
            "accuracy": acc,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "macro_f1_no_relation_excluded": macro_excl,
            "no_relation_label": args.no_relation_label,
            "no_relation_id": no_rel_id,
        }
        write_json(out_dir / "metrics.json", metrics)

        # per label csv
        per_label_rows: List[List[Any]] = []
        for i in range(num_labels):
            p, r, f1, supp = per_label[i]
            lab = id_to_label(i, id2label_map)
            per_label_rows.append([i, lab, supp, round(p, 6), round(r, 6), round(f1, 6), int(i == (no_rel_id if no_rel_id is not None else -1))])
        write_csv(out_dir / "per_label.csv", ["id", "label", "support", "precision", "recall", "f1", "is_no_relation"], per_label_rows)

        # confusions csv
        conf_rows: List[List[Any]] = []
        for t, p, c in confs:
            conf_rows.append([c, t, id_to_label(t, id2label_map), p, id_to_label(p, id2label_map)])
        write_csv(out_dir / "confusions.csv", ["count", "true_id", "true_label", "pred_id", "pred_label"], conf_rows)

        # thresholds
        if args.fit_thresholds and thresholds:
            thr_path = Path(args.threshold_out) if args.threshold_out else (out_dir / "thresholds.json")
            write_json(thr_path, thresholds)
            print(f"Saved thresholds to: {thr_path}")

        print(f"\nSaved reports to: {out_dir}")


if __name__ == "__main__":
    main()