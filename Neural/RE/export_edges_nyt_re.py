# Neural/RE/export_edges_from_nyt.py
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


@dataclass
class Pred:
    label: str
    conf: float
    topk: List[Tuple[str, float]]  # [(label, prob), ...]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Bad JSON on line {line_no} in {path}: {e}") from e
    return rows


def load_label_maps(processed_dir: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    label2id_path = processed_dir / "label2id.json"
    id2label_path = processed_dir / "id2label.json"
    if not label2id_path.exists() or not id2label_path.exists():
        raise FileNotFoundError(
            "Could not find label maps. Expected:\n"
            f"  {label2id_path}\n"
            f"  {id2label_path}\n"
            "These should be created by preprocess_nyt_re.py."
        )

    label2id = json.loads(label2id_path.read_text(encoding="utf-8"))
    id2label_raw = json.loads(id2label_path.read_text(encoding="utf-8"))

    # id2label.json might have string keys depending on how it was written
    id2label: Dict[int, str] = {}
    for k, v in id2label_raw.items():
        try:
            id2label[int(k)] = v
        except Exception:
            # if already int or unusual formatting
            id2label[k] = v  # type: ignore

    return label2id, id2label


def device_auto() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # optional: mps if user runs on mac
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict_batch(
    texts: List[str],
    model,
    tokenizer,
    device: torch.device,
    top_k: int = 5,
    max_length: int = 256,
) -> Tuple[List[int], List[float], List[List[Tuple[int, float]]]]:
    """
    Returns:
      pred_ids: [B]
      pred_confs: [B]
      topk_ids_probs: [B][(id, prob), ...] sorted desc
    """
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    out = model(**enc)
    logits = out.logits  # [B, C]
    probs = torch.softmax(logits, dim=-1)

    # Top-k
    k = min(top_k, probs.shape[-1])
    top_probs, top_ids = torch.topk(probs, k=k, dim=-1)

    pred_ids = torch.argmax(probs, dim=-1).tolist()
    pred_confs = probs.max(dim=-1).values.tolist()

    topk_list: List[List[Tuple[int, float]]] = []
    for i in range(probs.shape[0]):
        pairs = [(int(top_ids[i, j].item()), float(top_probs[i, j].item())) for j in range(k)]
        topk_list.append(pairs)

    return pred_ids, pred_confs, topk_list


def accept_prediction(
    conf: float,
    topk: List[Tuple[str, float]],
    threshold: float,
    min_conf: float,
    margin: float,
) -> bool:
    if conf >= threshold:
        return True
    if conf < min_conf:
        return False
    if len(topk) < 2:
        return conf >= min_conf
    # margin between top1 and top2
    return (topk[0][1] - topk[1][1]) >= margin


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        default="Neural/RE/models/spanbert_nyt_re",
        help="Path to fine-tuned model dir (contains config.json, model.safetensors, tokenizer files).",
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default="Neural/RE/processed",
        help="Path to processed dir (contains label2id.json, id2label.json, JSONL splits).",
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default="Neural/RE/processed/nyt_re_test.jsonl",
        help="Which split to export from (JSONL produced by preprocess_nyt_re.py).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="Neural/RE/exports",
        help="Where to write edges.csv / edges.jsonl / summary.",
    )

    # Filtering rules
    parser.add_argument("--threshold", type=float, default=0.80, help="Accept if conf >= threshold.")
    parser.add_argument(
        "--min_conf",
        type=float,
        default=0.50,
        help="If conf < min_conf, always reject. If >= min_conf, may accept via margin rule.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.15,
        help="Accept if (top1 - top2) >= margin and conf >= min_conf.",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Store top-k labels for debugging.")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size.")
    parser.add_argument("--max_length", type=int, default=256, help="Tokenizer max_length.")
    parser.add_argument(
        "--max_rows",
        type=int,
        default=0,
        help="If >0, only process first N rows (useful for quick tests).",
    )

    # Optional: ignore "NA" style labels if your dataset has them
    parser.add_argument(
        "--drop_labels",
        type=str,
        default="",
        help="Comma-separated labels to drop from export, e.g. 'NA,no_relation'.",
    )

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    processed_dir = Path(args.processed_dir)
    input_jsonl = Path(args.input_jsonl)
    out_dir = Path(args.out_dir)

    if not model_dir.exists():
        raise FileNotFoundError(f"model_dir not found: {model_dir}")
    if not processed_dir.exists():
        raise FileNotFoundError(f"processed_dir not found: {processed_dir}")
    if not input_jsonl.exists():
        raise FileNotFoundError(f"input_jsonl not found: {input_jsonl}")

    drop_labels = set([x.strip() for x in args.drop_labels.split(",") if x.strip()])

    label2id, id2label = load_label_maps(processed_dir)

    device = device_auto()
    print(f"Device: {device}")
    print(f"Model dir: {model_dir}")
    print(f"Input: {input_jsonl}")
    print(f"Out dir: {out_dir}")
    print(f"num_labels (label2id): {len(label2id)}")
    if drop_labels:
        print(f"Dropping labels: {sorted(drop_labels)}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    rows = read_jsonl(input_jsonl)
    if args.max_rows and args.max_rows > 0:
        rows = rows[: args.max_rows]

    ensure_dir(out_dir)
    csv_path = out_dir / "edges.csv"
    jsonl_path = out_dir / "edges.jsonl"
    summary_path = out_dir / "export_summary.json"

    kept = 0
    seen = 0
    dropped_by_label = 0
    dropped_by_filter = 0

    # We'll also track label distribution
    label_counts: Dict[str, int] = {}

    with csv_path.open("w", encoding="utf-8", newline="") as f_csv, jsonl_path.open(
        "w", encoding="utf-8"
    ) as f_jsonl:
        writer = csv.DictWriter(
            f_csv,
            fieldnames=[
                "source",
                "relation",
                "target",
                "confidence",
                "topk",
                "text",
                "head",
                "tail",
                "split",
                "meta",
            ],
        )
        writer.writeheader()

        batch_texts: List[str] = []
        batch_rows: List[Dict[str, Any]] = []

        def flush_batch() -> None:
            nonlocal kept, seen, dropped_by_label, dropped_by_filter
            if not batch_texts:
                return

            pred_ids, pred_confs, topk_ids_probs = predict_batch(
                batch_texts,
                model=model,
                tokenizer=tokenizer,
                device=device,
                top_k=args.top_k,
                max_length=args.max_length,
            )

            for r, pred_id, conf, topk_pairs in zip(batch_rows, pred_ids, pred_confs, topk_ids_probs):
                seen += 1

                pred_label = id2label[int(pred_id)]
                topk_labels = [(id2label[int(i)], p) for (i, p) in topk_pairs]

                # drop unwanted relations
                if pred_label in drop_labels:
                    dropped_by_label += 1
                    continue

                if not accept_prediction(
                    conf=conf,
                    topk=topk_labels,
                    threshold=args.threshold,
                    min_conf=args.min_conf,
                    margin=args.margin,
                ):
                    dropped_by_filter += 1
                    continue

                kept += 1
                label_counts[pred_label] = label_counts.get(pred_label, 0) + 1

                out_obj = {
                    "source": r.get("head", ""),
                    "relation": pred_label,
                    "target": r.get("tail", ""),
                    "confidence": round(float(conf), 6),
                    "topk": [(lab, round(float(p), 6)) for lab, p in topk_labels],
                    "text": r.get("text", ""),
                    "head": r.get("head", ""),
                    "tail": r.get("tail", ""),
                    "split": r.get("meta", {}).get("split", ""),
                    "meta": r.get("meta", {}),
                }

                writer.writerow(
                    {
                        "source": out_obj["source"],
                        "relation": out_obj["relation"],
                        "target": out_obj["target"],
                        "confidence": out_obj["confidence"],
                        "topk": json.dumps(out_obj["topk"], ensure_ascii=False),
                        "text": out_obj["text"],
                        "head": out_obj["head"],
                        "tail": out_obj["tail"],
                        "split": out_obj["split"],
                        "meta": json.dumps(out_obj["meta"], ensure_ascii=False),
                    }
                )
                f_jsonl.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

        for r in rows:
            batch_texts.append(r["text"])
            batch_rows.append(r)
            if len(batch_texts) >= args.batch_size:
                flush_batch()
                batch_texts, batch_rows = [], []

        flush_batch()

    summary = {
        "input": str(input_jsonl),
        "model_dir": str(model_dir),
        "processed_dir": str(processed_dir),
        "out_dir": str(out_dir),
        "threshold": args.threshold,
        "min_conf": args.min_conf,
        "margin": args.margin,
        "top_k": args.top_k,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "max_rows": args.max_rows,
        "drop_labels": sorted(drop_labels),
        "seen": seen,
        "kept": kept,
        "dropped_by_label": dropped_by_label,
        "dropped_by_filter": dropped_by_filter,
        "kept_ratio": (kept / seen) if seen else 0.0,
        "label_counts": dict(sorted(label_counts.items(), key=lambda kv: kv[1], reverse=True)),
        "outputs": {
            "csv": str(csv_path),
            "jsonl": str(jsonl_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nDone.")
    print(f"Seen: {seen}")
    print(f"Kept: {kept}  (ratio={summary['kept_ratio']:.3f})")
    print(f"Dropped by label: {dropped_by_label}")
    print(f"Dropped by filter: {dropped_by_filter}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {jsonl_path}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
