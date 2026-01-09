#!/usr/bin/env python3
"""
Train SpanBERT (or BERT) for sentence-level relation classification on processed NYT RE.

Input files (from preprocess step):
  Neural/RE/processed/nyt_re_{train,valid,test}.jsonl
  Neural/RE/processed/label2id.json
  Neural/RE/processed/id2label.json

Output:
  Neural/RE/models/spanbert_nyt_re/

Run:
  python Neural/RE/train_spanbert_nyt_re.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --------------------
# Paths
# --------------------
ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = Path(__file__).resolve().parent / "processed"
MODEL_DIR = Path(__file__).resolve().parent / "models" / "spanbert_nyt_re"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = str(PROC_DIR / "nyt_re_train.jsonl")
VALID_FILE = str(PROC_DIR / "nyt_re_valid.jsonl")
TEST_FILE  = str(PROC_DIR / "nyt_re_test.jsonl")

LABEL2ID_PATH = PROC_DIR / "label2id.json"
ID2LABEL_PATH = PROC_DIR / "id2label.json"


# --------------------
# Model choice (locked)
# --------------------
MODEL_NAME = "SpanBERT/spanbert-base-cased"  # encoder
TEXT_FIELD = "text"
LABEL_FIELD = "label_id"


def load_label_maps() -> tuple[Dict[str, int], Dict[int, str]]:
    label2id = json.loads(LABEL2ID_PATH.read_text(encoding="utf-8"))
    id2label_str = json.loads(ID2LABEL_PATH.read_text(encoding="utf-8"))
    # json keys are strings; convert to int
    id2label = {int(k): v for k, v in id2label_str.items()}
    return label2id, id2label


def compute_metrics(eval_pred):
    """
    Report accuracy + macro F1 (main) + micro F1.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = (preds == labels).mean()

    # Macro/Micro F1 without sklearn dependency is annoying.
    # We'll use a tiny manual implementation.
    num_labels = int(np.max(labels)) + 1

    # per-class counts
    tp = np.zeros(num_labels, dtype=np.int64)
    fp = np.zeros(num_labels, dtype=np.int64)
    fn = np.zeros(num_labels, dtype=np.int64)

    for p, y in zip(preds, labels):
        if p == y:
            tp[y] += 1
        else:
            fp[p] += 1
            fn[y] += 1

    # per-class F1
    f1s = []
    for k in range(num_labels):
        precision = tp[k] / (tp[k] + fp[k]) if (tp[k] + fp[k]) > 0 else 0.0
        recall = tp[k] / (tp[k] + fn[k]) if (tp[k] + fn[k]) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1s.append(f1)

    macro_f1 = float(np.mean(f1s))

    # micro F1 = micro precision = micro recall for multiclass
    micro_tp = tp.sum()
    micro_fp = fp.sum()
    micro_fn = fn.sum()
    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)) if (micro_precision + micro_recall) > 0 else 0.0

    return {
        "accuracy": float(acc),
        "macro_f1": macro_f1,
        "micro_f1": float(micro_f1),
    }


def main() -> None:
    print("Using model:", MODEL_NAME)
    print("Processed dir:", PROC_DIR)
    print("Saving to:", MODEL_DIR)

    label2id, id2label = load_label_maps()
    num_labels = len(label2id)
    print("num_labels:", num_labels)

    # Load dataset (jsonl)
    ds = load_dataset(
        "json",
        data_files={"train": TRAIN_FILE, "validation": VALID_FILE, "test": TEST_FILE},
    )
    print(ds)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    def tokenize_fn(batch: Dict[str, Any]):
        return tokenizer(
            batch[TEXT_FIELD],
            truncation=True,
            max_length=256,  # good baseline; you can raise later
        )

    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=[TEXT_FIELD, "relation", "head", "tail", "meta"])
    tokenized = tokenized.rename_column(LABEL_FIELD, "labels")
    tokenized.set_format("torch")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label={i: id2label[i] for i in range(num_labels)},
        label2id=label2id,
    )

    # Training args (Mac MPS friendly)
    # TrainingArguments API differs between transformers versions.
    # Some versions use evaluation_strategy, some use eval_strategy.
    common_args = dict(
        output_dir=str(MODEL_DIR),

        # evaluation / saving
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",

        # training speed
        learning_rate=2e-5,
        num_train_epochs=5,
        weight_decay=0.01,
        warmup_ratio=0.06,

        # GPU batching (RTX 3080 16GB)
        per_device_train_batch_size=32,   # start here
        per_device_eval_batch_size=64,

        # mixed precision
        fp16=True,        # use fp16 on 30-series
        bf16=False,

        # throughput
        dataloader_num_workers=8,
        dataloader_pin_memory=True,

        # stability
        gradient_accumulation_steps=1,
    )

    # Try the newer name first, fall back to older name
    try:
        args = TrainingArguments(
            eval_strategy="epoch",
            **common_args,
        )
    except TypeError:
        args = TrainingArguments(
            evaluation_strategy="epoch",
            **common_args,
        )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train
    trainer.train()

    # Final eval on test set
    print("\nEvaluating on test split...")
    metrics = trainer.evaluate(tokenized["test"])
    print(metrics)

    # Save final artifacts
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    (MODEL_DIR / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nSaved model + tokenizer + test_metrics.json to:", MODEL_DIR)


if __name__ == "__main__":
    main()