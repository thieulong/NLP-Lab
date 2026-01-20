#!/usr/bin/env python3
"""
Train SpanBERT (or BERT) for sentence-level relation classification on processed NYT RE.

Input files (from preprocess step):
  Neural/RE/processed/nyt_re_{train,valid,test}.jsonl
  Neural/RE/processed/label2id.json
  Neural/RE/processed/id2label.json

Output:
  Neural/RE/models/spanbert_nyt_re/

Key upgrade:
  - Supports "no_relation" class (expected label id = 0)
  - Handles class imbalance via:
      (A) optional downsampling of no_relation in TRAIN split
      (B) weighted cross-entropy loss (inverse-frequency weights)

Run:
  python Neural/RE/train_spanbert_nyt_re.py
  python Neural/RE/train_spanbert_nyt_re.py --downsample_no_relation --no_relation_max_ratio 2.0
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

# Allow TF32 on NVIDIA (harmless elsewhere)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

TEXT_FIELD = "text"
LABEL_FIELD = "label_id"


# --------------------
# Helpers
# --------------------
def device_auto() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore
        return torch.device("mps")
    return torch.device("cpu")


def read_label_maps(label2id_path: Path, id2label_path: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    label2id = json.loads(label2id_path.read_text(encoding="utf-8"))
    id2label_raw = json.loads(id2label_path.read_text(encoding="utf-8"))
    # JSON keys may be strings
    id2label = {int(k): v for k, v in id2label_raw.items()}
    return label2id, id2label


def compute_class_weights(labels: np.ndarray, num_labels: int, clip_max: float = 20.0) -> np.ndarray:
    """
    Inverse-frequency weights:
      w_k = total / (num_labels * count_k)
    This down-weights dominant classes and up-weights rare ones.
    """
    counts = np.bincount(labels, minlength=num_labels).astype(np.float64)
    total = counts.sum()
    weights = np.zeros(num_labels, dtype=np.float64)
    for k in range(num_labels):
        if counts[k] > 0:
            weights[k] = total / (num_labels * counts[k])
        else:
            weights[k] = 0.0

    # avoid extreme explosion on tiny classes
    weights = np.clip(weights, 0.0, clip_max)
    return weights


def downsample_no_relation_train(
    train_ds: Dataset,
    no_rel_id: int,
    max_ratio: float,
    seed: int,
) -> Dataset:
    """
    Keep all positive examples.
    For no_relation, keep at most max_ratio * num_positive.
    """
    labels = np.array(train_ds[LABEL_FIELD], dtype=np.int64)
    pos_idx = np.where(labels != no_rel_id)[0]
    neg_idx = np.where(labels == no_rel_id)[0]

    num_pos = int(pos_idx.shape[0])
    num_neg = int(neg_idx.shape[0])
    if num_pos == 0:
        print("[WARN] No positive examples found in train split; skipping downsampling.")
        return train_ds

    cap_neg = int(max_ratio * num_pos)
    if num_neg <= cap_neg:
        print(f"[Downsample] no_relation={num_neg} <= cap={cap_neg}, no downsampling needed.")
        return train_ds

    rng = np.random.default_rng(seed)
    keep_neg = rng.choice(neg_idx, size=cap_neg, replace=False)

    keep_idx = np.concatenate([pos_idx, keep_neg])
    rng.shuffle(keep_idx)

    print(
        f"[Downsample] positives={num_pos}, no_relation before={num_neg}, "
        f"cap={cap_neg} (ratio={max_ratio}), kept_total={len(keep_idx)}"
    )
    return train_ds.select(keep_idx.tolist())


@dataclass
class MetricConfig:
    num_labels: int
    no_rel_id: Optional[int]


def make_compute_metrics(metric_cfg: MetricConfig):
    """
    Returns a HF Trainer compute_metrics fn.
    Reports:
      - accuracy
      - macro_f1 (all labels)
      - micro_f1
      - macro_f1_no_relation_excluded (if no_rel_id is known)
    """

    def _compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        acc = float((preds == labels).mean())

        num_labels = metric_cfg.num_labels

        tp = np.zeros(num_labels, dtype=np.int64)
        fp = np.zeros(num_labels, dtype=np.int64)
        fn = np.zeros(num_labels, dtype=np.int64)

        for p, y in zip(preds, labels):
            if p == y:
                tp[y] += 1
            else:
                fp[p] += 1
                fn[y] += 1

        f1s = []
        for k in range(num_labels):
            precision = tp[k] / (tp[k] + fp[k]) if (tp[k] + fp[k]) > 0 else 0.0
            recall = tp[k] / (tp[k] + fn[k]) if (tp[k] + fn[k]) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            f1s.append(float(f1))

        macro_f1 = float(np.mean(f1s))

        micro_tp = tp.sum()
        micro_fp = fp.sum()
        micro_fn = fn.sum()
        micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
        micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
        micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)) if (micro_precision + micro_recall) > 0 else 0.0

        out = {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "micro_f1": float(micro_f1),
        }

        if metric_cfg.no_rel_id is not None and 0 <= metric_cfg.no_rel_id < num_labels:
            f1s_ex = [f1 for i, f1 in enumerate(f1s) if i != metric_cfg.no_rel_id]
            out["macro_f1_no_relation_excluded"] = float(np.mean(f1s_ex)) if f1s_ex else 0.0

        return out

    return _compute_metrics


class WeightedTrainer(Trainer):
    """
    Trainer with weighted cross-entropy loss.
    """

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits
        loss = F.cross_entropy(
            logits,
            labels,
            weight=self.class_weights.to(logits.device),
        )
        return (loss, outputs) if return_outputs else loss


# --------------------
# Main
# --------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc_dir", type=str, default=str(Path(__file__).resolve().parent / "processed"))
    ap.add_argument("--model_dir", type=str, default=str(Path(__file__).resolve().parent / "models" / "spanbert_nyt_re"))
    ap.add_argument("--base_model", type=str, default="SpanBERT/spanbert-base-cased")

    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)

    ap.add_argument("--train_bs", type=int, default=32)
    ap.add_argument("--eval_bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--downsample_no_relation", action="store_true")
    ap.add_argument("--no_relation_max_ratio", type=float, default=2.0, help="Keep at most ratio * positives of no_relation in TRAIN.")

    ap.add_argument("--use_weighted_loss", action="store_true", help="Use inverse-frequency class weights in CE loss.")
    ap.add_argument("--weight_clip_max", type=float, default=20.0)

    args = ap.parse_args()

    proc_dir = Path(args.proc_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_file = str(proc_dir / "nyt_re_train.jsonl")
    valid_file = str(proc_dir / "nyt_re_valid.jsonl")
    test_file = str(proc_dir / "nyt_re_test.jsonl")
    label2id_path = proc_dir / "label2id.json"
    id2label_path = proc_dir / "id2label.json"

    print("Base encoder:", args.base_model)
    print("Processed dir:", proc_dir)
    print("Saving to:", model_dir)

    label2id, id2label = read_label_maps(label2id_path, id2label_path)
    num_labels = len(label2id)
    no_rel_id = label2id.get("no_relation", None)

    print("num_labels:", num_labels)
    print("no_relation id:", no_rel_id)

    # Load dataset (jsonl)
    ds = load_dataset(
        "json",
        data_files={"train": train_file, "validation": valid_file, "test": test_file},
    )
    print(ds)

    # Optional: downsample no_relation in TRAIN ONLY
    if args.downsample_no_relation and no_rel_id is not None:
        ds["train"] = downsample_no_relation_train(
            ds["train"],
            no_rel_id=no_rel_id,
            max_ratio=args.no_relation_max_ratio,
            seed=args.seed,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    def tokenize_fn(batch: Dict[str, Any]):
        return tokenizer(
            batch[TEXT_FIELD],
            truncation=True,
            max_length=args.max_length,
        )

    # remove columns robustly (some splits may not include all keys)
    remove_cols = [TEXT_FIELD, "relation", "head", "tail", "meta"]
    def safe_remove_columns(cols: List[str], existing: List[str]) -> List[str]:
        s = set(existing)
        return [c for c in cols if c in s]

    tokenized = {}
    for split in ["train", "validation", "test"]:
        existing_cols = ds[split].column_names
        tokenized[split] = ds[split].map(
            tokenize_fn,
            batched=True,
            remove_columns=safe_remove_columns(remove_cols, existing_cols),
        )
        tokenized[split] = tokenized[split].rename_column(LABEL_FIELD, "labels")
        tokenized[split].set_format("torch")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=num_labels,
        id2label={i: id2label[i] for i in range(num_labels)},
        label2id=label2id,
    )

    device = device_auto()
    print("Device:", device)

    # Decide precision flags safely
    use_fp16 = torch.cuda.is_available()  # MPS generally doesn't want fp16 in Trainer
    use_bf16 = False

    common_args = dict(
        output_dir=str(model_dir),
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1_no_relation_excluded" if no_rel_id is not None else "macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",

        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,

        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,

        fp16=use_fp16,
        bf16=use_bf16,

        dataloader_num_workers=0 if device.type == "mps" else 8,
        dataloader_pin_memory=True if device.type == "cuda" else False,

        gradient_accumulation_steps=1,
        seed=args.seed,
    )

    # transformers version compatibility: eval_strategy vs evaluation_strategy
    try:
        train_args = TrainingArguments(eval_strategy="epoch", **common_args)
    except TypeError:
        train_args = TrainingArguments(evaluation_strategy="epoch", **common_args)

    metric_cfg = MetricConfig(num_labels=num_labels, no_rel_id=no_rel_id)
    compute_metrics = make_compute_metrics(metric_cfg)

    # Class weights from TRAIN split (after any downsampling)
    class_weights_t = torch.ones(num_labels, dtype=torch.float32)
    if args.use_weighted_loss:
        train_labels = np.array(ds["train"][LABEL_FIELD], dtype=np.int64)
        w = compute_class_weights(train_labels, num_labels=num_labels, clip_max=args.weight_clip_max)
        class_weights_t = torch.tensor(w, dtype=torch.float32)
        print("[WeightedLoss] class weights (id -> weight, label):")
        for i in range(num_labels):
            lab = id2label.get(i, str(i))
            print(f"  {i:2d}  w={float(class_weights_t[i]):.4f}  {lab}")

    TrainerCls = WeightedTrainer if args.use_weighted_loss else Trainer
    trainer_kwargs = dict(
        model=model,
        args=train_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if args.use_weighted_loss:
        trainer = TrainerCls(class_weights=class_weights_t, **trainer_kwargs)  # type: ignore
    else:
        trainer = TrainerCls(**trainer_kwargs)  # type: ignore

    trainer.train()

    print("\nEvaluating on test split...")
    metrics = trainer.evaluate(tokenized["test"])
    print(metrics)

    # Save final artifacts
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    (model_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nSaved model + tokenizer + test_metrics.json to:", model_dir)


if __name__ == "__main__":
    main()