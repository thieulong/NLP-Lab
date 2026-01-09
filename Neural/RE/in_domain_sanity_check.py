# Neural/RE/in_domain_sanity_check.py
import json
import random
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_label_maps(processed_dir: Path):
    """
    Expect these files created by preprocess_nyt_re.py:
      - processed/label2id.json
      - processed/id2label.json
    """
    label2id_path = processed_dir / "label2id.json"
    id2label_path = processed_dir / "id2label.json"

    if not label2id_path.exists() or not id2label_path.exists():
        raise FileNotFoundError(
            "Could not find label maps. Expected:\n"
            f"  {label2id_path}\n"
            f"  {id2label_path}\n"
            "Tip: re-run preprocess_nyt_re.py to generate them."
        )

    label2id = json.loads(label2id_path.read_text(encoding="utf-8"))
    id2label = json.loads(id2label_path.read_text(encoding="utf-8"))

    # Ensure keys are ints for id2label (sometimes saved as strings)
    id2label_int = {}
    for k, v in id2label.items():
        try:
            id2label_int[int(k)] = v
        except Exception:
            # If it's already int-like, keep as-is
            id2label_int[k] = v

    return label2id, id2label_int


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--processed_dir", required=True)
    ap.add_argument("--test_jsonl", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_length", type=int, default=256)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    processed_dir = Path(args.processed_dir)
    test_jsonl = Path(args.test_jsonl)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    _, id2label = load_label_maps(processed_dir)
    print("Loaded labels:", len(id2label))

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    rows = list(load_jsonl(test_jsonl))
    random.seed(args.seed)
    samples = random.sample(rows, k=min(args.k, len(rows)))

    for i, ex in enumerate(samples):
        text = ex["text"]
        gold = ex["relation"]
        head = ex.get("head", "")
        tail = ex.get("tail", "")

        inputs = tokenizer(
            text,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1)
            pred_id = int(torch.argmax(probs).item())
            conf = float(probs[pred_id].item())

        pred = id2label.get(pred_id, f"<UNK_ID:{pred_id}>")

        print("\n" + "-" * 100)
        print(f"[{i}] head={head!r} tail={tail!r}")
        print("gold:", gold)
        print("pred:", pred, f"conf={conf:.4f}")
        print("marked:", text)


if __name__ == "__main__":
    main()
