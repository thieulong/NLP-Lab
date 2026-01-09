#!/usr/bin/env python3
import json
import re
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import requests


OLLAMA_URL_DEFAULT = "http://localhost:11434/api/generate"
OLLAMA_MODEL_DEFAULT = "qwen2.5:7b-instruct"


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


def extract_first_json_obj(text: str) -> str:
    """
    Extract the first JSON object from output.
    Expected output: {"label": "...", "confidence": 0.xx}
    """
    text = text.strip()
    # direct
    if text.startswith("{") and text.endswith("}"):
        return text
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("No JSON object found in LLM output")
    return m.group(0)


def call_ollama(prompt: str, url: str, model: str, timeout: int, temperature: float, num_predict: int) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json().get("response", "").strip()


def llm_classify_relation(
    marked_text: str,
    labels: List[str],
    url: str,
    model: str,
    timeout: int,
    temperature: float,
    num_predict: int,
    retry: int = 2
) -> Tuple[str, Optional[float], str]:
    """
    Returns: (pred_label, confidence, raw_output)
    pred_label is forced into the known label set when possible.
    """
    label_list_str = "\n".join([f"- {l}" for l in labels])

    prompt = f"""
You are doing relation classification for knowledge-graph extraction.

You are given a single sentence that contains two marked entities:
[E1]...[/E1] is the head entity
[E2]...[/E2] is the tail entity

Pick EXACTLY ONE relation label from this list:
{label_list_str}

Return ONLY a JSON object with:
- "label": one label exactly from the l ist
- "confidence": a number between 0 and 1

No explanations. No extra text.

Sentence:
{marked_text}
""".strip()

    last_err = None
    for _ in range(retry + 1):
        try:
            out = call_ollama(prompt, url, model, timeout, temperature, num_predict)
            js = extract_first_json_obj(out)
            obj = json.loads(js)

            pred = str(obj.get("label", "")).strip()
            conf = obj.get("confidence", None)
            try:
                conf = float(conf) if conf is not None else None
            except Exception:
                conf = None

            if pred in labels:
                return pred, conf, out

            # small normalization attempts
            pred_norm = pred.replace(" ", "")
            for l in labels:
                if pred_norm == l.replace(" ", ""):
                    return l, conf, out

            # if invalid, raise to retry
            raise ValueError(f"Invalid label returned: {pred}")

        except Exception as e:
            last_err = e
            time.sleep(0.2)

    # fallback: choose a deterministic default so the run finishes
    return labels[0], None, f"FAILED_PARSE: {last_err}"


def top_confusions(cm: np.ndarray, labels: List[str], k: int = 5) -> List[Tuple[str, str, int]]:
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
    ap.add_argument("--test_jsonl", type=str, default="Neural/RE/processed/nyt_re_test.jsonl")
    ap.add_argument("--id2label", type=str, default="Neural/RE/processed/id2label.json")
    ap.add_argument("--out_dir", type=str, default="Neural/RE/benchmarks/qwen")
    ap.add_argument("--max_examples", type=int, default=300)  # keep it manageable first
    ap.add_argument("--seed_stride", type=int, default=1)     # 1 = take first N, 5 = every 5th example
    ap.add_argument("--ollama_url", type=str, default=OLLAMA_URL_DEFAULT)
    ap.add_argument("--ollama_model", type=str, default=OLLAMA_MODEL_DEFAULT)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--num_predict", type=int, default=256)
    args = ap.parse_args()

    test_path = Path(args.test_jsonl)
    id2label_path = Path(args.id2label)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    id2label = load_json(id2label_path)
    id2label_map = {int(k): v for k, v in id2label.items()}
    labels = [id2label_map[i] for i in range(len(id2label_map))]

    rows = read_jsonl(test_path)

    # subsample for speed and cost
    picked = []
    for i in range(0, len(rows), max(1, args.seed_stride)):
        picked.append(rows[i])
        if len(picked) >= args.max_examples:
            break

    print(f"Evaluating Qwen on {len(picked)} examples")
    print(f"Ollama: {args.ollama_url} | model={args.ollama_model}")
    print(f"Labels: {len(labels)}")

    y_true = []
    y_pred = []
    records = []

    for idx, r in enumerate(picked):
        text = r["text"]
        gold = r["relation"]

        pred, conf, raw = llm_classify_relation(
            marked_text=text,
            labels=labels,
            url=args.ollama_url,
            model=args.ollama_model,
            timeout=args.timeout,
            temperature=args.temperature,
            num_predict=args.num_predict
        )

        y_true.append(gold)
        y_pred.append(pred)

        records.append({
            "i": idx,
            "gold": gold,
            "pred": pred,
            "confidence": conf,
            "text": text,
            "raw": raw if len(raw) <= 2000 else raw[:2000] + "...(truncated)"
        })

        if (idx + 1) % 25 == 0:
            print(f"  done {idx+1}/{len(picked)}")

    # Save raw predictions
    (out_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in records),
        encoding="utf-8"
    )

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
        "settings": {
            "max_examples": args.max_examples,
            "seed_stride": args.seed_stride,
            "temperature": args.temperature,
            "num_predict": args.num_predict,
            "model": args.ollama_model,
        }
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSaved:")
    print(f"  {out_dir / 'predictions.jsonl'}")
    print(f"  {out_dir / 'per_label_report.csv'}")
    print(f"  {out_dir / 'confusion_matrix.npy'}")
    print(f"  {out_dir / 'top5_confusions.json'}")
    print(f"  {out_dir / 'summary.json'}")
    print("\nTop-5 confusions:")
    for a, b, c in top5:
        print(f"  {a}  ->  {b}   ({c})")


if __name__ == "__main__":
    main()
