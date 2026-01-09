#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import spacy


# -----------------------------
# Ollama settings (LLM)
# -----------------------------
OLLAMA_URL_DEFAULT = "http://localhost:11434/api/generate"
OLLAMA_MODEL_DEFAULT = "qwen2.5:7b-instruct"


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Edge:
    head: str
    relation: str
    tail: str
    source: str               # "spanbert" | "llm"
    conf: float               # for llm we can set 1.0 or omit, but keep for easy filtering
    sent_id: int
    sentence: str
    head_type: str
    tail_type: str
    topk: Optional[List[Tuple[str, float]]] = None
    note: Optional[str] = None


# -----------------------------
# Utility: JSON array extraction
# -----------------------------
def _extract_json_array(text: str) -> str:
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        return text
    m = re.search(r"\[[\s\S]*\]", text)
    if not m:
        raise ValueError("No JSON array found in LLM output")
    return m.group(0)


# -----------------------------
# Load label maps (robust)
# -----------------------------
def load_label_maps(model_dir: Path, processed_dir: Optional[Path] = None) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Prefer label maps from the fine-tuned model's config.json.
    Fallback to processed_dir/label2id.json and processed_dir/id2label.json.
    """
    config_path = model_dir / "config.json"
    if config_path.exists():
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        label2id = cfg.get("label2id", None)
        id2label = cfg.get("id2label", None)
        if isinstance(label2id, dict) and isinstance(id2label, dict) and len(label2id) > 0:
            # id2label keys may be strings, normalize to int
            id2label_int = {}
            for k, v in id2label.items():
                try:
                    id2label_int[int(k)] = v
                except Exception:
                    pass
            if len(id2label_int) == len(label2id):
                return label2id, id2label_int

    if processed_dir:
        l2i = processed_dir / "label2id.json"
        i2l = processed_dir / "id2label.json"
        if l2i.exists() and i2l.exists():
            label2id = json.loads(l2i.read_text(encoding="utf-8"))
            id2label_raw = json.loads(i2l.read_text(encoding="utf-8"))
            id2label = {int(k): v for k, v in id2label_raw.items()}
            return label2id, id2label

    raise FileNotFoundError(
        "Could not find label maps in model_dir/config.json or processed_dir/{label2id.json,id2label.json}."
    )


# -----------------------------
# spaCy entity handling
# -----------------------------
def dedupe_entities(ents: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen = set()
    out = []
    for t, lab in ents:
        key = (t.strip(), lab.strip())
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def allowed_pair(head_type: str, tail_type: str) -> bool:
    """
    Conservative typed-pair filter to reduce nonsense pairs.
    Adjust as you like.
    """
    # location-location relations
    loc = {"GPE", "LOC", "FAC"}
    # org/person
    org = {"ORG"}
    person = {"PERSON"}

    if head_type in org and tail_type in loc:
        return True  # place_founded etc.
    if head_type in org and tail_type in person:
        return True  # founders, major_shareholders etc.
    if head_type in person and tail_type in org:
        return True  # business/person/company
    if head_type in loc and tail_type in loc:
        return True  # contains/capital/admin divisions
    if head_type in person and tail_type in loc:
        return True  # lived/born/death/nationality (imperfect but ok)
    return False


def build_pairs(ents: List[Tuple[str, str]], max_pairs: int = 50) -> List[Tuple[int, int]]:
    pairs = []
    n = len(ents)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pairs.append((i, j))
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def mark_e1_e2(sentence: str, e1: str, e2: str) -> str:
    """
    Insert [E1]...[/E1] and [E2]...[/E2] around the first match of each entity string.
    If it fails, fallback to original sentence (SpanBERT will still run, but performance drops).
    """
    s = sentence
    # Mark longer one first to reduce overlap issues
    first, second = (e1, e2) if len(e1) >= len(e2) else (e2, e1)

    def _mark(text: str, ent: str, tag_open: str, tag_close: str) -> str:
        # word-boundary-ish match, but keep it simple
        pattern = re.escape(ent)
        m = re.search(pattern, text)
        if not m:
            return text
        a, b = m.start(), m.end()
        return text[:a] + tag_open + text[a:b] + tag_close + text[b:]

    if first == e1:
        s = _mark(s, e1, "[E1]", "[/E1]")
        s = _mark(s, e2, "[E2]", "[/E2]")
    else:
        s = _mark(s, e2, "[E2]", "[/E2]")
        s = _mark(s, e1, "[E1]", "[/E1]")
    return s


# -----------------------------
# SpanBERT predictor
# -----------------------------
@torch.no_grad()
def spanbert_predict(
    model,
    tokenizer,
    id2label: Dict[int, str],
    text_marked: str,
    device: str,
    topk: int = 5,
) -> Tuple[str, float, List[Tuple[str, float]]]:
    inputs = tokenizer(
        text_marked,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    ).to(device)

    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
    pred_id = int(np.argmax(probs))
    pred_label = id2label[pred_id]
    conf = float(probs[pred_id])

    idxs = np.argsort(-probs)[:topk]
    top = [(id2label[int(i)], float(probs[int(i)])) for i in idxs]
    return pred_label, conf, top


# -----------------------------
# LLM predictor (forced classification with NO_RELATION)
# -----------------------------
def llm_predict_relation(
    sentence: str,
    e1: str,
    e1_type: str,
    e2: str,
    e2_type: str,
    allowed_relations: List[str],
    ollama_url: str,
    ollama_model: str,
    temperature: float = 0.0,
    num_predict: int = 256,
    timeout: int = 120,
) -> Tuple[str, str]:
    """
    Returns (relation, note). relation can be in allowed_relations or "NO_RELATION".
    """
    prompt = f"""
You are doing relation extraction as a strict classifier.

Given:
- Sentence: {sentence}
- Head entity (E1): {e1} (type: {e1_type})
- Tail entity (E2): {e2} (type: {e2_type})

Choose exactly ONE label from this list:
{json.dumps(allowed_relations, indent=2)}

Or choose "NO_RELATION" if the sentence does not explicitly state a relation between E1 and E2.

Rules:
- Use ONLY the sentence text, no outside knowledge.
- Be conservative: if uncertain, output "NO_RELATION".
- Output ONLY a JSON array with exactly 1 object:
  [{{"relation": "...", "note": "short justification"}}]
""".strip()

    payload = {
        "model": ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }

    resp = requests.post(ollama_url, json=payload, timeout=timeout)
    resp.raise_for_status()
    out = resp.json().get("response", "").strip()
    arr = json.loads(_extract_json_array(out))
    if not isinstance(arr, list) or len(arr) < 1 or not isinstance(arr[0], dict):
        return "NO_RELATION", "bad_output"

    rel = str(arr[0].get("relation", "NO_RELATION")).strip()
    note = str(arr[0].get("note", "")).strip()

    if rel != "NO_RELATION" and rel not in allowed_relations:
        rel = "NO_RELATION"
        note = "label_not_allowed"

    return rel, note


# -----------------------------
# Export helpers
# -----------------------------
def write_jsonl(path: Path, edges: List[Edge]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for e in edges:
            f.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")


def write_csv(path: Path, edges: List[Edge]) -> None:
    fieldnames = [
        "head", "relation", "tail", "source", "conf",
        "sent_id", "sentence", "head_type", "tail_type",
        "note", "topk"
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for e in edges:
            row = asdict(e)
            row["topk"] = json.dumps(e.topk, ensure_ascii=False) if e.topk else ""
            w.writerow(row)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True, help="SpanBERT fine-tuned model directory")
    ap.add_argument("--processed_dir", type=str, default="Neural/RE/processed", help="Where label2id/id2label live")
    ap.add_argument("--spacy_model", type=str, default="en_core_web_sm")
    ap.add_argument("--text", type=str, default=None, help="Paragraph text directly")
    ap.add_argument("--text_file", type=str, default=None, help="Path to a .txt file containing a paragraph")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max_pairs_per_sent", type=int, default=30)
    ap.add_argument("--spanbert_min_conf", type=float, default=0.50)
    ap.add_argument("--spanbert_margin", type=float, default=0.15)
    ap.add_argument("--llm", action="store_true", help="Enable LLM extraction as well")
    ap.add_argument("--ollama_url", type=str, default=OLLAMA_URL_DEFAULT)
    ap.add_argument("--ollama_model", type=str, default=OLLAMA_MODEL_DEFAULT)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    processed_dir = Path(args.processed_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.text_file:
        paragraph = Path(args.text_file).read_text(encoding="utf-8").strip()
    elif args.text:
        paragraph = args.text.strip()
    else:
        raise ValueError("Provide --text or --text_file")

    # spaCy
    nlp = spacy.load(args.spacy_model)
    doc = nlp(paragraph)
    sents = [s.text.strip() for s in doc.sents if s.text.strip()]

    # SpanBERT load
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    device = args.device
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    label2id, id2label = load_label_maps(model_dir, processed_dir)
    allowed_relations = sorted(label2id.keys())

    spanbert_edges: List[Edge] = []
    llm_edges: List[Edge] = []

    print("================================================================================")
    print("PARAGRAPH:")
    print(paragraph)
    print("================================================================================")
    print(f"Device: {device}")
    print(f"Sentences: {len(sents)}")
    print("================================================================================")

    for sent_id, sent in enumerate(sents):
        d = nlp(sent)
        ents_raw = [(e.text, e.label_) for e in d.ents]
        ents = dedupe_entities(ents_raw)

        print("\n" + "-" * 100)
        print(f"[SENT {sent_id}] {sent}")
        print(f"Entities: {ents}")

        if len(ents) < 2:
            continue

        pairs = build_pairs(ents, max_pairs=args.max_pairs_per_sent)
        kept_pairs = []
        for i, j in pairs:
            h, ht = ents[i]
            t, tt = ents[j]
            if allowed_pair(ht, tt):
                kept_pairs.append((i, j))

        print(f"Pairs (typed+limited): {len(kept_pairs)}")

        for (i, j) in kept_pairs:
            head, head_type = ents[i]
            tail, tail_type = ents[j]

            marked = mark_e1_e2(sent, head, tail)

            # SpanBERT
            pred, conf, topk = spanbert_predict(model, tokenizer, id2label, marked, device=device, topk=5)

            # margin filter
            top1 = topk[0][1] if topk else conf
            top2 = topk[1][1] if len(topk) > 1 else 0.0
            gap = float(top1 - top2)

            if conf >= args.spanbert_min_conf and gap >= args.spanbert_margin:
                spanbert_edges.append(
                    Edge(
                        head=head,
                        relation=pred,
                        tail=tail,
                        source="spanbert",
                        conf=conf,
                        sent_id=sent_id,
                        sentence=sent,
                        head_type=head_type,
                        tail_type=tail_type,
                        topk=topk,
                    )
                )

            # LLM
            if args.llm:
                rel, note = llm_predict_relation(
                    sentence=sent,
                    e1=head, e1_type=head_type,
                    e2=tail, e2_type=tail_type,
                    allowed_relations=allowed_relations,
                    ollama_url=args.ollama_url,
                    ollama_model=args.ollama_model,
                    temperature=0.0,
                    num_predict=256,
                )
                if rel != "NO_RELATION":
                    llm_edges.append(
                        Edge(
                            head=head,
                            relation=rel,
                            tail=tail,
                            source="llm",
                            conf=1.0,
                            sent_id=sent_id,
                            sentence=sent,
                            head_type=head_type,
                            tail_type=tail_type,
                            note=note,
                        )
                    )

    # Write outputs
    write_jsonl(out_dir / "spanbert_edges.jsonl", spanbert_edges)
    write_csv(out_dir / "spanbert_edges.csv", spanbert_edges)

    if args.llm:
        write_jsonl(out_dir / "llm_edges.jsonl", llm_edges)
        write_csv(out_dir / "llm_edges.csv", llm_edges)

        combined = spanbert_edges + llm_edges
        write_jsonl(out_dir / "combined_edges.jsonl", combined)
        write_csv(out_dir / "combined_edges.csv", combined)

    summary = {
        "num_sentences": len(sents),
        "spanbert_edges": len(spanbert_edges),
        "llm_edges": len(llm_edges) if args.llm else 0,
        "spanbert_settings": {
            "min_conf": args.spanbert_min_conf,
            "margin": args.spanbert_margin,
            "max_pairs_per_sent": args.max_pairs_per_sent,
        },
        "llm_settings": {
            "enabled": bool(args.llm),
            "model": args.ollama_model,
            "temperature": 0.0,
            "num_predict": 256,
        },
    }
    (out_dir / "triplet_export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n================================================================================")
    print("Done.")
    print(f"SpanBERT edges kept: {len(spanbert_edges)}")
    if args.llm:
        print(f"LLM edges kept:     {len(llm_edges)}")
        print(f"Wrote combined:     {len(spanbert_edges) + len(llm_edges)}")
    print(f"Out dir: {out_dir}")
    print("================================================================================")


if __name__ == "__main__":
    main()
