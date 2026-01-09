#!/usr/bin/env python3
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
import requests

# ---- Ollama settings (same pattern as your NER code) ----
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b-instruct"


def sentencize(text: str) -> List[str]:
    # simple sentence split, good enough for this experiment
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def load_label_maps(processed_dir: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    label2id = json.loads((processed_dir / "label2id.json").read_text(encoding="utf-8"))
    id2label_raw = json.loads((processed_dir / "id2label.json").read_text(encoding="utf-8"))
    # id2label json keys may be strings
    id2label = {int(k): v for k, v in id2label_raw.items()}
    return label2id, id2label


def mark_pair(sentence: str, e1: Tuple[int, int], e2: Tuple[int, int]) -> str:
    """
    Insert [E1] [/E1] and [E2] [/E2] into the sentence by character spans.
    e1, e2 are (start_char, end_char) within 'sentence'.
    """
    (s1, t1), (s2, t2) = e1, e2
    if s1 == s2 and t1 == t2:
        return ""

    # Ensure we insert from right to left so offsets do not shift
    spans = [("E1", s1, t1), ("E2", s2, t2)]
    spans_sorted = sorted(spans, key=lambda x: x[1], reverse=True)

    out = sentence
    for tag, s, t in spans_sorted:
        if not (0 <= s < t <= len(out)):
            return ""
        out = out[:s] + f"[{tag}]" + out[s:t] + f"[/{tag}]" + out[t:]
    return out


def ollama_classify(marked_sentence: str, allowed_labels: List[str]) -> Dict[str, Any]:
    """
    Ask the LLM to choose exactly one label from allowed_labels.
    Returns: { "relation": str, "rationale_short": str }
    """
    prompt = f"""
You are doing relation classification.

Choose EXACTLY ONE relation label for the relation between [E1] and [E2] in the sentence below.

Return ONLY a JSON object with:
- "relation": one of {allowed_labels}
- "rationale_short": a short phrase (max 12 words)

No extra text.

Sentence:
{marked_sentence}
""".strip()

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 256
        }
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=180)
    r.raise_for_status()
    output_text = r.json().get("response", "").strip()

    # extract first JSON object
    m = re.search(r"\{[\s\S]*\}", output_text)
    if not m:
        return {"relation": None, "rationale_short": "no_json"}

    try:
        obj = json.loads(m.group(0))
    except Exception:
        return {"relation": None, "rationale_short": "bad_json"}

    rel = str(obj.get("relation", "")).strip()
    rat = str(obj.get("rationale_short", "")).strip()
    if rel not in allowed_labels:
        return {"relation": None, "rationale_short": "label_not_allowed"}
    return {"relation": rel, "rationale_short": rat}


@torch.inference_mode()
def spanbert_predict(model, tokenizer, text: str, id2label: Dict[int, str], topk: int = 5) -> Dict[str, Any]:
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=256,
        padding=False,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    logits = model(**inputs).logits[0]
    probs = torch.softmax(logits, dim=-1)

    conf, pred_id = torch.max(probs, dim=-1)
    pred_label = id2label[int(pred_id.item())]

    top_probs, top_ids = torch.topk(probs, k=min(topk, probs.shape[-1]))
    top = [(id2label[int(i.item())], float(p.item())) for p, i in zip(top_probs, top_ids)]

    return {
        "pred": pred_label,
        "conf": float(conf.item()),
        "topk": top
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="Neural/RE/models/spanbert_nyt_re")
    ap.add_argument("--processed_dir", default="Neural/RE/processed")
    ap.add_argument("--spacy_model", default="en_core_web_trf")
    ap.add_argument("--max_pairs_per_sentence", type=int, default=20)
    ap.add_argument("--paragraph_file", default=None, help="Optional path to a txt file. If omitted, uses built-in demo paragraph.")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    processed_dir = Path(args.processed_dir)

    label2id, id2label = load_label_maps(processed_dir)
    allowed_labels = list(label2id.keys())

    print(f"SpanBERT model_dir: {model_dir}")
    print(f"Processed dir:      {processed_dir}")
    print(f"num_labels:         {len(allowed_labels)}")
    print(f"spaCy model:        {args.spacy_model}")

    # ---- Load SpanBERT fine-tuned model ----
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    # ---- Load spaCy ----
    import spacy
    nlp = spacy.load(args.spacy_model)

    # ---- Paragraph ----
    if args.paragraph_file:
        paragraph = Path(args.paragraph_file).read_text(encoding="utf-8").strip()
    else:
        paragraph = (
            "Apple was founded in Cupertino, and Tim Cook later led Apple from California. "
            "Google was founded in Mountain View by Larry Page and Sergey Brin. "
            "In Russia, Moscow is the capital, but in Kazakhstan, Astana is the capital. "
            "In New York, Manhattan contains the Upper West Side, and Queens contains Long Island City. "
            "Elon Musk is associated with Tesla, and Tesla has major shareholders including Elon Musk."
        )

    print("\n" + "=" * 100)
    print("PARAGRAPH:")
    print(paragraph)

    sentences = sentencize(paragraph)

    print("\n" + "=" * 100)
    print(f"Device: {device}")
    print(f"Sentences: {len(sentences)}")
    print("=" * 100)

    pair_idx = 0
    for s_idx, sent in enumerate(sentences):
        doc = nlp(sent)

        ents = []
        for ent in doc.ents:
            if ent.label_ in ("PERSON", "ORG", "GPE", "LOC"):
                ents.append((ent.text, ent.start_char, ent.end_char, ent.label_))

        # de-dup by span
        uniq = []
        seen = set()
        for t, a, b, lab in ents:
            key = (a, b)
            if key not in seen:
                uniq.append((t, a, b, lab))
                seen.add(key)

        if len(uniq) < 2:
            continue

        # generate ordered pairs (E1, E2) within sentence
        pairs = []
        for i in range(len(uniq)):
            for j in range(len(uniq)):
                if i == j:
                    continue
                pairs.append((uniq[i], uniq[j]))

        pairs = pairs[: args.max_pairs_per_sentence]

        print("\n" + "-" * 100)
        print(f"[SENT {s_idx}] {sent}")
        print(f"Entities: {[(t, lab) for (t, _, _, lab) in uniq]}")
        print(f"Pairs (limited): {len(pairs)}")

        for (e1, e2) in pairs:
            (t1, s1, e1c, lab1) = e1
            (t2, s2, e2c, lab2) = e2

            marked = mark_pair(sent, (s1, e1c), (s2, e2c))
            if not marked:
                continue

            span_out = spanbert_predict(model, tokenizer, marked, id2label, topk=3)

            llm_out = ollama_classify(marked, allowed_labels)

            print("\n" + "-" * 60)
            print(f"[{pair_idx}] E1='{t1}' ({lab1})  ->  E2='{t2}' ({lab2})")
            print(f"MARKED: {marked}")
            print(f"SpanBERT: {span_out['pred']}  conf={span_out['conf']:.4f}  top3={span_out['topk']}")
            print(f"LLM:      {llm_out.get('relation')}  note={llm_out.get('rationale_short')}")
            pair_idx += 1


if __name__ == "__main__":
    main()
