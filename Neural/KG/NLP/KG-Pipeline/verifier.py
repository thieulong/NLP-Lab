from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import REL_TO_TEMPLATE


def build_hypothesis(head: str, rel: str, tail: str) -> str:
    tpl = REL_TO_TEMPLATE.get(rel)
    if tpl:
        return tpl.format(h=head, t=tail)
    # fallback: still usable, but weaker
    # (keeps it dataset-agnostic)
    return f"{head} has relation {rel} with {tail}."

def _find_mnli_label_ids(model) -> tuple[int, int, int]:
    """
    Return (entailment_id, neutral_id, contradiction_id) for MNLI heads.
    Works across HF checkpoints with different label order.
    """
    id2label = getattr(model.config, "id2label", {}) or {}
    # normalize
    norm = {i: str(l).lower() for i, l in id2label.items()}
    entail = next((i for i, l in norm.items() if "entail" in l), None)
    neut  = next((i for i, l in norm.items() if "neutral" in l), None)
    contra= next((i for i, l in norm.items() if "contrad" in l), None)

    # common fallback for roberta-large-mnli: 0=contradiction,1=neutral,2=entailment
    if entail is None or neut is None or contra is None:
        return (2, 1, 0)
    return (entail, neut, contra)

def load_verifier(verifier_model_name: str, device: str):
    tok = AutoTokenizer.from_pretrained(verifier_model_name, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(verifier_model_name)
    mdl.to(device)
    mdl.eval()
    ent_id, neu_id, con_id = _find_mnli_label_ids(mdl)
    return tok, mdl, ent_id, neu_id, con_id

@lru_cache(maxsize=20000)
def _verifier_cached_key(premise: str, hypothesis: str) -> tuple[str, str]:
    # cache key normalization
    return (premise.strip(), hypothesis.strip())

def verifier_entailment_prob(
    premise: str,
    hypothesis: str,
    *,
    verifier_tokenizer,
    verifier_model,
    entailment_id: int,
    temperature: float,
    device: str,
    max_length: int = 256
) -> float:
    # cache on normalized strings
    premise, hypothesis = _verifier_cached_key(premise, hypothesis)

    enc = verifier_tokenizer(
        premise,
        hypothesis,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = verifier_model(**enc).logits  # [1, num_labels]
        if temperature and temperature != 1.0:
            logits = logits / float(temperature)
        probs = F.softmax(logits, dim=-1)[0]
        return float(probs[entailment_id].item())

