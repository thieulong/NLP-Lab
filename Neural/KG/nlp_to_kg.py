#!/usr/bin/env python3
"""
End-to-end KG triple extraction from raw text:
  raw text -> spaCy sentence split + NER -> candidate pair proposal (rich patterns)
  -> SpanBERT NYT RE classifier -> type gating + per-relation thresholds
  -> optional semantic trigger gating
  -> print accepted triples + rejected triples grouped by reason

Run:
  python Neural/RE/e2e_kg_from_text.py --model_dir Neural/RE/models/spanbert_nyt_re --interactive --log_all
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Iterable

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import spacy


# -------------------------
# Config: label semantics
# -------------------------

REL_SCHEMA: Dict[str, List[Tuple[str, str]]] = {
    "/business/company/place_founded": [("ORG", "GPE"), ("ORG", "LOC")],
    "/business/company/founders": [("ORG", "PERSON")],
    "/business/company/major_shareholders": [("ORG", "PERSON"), ("ORG", "ORG")],
    "/business/company_shareholder/major_shareholder_of": [("PERSON", "ORG"), ("ORG", "ORG")],
    "/business/person/company": [("PERSON", "ORG")],

    "/people/person/place_lived": [("PERSON", "GPE"), ("PERSON", "LOC")],
    "/people/person/place_of_birth": [("PERSON", "GPE"), ("PERSON", "LOC")],
    "/people/deceased_person/place_of_death": [("PERSON", "GPE"), ("PERSON", "LOC")],
    "/people/person/nationality": [("PERSON", "GPE"), ("PERSON", "NORP")],
    "/people/person/children": [("PERSON", "PERSON")],

    "/location/country/capital": [("GPE", "GPE"), ("GPE", "LOC")],
    "/location/administrative_division/country": [("GPE", "GPE"), ("LOC", "GPE")],
    "/location/country/administrative_divisions": [("GPE", "GPE")],
    "/location/location/contains": [("GPE", "GPE"), ("GPE", "LOC"), ("LOC", "LOC"), ("LOC", "GPE")],
    "/location/neighborhood/neighborhood_of": [("GPE", "GPE"), ("LOC", "GPE"), ("GPE", "LOC")],
}

REL_THRESH: Dict[str, float] = {
    "/location/location/contains": 0.99,
    "/people/person/place_lived": 0.95,
    "/people/person/place_of_birth": 0.95,
    "/people/deceased_person/place_of_death": 0.95,
    "/location/administrative_division/country": 0.95,

    "/location/country/capital": 0.90,
    "/business/company/place_founded": 0.90,
    "/business/company/founders": 0.85,
    "/business/person/company": 0.90,
    "/people/person/nationality": 0.90,
}

REL_TRIGGERS: Dict[str, List[str]] = {
    "/business/company/place_founded": ["founded", "headquartered", "based", "incorporated"],
    "/business/company/founders": ["founded by", "co-founded", "founder"],
    "/business/person/company": ["works at", "joined", "ceo", "executive", "employee", "employed", "president"],
    "/business/company/major_shareholders": ["major shareholder", "major shareholders", "stake", "owns", "ownership"],
    "/location/country/capital": ["capital"],
    "/people/person/place_of_birth": ["born"],
    "/people/person/place_lived": ["lives in", "lived in", "resides in", "resident"],
    "/people/person/nationality": ["nationality", "citizen", "citizenship"],
}


# -------------------------
# Utilities
# -------------------------

def device_auto() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore
        return torch.device("mps")
    return torch.device("cpu")


def coarse_ent_type(spacy_label: str) -> str:
    if spacy_label in {"PERSON", "ORG", "GPE", "LOC", "NORP", "FAC"}:
        return spacy_label
    return spacy_label


def mark_two_spans(sent_text: str, span1: Tuple[int, int], span2: Tuple[int, int]) -> str:
    (a1, b1), (a2, b2) = span1, span2
    if a1 == a2 and b1 == b2:
        return ""

    if a1 < a2:
        first = ("E1", a1, b1)
        second = ("E2", a2, b2)
    else:
        first = ("E2", a2, b2)
        second = ("E1", a1, b1)

    tag1, s1, e1 = first
    tag2, s2, e2 = second

    out = sent_text
    out = out[:e2] + f"[/{tag2}]" + out[e2:]
    out = out[:s2] + f"[{tag2}]" + out[s2:]
    out = out[:e1] + f"[/{tag1}]" + out[e1:]
    out = out[:s1] + f"[{tag1}]" + out[s1:]
    return out


@dataclass
class CandidatePair:
    head_text: str
    tail_text: str
    head_type: str
    tail_type: str
    head_span: Tuple[int, int]
    tail_span: Tuple[int, int]
    pattern: str  # which pattern proposed this


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


# -------------------------
# Pair proposal (richer patterns)
# -------------------------

def propose_pairs(sent) -> List[CandidatePair]:
    """
    Propose entity pairs that are likely supported by the sentence structure.

    Implemented patterns (high-level):
      P1  SVO:        VERB nsubj -> dobj
      P2  pred+prep:  predicate (VERB/ADJ) with nsubj and prep->pobj
      P3  copular NOUN/ADJ predicate: ROOT has child 'cop' and nsubj, plus prep->pobj
      P3b copular VERB ROOT ("be"): VERB has nsubj(entity) + attr(noun) + prep->pobj
      P4  possessive-copular: possessor + (capital/founder/shareholder/etc) + cop + attr
      P4b possessive under nsubj (VERB ROOT): nsubj(noun with poss) + attr(entity)
      P5  apposition: X, CEO Y (appos links)
      P6  noun-of:    (role noun) + prep 'of' -> entity (with nearby subject or possessor)
      P7  relative clause subject recovery: who/which/that as nsubj -> antecedent entity
      P8  conjunction expansion
      P0  fallback: if no candidates, propose nearby entity pairs (windowed)
    """

    # Token index -> (ent_text, ent_type, (start,end)) for tokens inside an entity span
    tok2ent: Dict[int, Tuple[str, str, Tuple[int, int]]] = {}
    for ent in sent.ents:
        t = coarse_ent_type(ent.label_)
        span = (ent.start_char - sent.start_char, ent.end_char - sent.start_char)
        for tok in ent:
            tok2ent[tok.i] = (ent.text, t, span)

    def ent_from_token(tok) -> Optional[Tuple[str, str, Tuple[int, int]]]:
        return tok2ent.get(tok.i)

    def first_ent_in_subtree(tok) -> Optional[Tuple[str, str, Tuple[int, int]]]:
        for t in tok.subtree:
            if t.i in tok2ent:
                return tok2ent[t.i]
        return None

    def ent_or_subtree(tok) -> Optional[Tuple[str, str, Tuple[int, int]]]:
        e = ent_from_token(tok)
        if e:
            return e
        return first_ent_in_subtree(tok)

    def antecedent_entity_for_relpron(tok) -> Optional[Tuple[str, str, Tuple[int, int]]]:
        head = tok.head
        for _ in range(3):
            e = ent_or_subtree(head)
            if e:
                return e
            head = head.head
            if head is None:
                break
        return None

    def add_pair(cands: List[CandidatePair], h, t, pattern: str) -> None:
        h_text, h_type, h_span = h
        t_text, t_type, t_span = t
        if not h_text or not t_text or h_text == t_text:
            return
        cands.append(CandidatePair(
            head_text=h_text, tail_text=t_text,
            head_type=h_type, tail_type=t_type,
            head_span=h_span, tail_span=t_span,
            pattern=pattern,
        ))

    cands: List[CandidatePair] = []

    # -------------------
    # P1: SVO (nsubj + VERB + dobj)
    # -------------------
    for tok in sent:
        if tok.pos_ != "VERB":
            continue
        subj = None
        obj = None
        for child in tok.children:
            if child.dep_ == "nsubj":
                subj = ent_or_subtree(child)
                if subj and subj[0].lower() in {"who", "which", "that"}:
                    subj = antecedent_entity_for_relpron(child)
            elif child.dep_ == "dobj":
                obj = ent_or_subtree(child)

        if subj and obj:
            add_pair(cands, subj, obj, "P1_SVO")

    # -------------------
    # P2: predicate (VERB/ADJ) with prep->pobj and a subject
    # -------------------
    for tok in sent:
        if tok.pos_ not in ("VERB", "ADJ"):
            continue

        subj = None
        for child in tok.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                subj = ent_or_subtree(child)
                if subj and subj[0].lower() in {"who", "which", "that"}:
                    subj = antecedent_entity_for_relpron(child)

        if not subj:
            continue

        for child in tok.children:
            if child.dep_ != "prep":
                continue
            for grand in child.children:
                if grand.dep_ == "pobj":
                    pobj = ent_or_subtree(grand)
                    if pobj:
                        add_pair(cands, subj, pobj, "P2_pred_prep")

    # -------------------
    # P3: copular NOUN/ADJ predicate (token has 'cop' child)
    # -------------------
    for tok in sent:
        has_cop = any(ch.dep_ == "cop" for ch in tok.children)
        if not has_cop:
            continue

        subj = None
        for ch in tok.children:
            if ch.dep_ == "nsubj":
                subj = ent_or_subtree(ch)

        if not subj:
            continue

        for ch in tok.children:
            if ch.dep_ != "prep":
                continue
            for grand in ch.children:
                if grand.dep_ == "pobj":
                    pobj = ent_or_subtree(grand)
                    if pobj:
                        add_pair(cands, subj, pobj, "P3_copular_prep")

    # -------------------
    # P3b: VERB ROOT copular ("be") with nsubj + attr(noun) + prep->pobj
    # Fixes: "Elon Musk is a major shareholder of Tesla."
    # -------------------
    for tok in sent:
        if tok.pos_ != "VERB":
            continue
        # look for "is/was/are/were" kind of constructions
        has_attr = any(ch.dep_ == "attr" for ch in tok.children)
        has_subj = any(ch.dep_ == "nsubj" for ch in tok.children)
        if not (has_attr and has_subj):
            continue

        subj_ent = None
        attr_tok = None
        for ch in tok.children:
            if ch.dep_ == "nsubj":
                subj_ent = ent_or_subtree(ch)
                if subj_ent and subj_ent[0].lower() in {"who", "which", "that"}:
                    subj_ent = antecedent_entity_for_relpron(ch)
            elif ch.dep_ == "attr":
                attr_tok = ch

        if not (subj_ent and attr_tok):
            continue

        # From the attr noun, find "of X"/other preps leading to pobj entities
        for ch in attr_tok.children:
            if ch.dep_ != "prep":
                continue
            for grand in ch.children:
                if grand.dep_ == "pobj":
                    pobj = ent_or_subtree(grand)
                    if pobj:
                        add_pair(cands, subj_ent, pobj, "P3b_be_attr_prep")

    # -------------------
    # P4: possessive-copular bridging when predicate noun has cop + poss + attr
    # -------------------
    for tok in sent:
        has_cop = any(ch.dep_ == "cop" for ch in tok.children)
        if not has_cop:
            continue

        possessor = None
        attr_ent = None

        for ch in tok.children:
            if ch.dep_ == "poss":
                possessor = ent_or_subtree(ch)
            elif ch.dep_ == "attr":
                attr_ent = ent_or_subtree(ch)

        of_obj = None
        for ch in tok.children:
            if ch.dep_ == "prep" and ch.lemma_.lower() == "of":
                for grand in ch.children:
                    if grand.dep_ == "pobj":
                        of_obj = ent_or_subtree(grand)

        if attr_ent and possessor:
            add_pair(cands, possessor, attr_ent, "P4_poss_cop_attr")
        if attr_ent and of_obj:
            add_pair(cands, of_obj, attr_ent, "P4_of_cop_attr")

    # -------------------
    # P4b: VERB ROOT ("be") with nsubj noun that has possessor + attr entity
    # Fixes: "Russia's capital is Moscow."
    # -------------------
    for tok in sent:
        if tok.pos_ != "VERB":
            continue
        nsubj_tok = None
        attr_ent = None
        for ch in tok.children:
            if ch.dep_ == "nsubj":
                nsubj_tok = ch
            elif ch.dep_ == "attr":
                attr_ent = ent_or_subtree(ch)

        if not (nsubj_tok and attr_ent):
            continue

        # possessor lives under the nsubj noun ("capital" <- poss "Russia")
        possessor = None
        for ch in nsubj_tok.children:
            if ch.dep_ == "poss":
                possessor = ent_or_subtree(ch)

        if possessor and attr_ent:
            add_pair(cands, possessor, attr_ent, "P4b_be_nsubjPoss_attr")

    # -------------------
    # P5: apposition
    # -------------------
    for tok in sent:
        if tok.dep_ != "appos":
            continue
        appos_ent = ent_or_subtree(tok)
        head_ent = ent_or_subtree(tok.head)
        if appos_ent and head_ent:
            add_pair(cands, head_ent, appos_ent, "P5_appos")
            add_pair(cands, appos_ent, head_ent, "P5_appos_rev")

    # -------------------
    # P6: noun-of
    # -------------------
    for tok in sent:
        if tok.pos_ != "NOUN":
            continue

        of_obj = None
        for ch in tok.children:
            if ch.dep_ == "prep" and ch.lemma_.lower() == "of":
                for grand in ch.children:
                    if grand.dep_ == "pobj":
                        of_obj = ent_or_subtree(grand)

        if not of_obj:
            continue

        possessor = None
        for ch in tok.children:
            if ch.dep_ == "poss":
                possessor = ent_or_subtree(ch)

        if possessor and of_obj:
            add_pair(cands, possessor, of_obj, "P6_noun_of_poss")

        has_cop = any(ch.dep_ == "cop" for ch in tok.children)
        if has_cop:
            subj = None
            for ch in tok.children:
                if ch.dep_ == "nsubj":
                    subj = ent_or_subtree(ch)
            if subj and of_obj:
                add_pair(cands, subj, of_obj, "P6_noun_of_cop_subj")

    # -------------------
    # Deduplicate (keep earliest)
    # -------------------
    seen = set()
    uniq: List[CandidatePair] = []
    for c in cands:
        key = (c.head_span, c.tail_span, c.pattern)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)

    # -------------------
    # P8: conjunction expansion
    # -------------------
    span2tokens: Dict[Tuple[int, int], List[Any]] = {}
    for tok in sent:
        e = ent_from_token(tok)
        if not e:
            continue
        _, _, sp = e
        span2tokens.setdefault(sp, []).append(tok)

    def conjunct_entity_spans_for_entity_span(ent_span: Tuple[int, int]) -> List[Tuple[str, str, Tuple[int, int]]]:
        toks = span2tokens.get(ent_span, [])
        out: List[Tuple[str, str, Tuple[int, int]]] = []
        for tok in toks:
            for conj in tok.conjuncts:
                e2 = ent_or_subtree(conj)
                if e2:
                    out.append(e2)
        seen_sp = set()
        uniq_out = []
        for e2 in out:
            sp = e2[2]
            if sp in seen_sp:
                continue
            seen_sp.add(sp)
            uniq_out.append(e2)
        return uniq_out

    expanded: List[CandidatePair] = list(uniq)
    for c in uniq:
        for h2 in conjunct_entity_spans_for_entity_span(c.head_span):
            expanded.append(CandidatePair(
                head_text=h2[0], head_type=h2[1], head_span=h2[2],
                tail_text=c.tail_text, tail_type=c.tail_type, tail_span=c.tail_span,
                pattern=c.pattern + "+P8_head_conj",
            ))
        for t2 in conjunct_entity_spans_for_entity_span(c.tail_span):
            expanded.append(CandidatePair(
                head_text=c.head_text, head_type=c.head_type, head_span=c.head_span,
                tail_text=t2[0], tail_type=t2[1], tail_span=t2[2],
                pattern=c.pattern + "+P8_tail_conj",
            ))

    final_seen = set()
    final: List[CandidatePair] = []
    for c in expanded:
        key = (c.head_span, c.tail_span, c.pattern)
        if key in final_seen:
            continue
        final_seen.add(key)
        final.append(c)

    # -------------------
    # P0: fallback (only if nothing proposed)
    # Windowed entity-pair proposal to avoid 0/0 in weird syntax or new domains.
    # -------------------
    if not final:
        ents = list(sent.ents)
        # keep it conservative: only propose pairs within ~12 tokens
        MAX_TOKEN_DISTANCE = 12
        for i in range(len(ents)):
            for j in range(i + 1, len(ents)):
                e1 = ents[i]
                e2 = ents[j]
                if abs(e1.start - e2.start) > MAX_TOKEN_DISTANCE:
                    continue
                h = (e1.text, coarse_ent_type(e1.label_), (e1.start_char - sent.start_char, e1.end_char - sent.start_char))
                t = (e2.text, coarse_ent_type(e2.label_), (e2.start_char - sent.start_char, e2.end_char - sent.start_char))
                add_pair(final, h, t, "P0_fallback_window")
                add_pair(final, t, h, "P0_fallback_window_rev")

        # dedup fallback
        ded = set()
        ded_final = []
        for c in final:
            key = (c.head_span, c.tail_span, c.pattern)
            if key not in ded:
                ded.add(key)
                ded_final.append(c)
        final = ded_final

    return final


# -------------------------
# SpanBERT prediction + gating
# -------------------------

@torch.no_grad()
def predict_one(text: str, model, tokenizer, device: torch.device, top_k: int = 5, max_length: int = 256):
    enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc)
    probs = torch.softmax(out.logits, dim=-1).squeeze(0)
    conf = float(probs.max().item())
    pred_id = int(torch.argmax(probs).item())

    k = min(top_k, probs.shape[-1])
    top_probs, top_ids = torch.topk(probs, k=k)
    topk = [(int(i.item()), float(p.item())) for i, p in zip(top_ids, top_probs)]
    return pred_id, conf, topk


def passes_schema(rel: str, head_type: str, tail_type: str) -> bool:
    if rel not in REL_SCHEMA:
        return True
    allowed = REL_SCHEMA[rel]
    return any((head_type == h and tail_type == t) for (h, t) in allowed)


def passes_trigger_gate(rel: str, sentence_text: str) -> bool:
    triggers = REL_TRIGGERS.get(rel)
    if not triggers:
        return True
    s = sentence_text.lower()
    return any(trig in s for trig in triggers)


def get_rel_threshold(rel: str, default_threshold: float) -> float:
    return REL_THRESH.get(rel, default_threshold)


# -------------------------
# Logging records
# -------------------------

@dataclass
class DecisionRecord:
    sent_id: int
    sentence: str
    head: str
    tail: str
    head_type: str
    tail_type: str
    pattern: str
    marked: str
    pred_rel: str
    conf: float
    margin: Optional[float]
    topk: List[Tuple[str, float]]
    decision: str
    reason: str


def print_grouped(records: List[DecisionRecord], log_limit: int, title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    groups: Dict[str, List[DecisionRecord]] = {}
    for r in records:
        groups.setdefault(r.reason, []).append(r)

    for reason, items in sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True):
        print(f"{reason:30s} count={len(items)}")

    for reason, items in sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True):
        print("\n" + "-" * 100)
        print(f"REASON: {reason}  (showing up to {log_limit})")
        for r in items[:log_limit]:
            print(f"[SENT {r.sent_id}] {r.sentence}")
            print(f"PAIR: {r.head} ({r.head_type}) -> {r.tail} ({r.tail_type})")
            print(f"PATTERN: {r.pattern}")
            mv = None if r.margin is None else round(r.margin, 6)
            print(f"PRED: {r.pred_rel}  conf={r.conf:.6f}  margin={mv}")
            print(f"TOPK: {[(lab, round(p,6)) for lab,p in r.topk]}")
            print()


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default="Neural/RE/models/spanbert_nyt_re")
    ap.add_argument("--spacy_model", type=str, default="en_core_web_trf")
    ap.add_argument("--text", type=str, default="")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--default_threshold", type=float, default=0.90)
    ap.add_argument("--min_margin", type=float, default=0.15)
    ap.add_argument("--use_trigger_gate", action="store_true")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--max_length", type=int, default=256)

    ap.add_argument("--log_all", action="store_true")
    ap.add_argument("--log_limit", type=int, default=10)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"model_dir not found: {model_dir}")

    print(f"Loading spaCy: {args.spacy_model}")
    nlp = spacy.load(args.spacy_model)

    device = device_auto()
    print(f"Device: {device}")
    print(f"Loading SpanBERT from: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    def run_once(raw_text: str):
        raw_text = raw_text.strip()
        if not raw_text:
            return

        doc = nlp(raw_text)

        accepted: List[DecisionRecord] = []
        rejected: List[DecisionRecord] = []

        for sent_id, sent in enumerate(doc.sents):
            sent_text = sent.text
            cands = propose_pairs(sent)

            print("ENTITIES:", [(e.text, e.label_) for e in sent.ents])
            print("CANDS:", [(c.head_text, c.tail_text, c.pattern) for c in propose_pairs(sent)])

            for c in cands:
                marked = mark_two_spans(sent_text, c.head_span, c.tail_span)
                if not marked:
                    rejected.append(DecisionRecord(
                        sent_id=sent_id, sentence=sent_text,
                        head=c.head_text, tail=c.tail_text,
                        head_type=c.head_type, tail_type=c.tail_type,
                        pattern=c.pattern,
                        marked="",
                        pred_rel="",
                        conf=0.0, margin=None, topk=[],
                        decision="REJECT", reason="reject_mark_failed",
                    ))
                    continue

                pred_id, conf, topk_ids = predict_one(
                    marked, model=model, tokenizer=tokenizer, device=device,
                    top_k=args.top_k, max_length=args.max_length
                )
                rel = id_to_label(model, pred_id)
                topk = [(id_to_label(model, i), p) for (i, p) in topk_ids]

                margin_val: Optional[float] = None
                if len(topk_ids) >= 2:
                    margin_val = topk_ids[0][1] - topk_ids[1][1]

                # Gate 1: confidence threshold
                rel_thr = get_rel_threshold(rel, args.default_threshold)
                if conf < rel_thr:
                    rejected.append(DecisionRecord(
                        sent_id, sent_text,
                        c.head_text, c.tail_text, c.head_type, c.tail_type,
                        c.pattern,
                        marked, rel, conf, margin_val, topk,
                        "REJECT", "reject_conf_threshold",
                    ))
                    continue

                # Gate 2: margin
                if margin_val is not None and margin_val < args.min_margin:
                    rejected.append(DecisionRecord(
                        sent_id, sent_text,
                        c.head_text, c.tail_text, c.head_type, c.tail_type,
                        c.pattern,
                        marked, rel, conf, margin_val, topk,
                        "REJECT", "reject_margin",
                    ))
                    continue

                # Gate 3: schema/type gating
                if not passes_schema(rel, c.head_type, c.tail_type):
                    rejected.append(DecisionRecord(
                        sent_id, sent_text,
                        c.head_text, c.tail_text, c.head_type, c.tail_type,
                        c.pattern,
                        marked, rel, conf, margin_val, topk,
                        "REJECT", "reject_schema",
                    ))
                    continue

                # Gate 4: trigger gate (optional)
                if args.use_trigger_gate and not passes_trigger_gate(rel, sent_text):
                    rejected.append(DecisionRecord(
                        sent_id, sent_text,
                        c.head_text, c.tail_text, c.head_type, c.tail_type,
                        c.pattern,
                        marked, rel, conf, margin_val, topk,
                        "REJECT", "reject_trigger",
                    ))
                    continue

                accepted.append(DecisionRecord(
                    sent_id, sent_text,
                    c.head_text, c.tail_text, c.head_type, c.tail_type,
                    c.pattern,
                    marked, rel, conf, margin_val, topk,
                    "ACCEPT", "accept",
                ))

        print("\nAccepted triples:")
        for r in accepted:
            print(f"({r.head}, {r.pred_rel}, {r.tail})  conf={r.conf:.6f}  via={r.pattern}")
        print(f"\nTotal accepted: {len(accepted)}")
        print(f"Total rejected: {len(rejected)}")

        if args.log_all:
            print_grouped(rejected, log_limit=args.log_limit, title="REJECTED CANDIDATES BY REASON")

    if args.interactive:
        print("\nEnter text (blank line to exit):")
        while True:
            inp = input("> ").strip()
            if not inp:
                break
            run_once(inp)


if __name__ == "__main__":
    main()