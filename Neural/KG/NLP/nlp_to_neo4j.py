#!/usr/bin/env python3
"""
nlp_to_neo4j.py

End-to-end: raw text -> KG triples -> Neo4j

Pipeline (NLP-based, non-LLM):
  1) spaCy: sentence split + NER
  2) candidate pair proposal (dependency patterns + conservative fallback window)
  3) SpanBERT NYT-RE classifier (with optional temperature scaling)
  4) Quality control gates:
       - no_relation abstain
       - schema/type gating (canonical direction)
       - per-relation confidence thresholds (optionally loaded from thresholds.json)
       - top1-top2 margin
       - trigger gating for high-risk relations (children/shareholders etc.)
       - fallback-only restrictions (fallback pairs must be EXTRA confident)
  5) Post-processing:
       - accept dedup by (head, rel, tail), prefer non-fallback + higher confidence
  6) Neo4j:
       - wipe database (optional)
       - MERGE nodes + MERGE typed relationships (type derived from relation string)

Run (use built-in sample paragraph):
  python Neural/KG/nlp_to_neo4j.py \
    --model_dir Neural/RE/models/spanbert_nyt_re_norel \
    --thresholds_json Neural/RE/benchmarks/spanbert_norel_valid/thresholds.json \
    --no_relation_label no_relation

Run + write to Neo4j (credentials via env vars):
  export NEO4J_URI='neo4j+s://...databases.neo4j.io'
  export NEO4J_USERNAME='neo4j'
  export NEO4J_PASSWORD='...'
  python Neural/KG/nlp_to_neo4j.py \
    --model_dir Neural/RE/models/spanbert_nyt_re_norel \
    --thresholds_json Neural/RE/benchmarks/spanbert_norel_valid/thresholds.json \
    --no_relation_label no_relation \
    --write_neo4j
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from functools import lru_cache
import torch
import torch.nn.functional as F

import spacy

try:
    from neo4j import GraphDatabase  # type: ignore
except Exception:
    GraphDatabase = None  # type: ignore


# ======================================================================================
# Config
# ======================================================================================

# Canonical direction + coarse entity type signatures (head_type, tail_type)
# spaCy coarse types: PERSON, ORG, GPE, LOC, NORP, FAC, etc.
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

# Default thresholds (will be overridden by --thresholds_json if provided)
REL_THRESH: Dict[str, float] = {
    "/location/location/contains": 0.99,
    "/people/person/place_lived": 0.95,
    "/people/person/place_of_birth": 0.95,
    "/people/deceased_person/place_of_death": 0.95,
    "/location/administrative_division/country": 0.95,
    "/location/country/administrative_divisions": 0.95,

    "/location/country/capital": 0.90,
    "/business/company/place_founded": 0.90,
    "/business/company/founders": 0.85,
    "/business/person/company": 0.90,
    "/people/person/nationality": 0.90,
    "/business/company/major_shareholders": 0.90,
    "/business/company_shareholder/major_shareholder_of": 0.90,
    "/people/person/children": 0.95,
}

# Relations that are especially error-prone without explicit lexical cues.
# If predicted, require sentence-level triggers.
TRIGGER_RELATIONS = {
    "/people/person/children",
    "/business/company/major_shareholders",
    "/business/company_shareholder/major_shareholder_of",
}

REL_TRIGGERS: Dict[str, List[str]] = {
    "/business/company/place_founded": ["founded", "headquartered", "based", "incorporated"],
    "/business/company/founders": ["founded by", "co-founded", "founder", "cofounder"],
    "/business/person/company": ["works at", "joined", "ceo", "executive", "employee", "employed", "president"],
    "/business/company/major_shareholders": ["major shareholder", "major shareholders", "stake", "owns", "ownership"],
    "/business/company_shareholder/major_shareholder_of": ["major shareholder", "stake", "owns", "ownership"],
    "/location/country/capital": ["capital"],
    "/people/person/place_of_birth": ["born"],
    "/people/person/place_lived": ["lives in", "lived in", "resides in", "resident", "moved to"],
    "/people/person/nationality": ["nationality", "citizen", "citizenship"],
    "/people/person/children": ["son", "daughter", "child", "children", "father", "mother", "parents"],
}

# Fallback window pairs are noisy: demand extra confidence and/or trigger.
FALLBACK_PATTERN_PREFIX = "P0_fallback_window"
FALLBACK_EXTRA_CONF = 0.03  # add to per-rel threshold, capped below

# Prevent some relations from being accepted from fallback unless triggers are present.
NO_FALLBACK_UNLESS_TRIGGER = set(TRIGGER_RELATIONS)

# -------------------------
# Binary verifier (NLI-based)
# -------------------------

DEFAULT_VERIFIER_MODEL = "roberta-large-mnli"  # strong general-purpose NLI

REL_TO_TEMPLATE = {
    "/business/company/place_founded": "{h} was founded in {t}.",
    "/business/company/founders": "{t} founded {h}.",
    "/business/person/company": "{h} is associated with the company {t}.",
    "/business/company/major_shareholders": "{t} is a major shareholder of {h}.",
    "/business/company_shareholder/major_shareholder_of": "{h} is a major shareholder of {t}.",

    "/location/location/contains": "{h} contains {t}.",
    "/location/neighborhood/neighborhood_of": "{h} is a neighborhood of {t}.",
    "/location/country/capital": "{t} is the capital of {h}.",
    "/location/administrative_division/country": "{h} is in the country {t}.",
    "/location/country/administrative_divisions": "{t} is an administrative division of {h}.",

    "/people/person/place_lived": "{h} has lived in {t}.",
    "/people/person/place_of_birth": "{h} was born in {t}.",
    "/people/deceased_person/place_of_death": "{h} died in {t}.",
    "/people/person/nationality": "{h} has nationality {t}.",
    "/people/person/children": "{t} is a child of {h}.",
}

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


# ======================================================================================
# Utilities
# ======================================================================================

def device_auto() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore
        return torch.device("mps")
    return torch.device("cpu")


def coarse_ent_type(spacy_label: str) -> str:
    # Keep a stable coarse mapping; you can refine later if needed.
    if spacy_label in {"PERSON", "ORG", "GPE", "LOC", "NORP", "FAC"}:
        return spacy_label
    return spacy_label


def normalize_quotes(s: str) -> str:
    return s.replace("’", "'").replace("“", '"').replace("”", '"')


def mark_two_spans(sent_text: str, span1: Tuple[int, int], span2: Tuple[int, int]) -> str:
    """
    Insert [E1]...[/E1], [E2]...[/E2] around character spans.
    Spans are relative to the sentence text.
    """
    (a1, b1), (a2, b2) = span1, span2
    if a1 == a2 and b1 == b2:
        return ""

    # Ensure deterministic tagging order for insertion
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


@dataclass(frozen=True)
class CandidatePair:
    head_text: str
    tail_text: str
    head_type: str
    tail_type: str
    head_span: Tuple[int, int]
    tail_span: Tuple[int, int]
    pattern: str


def id_to_label(model: Any, pred_id: int) -> str:
    id2label = model.config.id2label
    if isinstance(id2label, dict):
        if pred_id in id2label:
            return str(id2label[pred_id])
        s = str(pred_id)
        if s in id2label:
            return str(id2label[s])
        # last resort: scan keys that might be strings of ints
        for k, v in id2label.items():
            try:
                if int(k) == pred_id:
                    return str(v)
            except Exception:
                pass
        raise KeyError(f"pred_id={pred_id} not found in id2label keys (sample)={list(id2label)[:10]}")
    # list/tuple
    return str(id2label[pred_id])


def load_thresholds_json(path: Path) -> Dict[str, float]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict) and "thresholds" in obj and isinstance(obj["thresholds"], dict):
        obj = obj["thresholds"]
    if not isinstance(obj, dict):
        raise ValueError("thresholds_json must be {rel: thr} or {'thresholds': {rel: thr}}")
    out: Dict[str, float] = {}
    for k, v in obj.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            continue
    return out


def get_rel_threshold(rel: str, default_threshold: float) -> float:
    return REL_THRESH.get(rel, default_threshold)


def passes_schema(rel: str, head_type: str, tail_type: str) -> bool:
    allowed = REL_SCHEMA.get(rel)
    if not allowed:
        return True
    return any((head_type == h and tail_type == t) for (h, t) in allowed)


def passes_trigger(rel: str, sentence_text: str) -> bool:
    triggers = REL_TRIGGERS.get(rel)
    if not triggers:
        return True
    s = sentence_text.lower()
    return any(t in s for t in triggers)


def rel_to_neo4j_type(rel: str) -> str:
    """
    Neo4j relationship types must match: [A-Z_][A-Z0-9_]*
    We'll derive a stable type and also store the original 'rel' as a property.
    """
    r = rel.strip()
    if not r:
        return "RELATED_TO"
    # keep only alnum and underscores
    r = r.strip("/")
    r = re.sub(r"[^0-9A-Za-z_]+", "_", r)
    r = re.sub(r"_+", "_", r).strip("_")
    if not r:
        return "RELATED_TO"
    r = r.upper()
    if not re.match(r"^[A-Z_]", r):
        r = "_" + r
    return r


# ======================================================================================
# Candidate Pair Proposal
# ======================================================================================

def propose_pairs(sent, window_k: int = 2) -> List[CandidatePair]:
    """
    Propose entity pairs likely supported by the sentence structure.

    Patterns:
      P1  SVO:          VERB nsubj -> dobj
      P2  pred+prep:    predicate (VERB/ADJ) with nsubj/nsubjpass and prep/agent->pobj
                        (IMPORTANT: includes passive 'agent' such as "was founded by X")
      P3  copular prep: predicate NOUN/ADJ is ROOT, has 'cop' and nsubj, plus prep->pobj
      P4  poss copular: possessor + (capital/founder/etc) + cop + attr
                        also handles "capital of Russia is Moscow" (prep 'of')
      P5  apposition:   X, CEO Y (appos)
      P6  noun-of:      (role noun) + prep 'of' -> entity, connect subj/possessor -> of_obj
      P7  relcl recovery: who/which/that subject -> antecedent entity
      P8  conj expand:  if X->Y found and X has conjuncts, add conjunct->Y (also for Y)
      P0  fallback:     within-window ordered pairs between entity mentions (noisy)
                        + additional general safety: max token-distance between entity roots
    """

    # --- general fallback safety knob (not dataset-specific) ---
    MAX_FALLBACK_TOKEN_DIST = 18  # reduce if you want even safer fallback

    # token index -> entity tuple for tokens inside an entity span
    tok2ent: Dict[int, Tuple[str, str, Tuple[int, int]]] = {}
    ents = list(sent.ents)

    # span -> (ent_text, coarse_type, span) + root token index for distance gating
    span2ent: Dict[Tuple[int, int], Tuple[str, str, Tuple[int, int]]] = {}
    span2root_i: Dict[Tuple[int, int], int] = {}

    for ent in ents:
        t = coarse_ent_type(ent.label_)
        span = (ent.start_char - sent.start_char, ent.end_char - sent.start_char)
        e_tuple = (ent.text, t, span)
        span2ent[span] = e_tuple
        # root token position within the sentence (token indices are doc-level, but diff still works)
        span2root_i[span] = ent.root.i

        for tok in ent:
            tok2ent[tok.i] = e_tuple

    def ent_from_token(tok) -> Optional[Tuple[str, str, Tuple[int, int]]]:
        return tok2ent.get(tok.i)

    def first_ent_in_subtree(tok) -> Optional[Tuple[str, str, Tuple[int, int]]]:
        for t in tok.subtree:
            e = tok2ent.get(t.i)
            if e:
                return e
        return None

    def ent_or_subtree(tok) -> Optional[Tuple[str, str, Tuple[int, int]]]:
        return ent_from_token(tok) or first_ent_in_subtree(tok)

    def antecedent_entity_for_relpron(tok) -> Optional[Tuple[str, str, Tuple[int, int]]]:
        head = tok.head
        for _ in range(3):
            e = ent_or_subtree(head)
            if e and e[0].lower() not in {"who", "which", "that"}:
                return e
            head = head.head
        return None

    def add_pair(out: List[CandidatePair], h, t, pattern: str) -> None:
        h_text, h_type, h_span = h
        t_text, t_type, t_span = t
        if not h_text or not t_text or h_text == t_text:
            return
        out.append(CandidatePair(
            head_text=h_text,
            tail_text=t_text,
            head_type=h_type,
            tail_type=t_type,
            head_span=h_span,
            tail_span=t_span,
            pattern=pattern,
        ))

    def add_pair_with_conj(out: List[CandidatePair], h, t_tok, pattern: str) -> None:
        """
        Add pair h -> entity(t_tok) if present, and also h -> each conjunct entity of t_tok.
        This is robust for 'by Steve Jobs and Steve Wozniak' cases.
        """
        t_ent = ent_or_subtree(t_tok)
        if t_ent:
            add_pair(out, h, t_ent, pattern)
        # conjunct expansion at token-level (in addition to later P8)
        for conj in t_tok.conjuncts:
            t2 = ent_or_subtree(conj)
            if t2:
                add_pair(out, h, t2, pattern + "+conj")

    cands: List[CandidatePair] = []

    # P1: SVO (active voice only)
    for tok in sent:
        if tok.pos_ != "VERB":
            continue
        subj = None
        obj = None
        for ch in tok.children:
            if ch.dep_ == "nsubj":
                subj = ent_or_subtree(ch)
                if subj and subj[0].lower() in {"who", "which", "that"}:
                    subj = antecedent_entity_for_relpron(ch)
            elif ch.dep_ == "dobj":
                obj = ent_or_subtree(ch)
        if subj and obj:
            add_pair(cands, subj, obj, "P1_SVO")

    # P2: predicate with prep/agent pobj (handles passive: "was founded by X")
    for tok in sent:
        if tok.pos_ not in ("VERB", "ADJ"):
            continue

        subj = None
        for ch in tok.children:
            if ch.dep_ in ("nsubj", "nsubjpass"):
                subj = ent_or_subtree(ch)
                if subj and subj[0].lower() in {"who", "which", "that"}:
                    subj = antecedent_entity_for_relpron(ch)
        if not subj:
            continue

        for ch in tok.children:
            # IMPORTANT: passive agent is often dep_ == "agent"
            if ch.dep_ not in ("prep", "agent"):
                continue

            # Optional: if you want to be stricter, prioritize 'by' as agent
            # but keep it general by allowing all preps/agents.
            # if ch.dep_ == "agent" and ch.lemma_.lower() != "by": continue

            for grand in ch.children:
                if grand.dep_ == "pobj":
                    # Add subj -> pobj and subj -> pobj conjuncts
                    add_pair_with_conj(cands, subj, grand, "P2_pred_prep")

    # P3: copular ROOT NOUN/ADJ with prep pobj
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

    # P4: possessive copular bridging (possessor -> attr)
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

        # also handle "X of Y is Z"
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

    # P5: apposition
    for tok in sent:
        if tok.dep_ != "appos":
            continue
        appos_ent = ent_or_subtree(tok)
        head_ent = ent_or_subtree(tok.head)
        if appos_ent and head_ent:
            add_pair(cands, head_ent, appos_ent, "P5_appos")
            add_pair(cands, appos_ent, head_ent, "P5_appos_rev")

    # P6: noun-of role phrases
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
        if possessor:
            add_pair(cands, possessor, of_obj, "P6_noun_of_poss")

        has_cop = any(ch.dep_ == "cop" for ch in tok.children)
        if has_cop:
            subj = None
            for ch in tok.children:
                if ch.dep_ == "nsubj":
                    subj = ent_or_subtree(ch)
            if subj:
                add_pair(cands, subj, of_obj, "P6_noun_of_cop_subj")

    # Dedup before conj expansion
    seen = set()
    uniq: List[CandidatePair] = []
    for c in cands:
        key = (c.head_span, c.tail_span, c.pattern)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)

    # P8: conjunction expansion (span-based)
    span2tokens: Dict[Tuple[int, int], List[Any]] = {}
    for tok in sent:
        e = ent_from_token(tok)
        if not e:
            continue
        span2tokens.setdefault(e[2], []).append(tok)

    def conjunct_entities(ent_span: Tuple[int, int]) -> List[Tuple[str, str, Tuple[int, int]]]:
        toks = span2tokens.get(ent_span, [])
        out = []
        for tok in toks:
            for conj in tok.conjuncts:
                e2 = ent_or_subtree(conj)
                if e2:
                    out.append(e2)
        seen_sp = set()
        uniq_out = []
        for e2 in out:
            if e2[2] in seen_sp:
                continue
            seen_sp.add(e2[2])
            uniq_out.append(e2)
        return uniq_out

    expanded: List[CandidatePair] = list(uniq)
    for c in uniq:
        for h2 in conjunct_entities(c.head_span):
            expanded.append(CandidatePair(
                head_text=h2[0], head_type=h2[1], head_span=h2[2],
                tail_text=c.tail_text, tail_type=c.tail_type, tail_span=c.tail_span,
                pattern=c.pattern + "+P8_head_conj",
            ))
        for t2 in conjunct_entities(c.tail_span):
            expanded.append(CandidatePair(
                head_text=c.head_text, head_type=c.head_type, head_span=c.head_span,
                tail_text=t2[0], tail_type=t2[1], tail_span=t2[2],
                pattern=c.pattern + "+P8_tail_conj",
            ))

    # Final dedup
    final_seen = set()
    final: List[CandidatePair] = []
    for c in expanded:
        key = (c.head_span, c.tail_span, c.pattern)
        if key in final_seen:
            continue
        final_seen.add(key)
        final.append(c)

    # P0: fallback window between nearby entity mentions (ordered pairs)
    # Safer fallback: require entities to be within MAX_FALLBACK_TOKEN_DIST by root token index
    if window_k and window_k > 0:
        existing_pairs = {(c.head_span, c.tail_span) for c in final}

        ent_items = []
        for ent in ents:
            t = coarse_ent_type(ent.label_)
            span = (ent.start_char - sent.start_char, ent.end_char - sent.start_char)
            root_i = ent.root.i
            ent_items.append((ent.text, t, span, root_i))

        # keep original order of mentions, but skip pairs that are too far apart token-wise
        for i, e1 in enumerate(ent_items):
            for j in range(i + 1, min(len(ent_items), i + 1 + window_k)):
                e2 = ent_items[j]
                dist = abs(e1[3] - e2[3])
                if dist > MAX_FALLBACK_TOKEN_DIST:
                    continue

                # forward
                if (e1[2], e2[2]) not in existing_pairs and e1[0] != e2[0]:
                    final.append(CandidatePair(
                        head_text=e1[0], head_type=e1[1], head_span=e1[2],
                        tail_text=e2[0], tail_type=e2[1], tail_span=e2[2],
                        pattern=f"{FALLBACK_PATTERN_PREFIX}",
                    ))
                    existing_pairs.add((e1[2], e2[2]))

                # reverse
                if (e2[2], e1[2]) not in existing_pairs and e1[0] != e2[0]:
                    final.append(CandidatePair(
                        head_text=e2[0], head_type=e2[1], head_span=e2[2],
                        tail_text=e1[0], tail_type=e1[1], tail_span=e1[2],
                        pattern=f"{FALLBACK_PATTERN_PREFIX}_rev",
                    ))
                    existing_pairs.add((e2[2], e1[2]))

    return final


# ======================================================================================
# SpanBERT Prediction
# ======================================================================================

@torch.no_grad()
def predict_one(
    marked_text: str,
    *,
    model,
    tokenizer,
    device: torch.device,
    top_k: int = 5,
    max_length: int = 256,
    temperature: float = 1.0,
):
    enc = tokenizer(marked_text, truncation=True, max_length=max_length, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc)
    logits = out.logits  # [1, C]
    if temperature and temperature != 1.0:
        logits = logits / float(temperature)
    probs = torch.softmax(logits, dim=-1).squeeze(0)  # [C]

    conf = float(probs.max().item())
    pred_id = int(torch.argmax(probs).item())

    k = min(int(top_k), int(probs.shape[-1]))
    top_probs, top_ids = torch.topk(probs, k=k)
    topk_ids = [(int(i.item()), float(p.item())) for i, p in zip(top_ids, top_probs)]
    return pred_id, conf, topk_ids


# ======================================================================================
# Logging structures
# ======================================================================================

@dataclass
class DecisionRecord:
    sent_id: int
    sentence: str
    head: str
    tail: str
    head_type: str
    tail_type: str
    pattern: str
    pred_rel: str
    conf: float
    decision: str
    reason: str

    margin: Optional[float] = None
    topk: List[Tuple[str, float]] = field(default_factory=list)

    # NEW: verifier logging
    verifier_entailment: Optional[float] = None


def print_grouped(rejected: List[DecisionRecord], limit: int = 10) -> None:
    if not rejected:
        print("\nNo rejected candidates logged.")
        return

    print("\n" + "=" * 100)
    print("REJECTED CANDIDATES BY REASON")
    print("=" * 100)

    groups: Dict[str, List[DecisionRecord]] = {}
    for r in rejected:
        groups.setdefault(r.reason, []).append(r)

    for reason, items in sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True):
        print(f"{reason:32s} count={len(items)}")

    for reason, items in sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True):
        print("\n" + "-" * 100)
        print(f"REASON: {reason} (showing up to {limit})")
        for r in items[:limit]:
            print(f"[SENT {r.sent_id}] {r.sentence}")
            print(f"PAIR: {r.head} ({r.head_type}) -> {r.tail} ({r.tail_type})  via={r.pattern}")
            print(f"PRED: {r.pred_rel}  conf={r.conf:.6f}  margin={None if r.margin is None else round(r.margin,6)}")
            if getattr(r, "verifier_entailment", None) is not None:
                print(f"VERIFIER: entail={r.verifier_entailment:.4f}")
            if r.topk:
                print("TOPK:", [(lab, round(p, 6)) for lab, p in r.topk])
            print()


# ======================================================================================
# Neo4j helpers
# ======================================================================================

def read_neo4j_creds(args) -> Tuple[str, str, str, str]:
    uri = args.neo4j_uri or os.getenv("NEO4J_URI", "")
    user = (
        args.neo4j_user
        or os.getenv("NEO4J_USER", "")
        or os.getenv("NEO4J_USERNAME", "")
    )
    pwd = args.neo4j_password or os.getenv("NEO4J_PASSWORD", "")
    db = args.neo4j_database or os.getenv("NEO4J_DATABASE", "")
    return uri, user, pwd, db


def wipe_neo4j(driver, database: str = "") -> None:
    query = "MATCH (n) DETACH DELETE n"
    with driver.session(database=database or None) as sess:
        sess.run(query)


def write_triples_neo4j(
    driver,
    *,
    triples: List[Tuple[str, str, str, float, str, str, str, str]],
    database: str = "",
) -> None:
    """
    triples: (head, rel, tail, conf, sentence, pattern, head_type, tail_type)
    """
    cypher = """
    MERGE (h:Entity {name: $h})
      ON CREATE SET h.created_at = timestamp()
    SET h.type = $h_type

    MERGE (t:Entity {name: $t})
      ON CREATE SET t.created_at = timestamp()
    SET t.type = $t_type

    WITH h, t
    CALL apoc.merge.relationship(
      h,
      $rel_type,
      {rel: $rel},        // identity properties
      {conf: $conf, evidence: $evidence, pattern: $pattern, updated_at: timestamp()},
      t
    ) YIELD rel
    RETURN rel
    """
    # Note: this uses APOC. Neo4j Aura typically has APOC core available.
    # If APOC is not available, replace with dynamic string relationship creation (more awkward).
    with driver.session(database=database or None) as sess:
        for h, rel, t, conf, evidence, pattern, h_type, t_type in triples:
            sess.run(
                cypher,
                {
                    "h": h,
                    "t": t,
                    "h_type": h_type,
                    "t_type": t_type,
                    "rel": rel,
                    "rel_type": rel_to_neo4j_type(rel),
                    "conf": float(conf),
                    "evidence": evidence,
                    "pattern": pattern,
                },
            )


# ======================================================================================
# Main
# ======================================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default="Neural/RE/models/spanbert_nyt_re")
    ap.add_argument("--spacy_model", type=str, default="en_core_web_trf")
    ap.add_argument("--text", type=str, default="")
    ap.add_argument("--interactive", action="store_true")

    ap.add_argument("--default_threshold", type=float, default=0.90)
    ap.add_argument("--min_margin", type=float, default=0.15)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="Softmax temperature scaling (1.0 = no scaling).")
    ap.add_argument("--thresholds_json", type=str, default="",
                    help="Path to thresholds.json to override REL_THRESH at runtime.")
    ap.add_argument("--no_relation_label", type=str, default="no_relation",
                    help="Label name for abstain class. Use empty string to disable.")
    ap.add_argument("--fallback_window_k", type=int, default=2,
                    help="Entity window size for fallback pairs (0 disables fallback).")
    
    # Verifier (binary supported-vs-not)
    ap.add_argument("--use_verifier", action="store_true", help="Enable NLI-based support verification.")
    ap.add_argument("--verifier_model", type=str, default=DEFAULT_VERIFIER_MODEL)
    ap.add_argument("--verifier_threshold", type=float, default=0.80, help="Min entailment prob to accept.")
    ap.add_argument("--verifier_temperature", type=float, default=1.0, help="Softmax temperature for verifier.")
    ap.add_argument("--verifier_max_length", type=int, default=256)

    ap.add_argument("--verifier_thresholds_json", type=str, default="",
                    help="Optional JSON mapping relation -> verifier entailment threshold (overrides --verifier_threshold per relation).")
    ap.add_argument("--explicit_only", action="store_true",
                    help="Only keep candidates from explicit syntax patterns (drop fallback window P0 and conj expansions).")

    ap.add_argument("--log_all", action="store_true")
    ap.add_argument("--log_limit", type=int, default=10)

    # Neo4j
    ap.add_argument("--write_neo4j", action="store_true")
    ap.add_argument("--neo4j_uri", type=str, default="")
    ap.add_argument("--neo4j_user", type=str, default="")
    ap.add_argument("--neo4j_password", type=str, default="")
    ap.add_argument("--neo4j_database", type=str, default="")

    args = ap.parse_args()

    # Optional: override REL_THRESH from fitted thresholds.json
    if args.thresholds_json:
        p = Path(args.thresholds_json)
        if not p.exists():
            raise FileNotFoundError(f"thresholds_json not found: {p}")
        loaded = load_thresholds_json(p)
        REL_THRESH.update(loaded)
        print(f"Loaded thresholds override from: {p} (n={len(loaded)})")

    # Optional: per-relation verifier thresholds (entailment probability)
    verifier_rel_thr: Dict[str, float] = {}
    if getattr(args, "verifier_thresholds_json", ""):
        vp = Path(args.verifier_thresholds_json)
        if not vp.exists():
            raise FileNotFoundError(f"verifier_thresholds_json not found: {vp}")
        verifier_rel_thr = load_thresholds_json(vp)
        print(f"Loaded verifier thresholds from: {vp} (n={len(verifier_rel_thr)})")

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

    verifier_tokenizer = verifier_model = None
    ent_id = neu_id = con_id = None

    if args.use_verifier:
        print(f"Loading verifier (NLI): {args.verifier_model}")
        verifier_tokenizer, verifier_model, ent_id, neu_id, con_id = load_verifier(args.verifier_model, device)

    def run_once(raw_text: str):
        raw_text = normalize_quotes(raw_text.strip())
        if not raw_text:
            return

        doc = nlp(raw_text)

        accepted: List[DecisionRecord] = []
        rejected: List[DecisionRecord] = []

        for sent_id, sent in enumerate(doc.sents):
            sent_text = normalize_quotes(sent.text)

            cands = propose_pairs(sent, window_k=args.fallback_window_k)

            for c in cands:
                # Explicit-only mode: drop noisy fallback pairs and conj-expanded pairs
                if getattr(args, "explicit_only", False):
                    if c.pattern.startswith(FALLBACK_PATTERN_PREFIX) or "+P8" in c.pattern:
                        rejected.append(DecisionRecord(
                            sent_id=sent_id,
                            sentence=sent_text,
                            head=c.head_text,
                            tail=c.tail_text,
                            head_type=c.head_type,
                            tail_type=c.tail_type,
                            pattern=c.pattern,
                            pred_rel="",
                            conf=0.0,
                            margin=None,
                            topk=[],
                            decision="REJECT",
                            reason="reject_explicit_only",
                        ))
                        continue

                marked = mark_two_spans(sent_text, c.head_span, c.tail_span)
                if not marked:
                    rejected.append(DecisionRecord(
                        sent_id=sent_id,
                        sentence=sent_text,
                        head=c.head_text,
                        tail=c.tail_text,
                        head_type=c.head_type,
                        tail_type=c.tail_type,
                        pattern=c.pattern,
                        pred_rel="",
                        conf=0.0,
                        margin=None,
                        topk=[],
                        decision="REJECT",
                        reason="reject_mark_failed",
                    ))
                    continue

                pred_id, conf, topk_ids = predict_one(
                    marked,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    top_k=args.top_k,
                    max_length=args.max_length,
                    temperature=args.temperature,
                )
                rel = id_to_label(model, pred_id)

                # decode topk for logging
                topk = [(id_to_label(model, i), p) for (i, p) in topk_ids]
                margin_val: Optional[float] = None
                if len(topk_ids) >= 2:
                    margin_val = topk_ids[0][1] - topk_ids[1][1]

                # Gate 0: no_relation abstain
                if args.no_relation_label and rel == args.no_relation_label:
                    rejected.append(DecisionRecord(
                        sent_id=sent_id,
                        sentence=sent_text,
                        head=c.head_text,
                        tail=c.tail_text,
                        head_type=c.head_type,
                        tail_type=c.tail_type,
                        pattern=c.pattern,
                        pred_rel=rel,
                        conf=conf,
                        margin=margin_val,
                        topk=topk,
                        decision="REJECT",
                        reason="reject_no_relation",
                    ))
                    continue

                # Gate 1: schema/type
                if not passes_schema(rel, c.head_type, c.tail_type):
                    rejected.append(DecisionRecord(
                        sent_id=sent_id,
                        sentence=sent_text,
                        head=c.head_text,
                        tail=c.tail_text,
                        head_type=c.head_type,
                        tail_type=c.tail_type,
                        pattern=c.pattern,
                        pred_rel=rel,
                        conf=conf,
                        margin=margin_val,
                        topk=topk,
                        decision="REJECT",
                        reason="reject_schema",
                    ))
                    continue

                # Gate 2: confidence threshold (stricter for fallback)
                thr = get_rel_threshold(rel, args.default_threshold)
                if c.pattern.startswith(FALLBACK_PATTERN_PREFIX):
                    thr = min(0.999, thr + FALLBACK_EXTRA_CONF)
                if conf < thr:
                    rejected.append(DecisionRecord(
                        sent_id=sent_id,
                        sentence=sent_text,
                        head=c.head_text,
                        tail=c.tail_text,
                        head_type=c.head_type,
                        tail_type=c.tail_type,
                        pattern=c.pattern,
                        pred_rel=rel,
                        conf=conf,
                        margin=margin_val,
                        topk=topk,
                        decision="REJECT",
                        reason="reject_conf_threshold",
                    ))
                    continue

                # Gate 3: margin
                if margin_val is not None and margin_val < args.min_margin:
                    rejected.append(DecisionRecord(
                        sent_id=sent_id,
                        sentence=sent_text,
                        head=c.head_text,
                        tail=c.tail_text,
                        head_type=c.head_type,
                        tail_type=c.tail_type,
                        pattern=c.pattern,
                        pred_rel=rel,
                        conf=conf,
                        margin=margin_val,
                        topk=topk,
                        decision="REJECT",
                        reason="reject_margin",
                    ))
                    continue

                # Gate 4: triggers for risky relations (always)
                if rel in TRIGGER_RELATIONS and not passes_trigger(rel, sent_text):
                    rejected.append(DecisionRecord(
                        sent_id=sent_id,
                        sentence=sent_text,
                        head=c.head_text,
                        tail=c.tail_text,
                        head_type=c.head_type,
                        tail_type=c.tail_type,
                        pattern=c.pattern,
                        pred_rel=rel,
                        conf=conf,
                        margin=margin_val,
                        topk=topk,
                        decision="REJECT",
                        reason="reject_trigger",
                    ))
                    continue

                # Gate 5: fallback restriction for risky relations
                if c.pattern.startswith(FALLBACK_PATTERN_PREFIX) and rel in NO_FALLBACK_UNLESS_TRIGGER:
                    if not passes_trigger(rel, sent_text):
                        rejected.append(DecisionRecord(
                            sent_id=sent_id,
                            sentence=sent_text,
                            head=c.head_text,
                            tail=c.tail_text,
                            head_type=c.head_type,
                            tail_type=c.tail_type,
                            pattern=c.pattern,
                            pred_rel=rel,
                            conf=conf,
                            margin=margin_val,
                            topk=topk,
                            decision="REJECT",
                            reason="reject_fallback_needs_trigger",
                        ))
                        continue
                
                # -------------------------
                # Binary verifier gate (NLI entailment)
                # -------------------------
                # -------------------------
                # Binary verifier gate (NLI entailment)
                # -------------------------
                if args.use_verifier:
                    hyp = build_hypothesis(c.head_text, rel, c.tail_text)
                    p_ent = verifier_entailment_prob(
                        premise=sent_text,
                        hypothesis=hyp,
                        verifier_tokenizer=verifier_tokenizer,
                        verifier_model=verifier_model,
                        entailment_id=ent_id,
                        temperature=args.verifier_temperature,
                        device=device,
                        max_length=args.verifier_max_length,
                    )
                    thr_v = verifier_rel_thr.get(rel, args.verifier_threshold)
                    if p_ent < thr_v:
                        rejected.append(DecisionRecord(
                            sent_id=sent_id,
                            sentence=sent_text,
                            head=c.head_text,
                            tail=c.tail_text,
                            head_type=c.head_type,
                            tail_type=c.tail_type,
                            pattern=c.pattern,
                            pred_rel=rel,
                            conf=conf,
                            margin=margin_val,
                            topk=topk,
                            verifier_entailment=p_ent,
                            decision="REJECT",
                            reason=f"reject_verifier_entail<{thr_v:.2f}",
                        ))
                        continue
                else:
                    p_ent = None
# ACCEPT
                accepted.append(DecisionRecord(
                    sent_id=sent_id,
                    sentence=sent_text,
                    head=c.head_text,
                    tail=c.tail_text,
                    head_type=c.head_type,
                    tail_type=c.tail_type,
                    pattern=c.pattern,
                    pred_rel=rel,
                    conf=conf,
                    margin=margin_val,
                    topk=topk,
                    verifier_entailment=p_ent,
                    decision="ACCEPT",
                    reason="accept",
                ))

        # -------------------------
        # POST-ACCEPT DEDUP
        # Prefer:
        #   1) non-fallback patterns
        #   2) higher confidence
        # -------------------------
        def is_fallback(pat: str) -> int:
            return 1 if pat.startswith(FALLBACK_PATTERN_PREFIX) else 0

        best: Dict[Tuple[str, str, str], DecisionRecord] = {}
        for r in accepted:
            key = (r.head, r.pred_rel, r.tail)
            if key not in best:
                best[key] = r
                continue
            cur = best[key]
            cand_rank = (is_fallback(r.pattern), -r.conf)
            cur_rank = (is_fallback(cur.pattern), -cur.conf)
            if cand_rank < cur_rank:
                best[key] = r

        accepted_dedup = list(best.values())
        accepted_dedup.sort(key=lambda r: (-r.conf, r.pred_rel, r.head, r.tail))

        # Print results
        print("\nAccepted triples:")
        for r in accepted_dedup:
            print(f"({r.head}, {r.pred_rel}, {r.tail})  conf={r.conf:.6f}  via={r.pattern}")
        print(f"\nTotal accepted: {len(accepted_dedup)}")
        print(f"Total rejected: {len(rejected)}")

        if args.log_all:
            print_grouped(rejected, limit=args.log_limit)

        # Neo4j write (wipe each run, as requested)
        if args.write_neo4j:
            if GraphDatabase is None:
                raise RuntimeError("neo4j driver not installed. Install: pip install neo4j")

            uri, user, pwd, db = read_neo4j_creds(args)
            if not uri or not user or not pwd:
                raise RuntimeError(
                    "Missing Neo4j credentials. Set env vars NEO4J_URI + (NEO4J_USER or NEO4J_USERNAME) + NEO4J_PASSWORD "
                    "or pass --neo4j_uri/--neo4j_user/--neo4j_password."
                )

            driver = GraphDatabase.driver(uri, auth=(user, pwd))

            print("\nWiping Neo4j database (MATCH (n) DETACH DELETE n)...")
            wipe_neo4j(driver, database=db)
            print("Neo4j wiped.")

            triples_for_db = [
                (r.head, r.pred_rel, r.tail, r.conf, r.sentence, r.pattern, r.head_type, r.tail_type)
                for r in accepted_dedup
            ]
            print(f"\nWriting {len(triples_for_db)} triples to Neo4j...")
            write_triples_neo4j(driver, triples=triples_for_db, database=db)
            driver.close()
            print("Neo4j write complete.")

            # Browser link
            try:
                host = uri.split("://", 1)[1]
                print(f"\nOpen Neo4j Browser: https://{host}/browser/")
            except Exception:
                pass

    if args.interactive:
        print("\nEnter text (blank line to exit):")
        while True:
            inp = input("> ").strip()
            if not inp:
                break
            run_once(inp)
    else:
        if not args.text:
            args.text = (
                "Apple was founded in Cupertino in 1976 by Steve Jobs and Steve Wozniak, a moment that helped establish Silicon Valley as a global technology hub. After joining Apple, Tim Cook later became the company’s chief executive and moved to California, where he has lived for many years. Tesla was founded in Silicon Valley, and Elon Musk eventually became a major shareholder and public face of the company. New York contains Manhattan, and the Upper West Side is a well-known neighborhood within Manhattan that has long attracted writers and artists. Russia’s capital is Moscow, while Kazakhstan’s capital is Astana, reflecting the political centers of the two former Soviet republics. During a recent interview, Elon Musk joked that artificial intelligence might someday run entire governments, a claim that sparked debate online. Steve Jobs admired Pablo Picasso and once said that creativity comes from connecting experiences rather than following rules. Although Apple and Tesla are often compared by investors, the two companies operate in very different industries and markets."
            )
        run_once(args.text)


if __name__ == "__main__":
    main()
