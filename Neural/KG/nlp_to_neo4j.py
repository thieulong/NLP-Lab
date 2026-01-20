#!/usr/bin/env python3
"""
Text -> triples -> Neo4j

Pipeline:
  text -> spaCy (sents + NER) -> candidate pairs (dependency + limited fallback)
  -> SpanBERT classifier -> gates -> ACCEPT/REJECT logs
  -> POST-ACCEPT DEDUP (head, rel, tail)
  -> wipe Neo4j -> MERGE nodes -> MERGE typed relationships

Run:
  python Neural/KG/text_to_neo4j.py --model_dir Neural/RE/models/spanbert_nyt_re --interactive --log_all --write_neo4j

Neo4j env vars (Aura):
  export NEO4J_URI="neo4j+s://<id>.databases.neo4j.io"
  export NEO4J_USERNAME="neo4j"      # OR NEO4J_USER
  export NEO4J_PASSWORD="..."
  export NEO4J_DATABASE="neo4j"      # optional

IMPORTANT:
- Your model has no 'no_relation'. So we must control noise via pair proposal + gating.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import spacy

try:
    from neo4j import GraphDatabase
except Exception:
    GraphDatabase = None


# -------------------------
# Config: schema + thresholds + triggers
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
    # children is very noisy without explicit cues
    "/people/person/children": 0.995,
}

REL_TRIGGERS: Dict[str, List[str]] = {
    "/business/company/place_founded": ["founded", "headquartered", "based", "incorporated"],
    "/business/company/founders": ["founded by", "co-founded", "cofounder", "founder"],
    "/business/person/company": ["works at", "joined", "ceo", "executive", "employee", "employed", "president"],
    "/business/company/major_shareholders": ["major shareholder", "stake", "owns", "ownership"],
    "/business/company_shareholder/major_shareholder_of": ["major shareholder", "stake", "owns", "ownership"],
    "/location/country/capital": ["capital"],
    "/people/person/place_of_birth": ["born"],
    "/people/person/place_lived": ["lives in", "lived in", "resides in", "resident"],
    "/people/person/nationality": ["citizen", "citizenship", "nationality"],
    "/people/person/children": ["son", "daughter", "children", "child"],
}

# Relations that are NOT allowed from fallback unless trigger matches
NO_FALLBACK_UNLESS_TRIGGER = {
    "/people/person/children",
    "/business/company/major_shareholders",
    "/business/company_shareholder/major_shareholder_of",
    "/location/country/capital",
}

FALLBACK_PATTERN_PREFIX = "P0_fallback_window"
FALLBACK_EXTRA_CONF = 0.02  # tighten threshold for fallback-derived accepts


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


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def mark_two_spans(sent_text: str, span1: Tuple[int, int], span2: Tuple[int, int]) -> str:
    (a1, b1), (a2, b2) = span1, span2
    if a1 == a2 and b1 == b2:
        return ""
    if a1 < a2:
        first = ("E1", a1, b1); second = ("E2", a2, b2)
    else:
        first = ("E2", a2, b2); second = ("E1", a1, b1)

    tag1, s1, e1 = first
    tag2, s2, e2 = second

    out = sent_text
    out = out[:e2] + f"[/{tag2}]" + out[e2:]
    out = out[:s2] + f"[{tag2}]" + out[s2:]
    out = out[:e1] + f"[/{tag1}]" + out[e1:]
    out = out[:s1] + f"[{tag1}]" + out[s1:]
    return out


def rel_to_neo4j_type(rel: str) -> str:
    t = rel.strip().strip("/")
    t = re.sub(r"[^A-Za-z0-9_]+", "_", t).upper()
    if not t or not re.match(r"^[A-Z]", t):
        t = "REL_" + t
    return t


def passes_schema(rel: str, head_type: str, tail_type: str) -> bool:
    if rel not in REL_SCHEMA:
        return True
    return any((head_type == h and tail_type == t) for (h, t) in REL_SCHEMA[rel])


def passes_trigger(rel: str, sentence_text: str) -> bool:
    triggers = REL_TRIGGERS.get(rel)
    if not triggers:
        return True
    s = sentence_text.lower()
    return any(trig in s for trig in triggers)


def get_rel_threshold(rel: str, default_thr: float) -> float:
    return REL_THRESH.get(rel, default_thr)


# -------------------------
# Data classes
# -------------------------

@dataclass
class CandidatePair:
    head_text: str
    tail_text: str
    head_type: str
    tail_type: str
    head_span: Tuple[int, int]
    tail_span: Tuple[int, int]
    pattern: str


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
    margin: Optional[float]
    topk: List[Tuple[str, float]]
    decision: str
    reason: str


# -------------------------
# Model label decode
# -------------------------

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
# Pair proposal
# -------------------------

def propose_pairs(sent, window_k: int = 2) -> List[CandidatePair]:
    """
    Dependency patterns + limited fallback window.

    window_k=2 is conservative; larger window increases noise a lot.
    """

    tok2ent: Dict[int, Tuple[str, str, Tuple[int, int]]] = {}
    ents_in_sent: List[Tuple[str, str, Tuple[int, int]]] = []

    for ent in sent.ents:
        t = coarse_ent_type(ent.label_)
        span = (ent.start_char - sent.start_char, ent.end_char - sent.start_char)
        ents_in_sent.append((norm_text(ent.text), t, span))
        for tok in ent:
            tok2ent[tok.i] = (norm_text(ent.text), t, span)

    def ent_from_token(tok) -> Optional[Tuple[str, str, Tuple[int, int]]]:
        return tok2ent.get(tok.i)

    def first_ent_in_subtree(tok) -> Optional[Tuple[str, str, Tuple[int, int]]]:
        for t in tok.subtree:
            if t.i in tok2ent:
                return tok2ent[t.i]
        return None

    def ent_or_subtree(tok) -> Optional[Tuple[str, str, Tuple[int, int]]]:
        return ent_from_token(tok) or first_ent_in_subtree(tok)

    def add_pair(out: List[CandidatePair], h, t, pattern: str) -> None:
        h_text, h_type, h_span = h
        t_text, t_type, t_span = t
        if not h_text or not t_text or h_text == t_text:
            return
        out.append(CandidatePair(h_text, t_text, h_type, t_type, h_span, t_span, pattern))

    cands: List[CandidatePair] = []

    # P1: SVO
    for tok in sent:
        if tok.pos_ != "VERB":
            continue
        subj = None
        obj = None
        for child in tok.children:
            if child.dep_ == "nsubj":
                subj = ent_or_subtree(child)
            elif child.dep_ == "dobj":
                obj = ent_or_subtree(child)
        if subj and obj:
            add_pair(cands, subj, obj, "P1_SVO")

    # P2: predicate + prep(pobj)
    for tok in sent:
        if tok.pos_ not in ("VERB", "ADJ"):
            continue
        subj = None
        for child in tok.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                subj = ent_or_subtree(child)
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

    # P4: possessive-copular: "Russia's capital is Moscow"
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

    # P0 fallback window (limited)
    # Connect only nearby entity mentions; add both directions.
    if len(ents_in_sent) >= 2:
        for i in range(len(ents_in_sent)):
            for j in range(i + 1, min(len(ents_in_sent), i + 1 + window_k)):
                e1 = ents_in_sent[i]
                e2 = ents_in_sent[j]
                add_pair(cands, e1, e2, "P0_fallback_window")
                add_pair(cands, e2, e1, "P0_fallback_window_rev")

    # Dedup candidates
    seen = set()
    uniq: List[CandidatePair] = []
    for c in cands:
        key = (c.head_span, c.tail_span, c.pattern)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


# -------------------------
# Prediction
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


def print_grouped(records: List[DecisionRecord], limit: int) -> None:
    print("\n" + "=" * 100)
    print("REJECTED CANDIDATES BY REASON")
    print("=" * 100)

    groups: Dict[str, List[DecisionRecord]] = {}
    for r in records:
        groups.setdefault(r.reason, []).append(r)

    for reason, items in sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True):
        print(f"{reason:32s} count={len(items)}")

    for reason, items in sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True):
        print("\n" + "-" * 100)
        print(f"REASON: {reason} (showing up to {limit})")
        for r in items[:limit]:
            print(f"[SENT {r.sent_id}] {r.sentence}")
            print(f"PAIR: {r.head} ({r.head_type}) -> {r.tail} ({r.tail_type}) via={r.pattern}")
            print(f"PRED: {r.pred_rel} conf={r.conf:.6f} margin={None if r.margin is None else round(r.margin, 6)}")
            print(f"TOPK: {[(lab, round(p,6)) for lab,p in r.topk]}")
            print()


# -------------------------
# Neo4j helpers
# -------------------------

def read_neo4j_creds(args) -> Tuple[str, str, str, str]:
    uri = (args.neo4j_uri or os.getenv("NEO4J_URI", "")).strip()
    user = (args.neo4j_user or os.getenv("NEO4J_USER", "") or os.getenv("NEO4J_USERNAME", "")).strip()
    pwd = (args.neo4j_password or os.getenv("NEO4J_PASSWORD", "")).strip()
    db = (args.neo4j_database or os.getenv("NEO4J_DATABASE", "neo4j")).strip()
    return uri, user, pwd, db


def wipe_neo4j(driver, database: str) -> None:
    with driver.session(database=database) as session:
        session.run("MATCH (n) DETACH DELETE n")


def write_triples_neo4j(driver, database: str, triples: List[Tuple[str, str, str, float, str, str, str, str]]) -> None:
    """
    triples: (head, rel_label, tail, conf, sentence, pattern, head_type, tail_type)
    Writes relationship type derived from rel_label and stores rel_label as property.
    """
    with driver.session(database=database) as session:
        for h, rel, t, conf, sent, pattern, htype, ttype in triples:
            rel_type = rel_to_neo4j_type(rel)
            cypher = f"""
            MERGE (a:Entity {{name: $h}})
              ON CREATE SET a.type = $htype
              ON MATCH  SET a.type = COALESCE(a.type, $htype)
            MERGE (b:Entity {{name: $t}})
              ON CREATE SET b.type = $ttype
              ON MATCH  SET b.type = COALESCE(b.type, $ttype)
            MERGE (a)-[r:{rel_type}]->(b)
            SET r.rel_label = $rel,
                r.confidence = $conf,
                r.pattern = $pattern,
                r.sentence = $sent
            """
            session.run(
                cypher,
                h=h, t=t, rel=rel, conf=float(conf),
                sent=sent, pattern=pattern, htype=htype, ttype=ttype
            )


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
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--max_length", type=int, default=256)

    ap.add_argument("--fallback_window_k", type=int, default=2)

    ap.add_argument("--log_all", action="store_true")
    ap.add_argument("--log_limit", type=int, default=10)

    # Neo4j
    ap.add_argument("--write_neo4j", action="store_true")
    ap.add_argument("--neo4j_uri", type=str, default="")
    ap.add_argument("--neo4j_user", type=str, default="")
    ap.add_argument("--neo4j_password", type=str, default="")
    ap.add_argument("--neo4j_database", type=str, default="")

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
        doc = nlp(raw_text)

        accepted: List[DecisionRecord] = []
        rejected: List[DecisionRecord] = []

        for sent_id, sent in enumerate(doc.sents):
            sent_text = sent.text
            cands = propose_pairs(sent, window_k=args.fallback_window_k)

            for c in cands:
                marked = mark_two_spans(sent_text, c.head_span, c.tail_span)
                if not marked:
                    rejected.append(DecisionRecord(
                        sent_id, sent_text, c.head_text, c.tail_text,
                        c.head_type, c.tail_type, c.pattern,
                        "", 0.0, None, [], "REJECT", "reject_mark_failed"
                    ))
                    continue

                pred_id, conf, topk_ids = predict_one(
                    marked, model=model, tokenizer=tokenizer, device=device,
                    top_k=args.top_k, max_length=args.max_length
                )
                rel = id_to_label(model, pred_id)
                topk = [(id_to_label(model, i), p) for (i, p) in topk_ids]
                margin = None
                if len(topk_ids) >= 2:
                    margin = topk_ids[0][1] - topk_ids[1][1]

                # Gate: schema
                if not passes_schema(rel, c.head_type, c.tail_type):
                    rejected.append(DecisionRecord(
                        sent_id, sent_text, c.head_text, c.tail_text,
                        c.head_type, c.tail_type, c.pattern,
                        rel, conf, margin, topk, "REJECT", "reject_schema"
                    ))
                    continue

                # Gate: confidence (+ stricter for fallback)
                thr = get_rel_threshold(rel, args.default_threshold)
                if c.pattern.startswith(FALLBACK_PATTERN_PREFIX):
                    thr = min(0.999, thr + FALLBACK_EXTRA_CONF)
                if conf < thr:
                    rejected.append(DecisionRecord(
                        sent_id, sent_text, c.head_text, c.tail_text,
                        c.head_type, c.tail_type, c.pattern,
                        rel, conf, margin, topk, "REJECT", "reject_conf_threshold"
                    ))
                    continue

                # Gate: margin
                if margin is not None and margin < args.min_margin:
                    rejected.append(DecisionRecord(
                        sent_id, sent_text, c.head_text, c.tail_text,
                        c.head_type, c.tail_type, c.pattern,
                        rel, conf, margin, topk, "REJECT", "reject_margin"
                    ))
                    continue

                # Gate: fallback restriction for noisy relations
                if c.pattern.startswith(FALLBACK_PATTERN_PREFIX) and rel in NO_FALLBACK_UNLESS_TRIGGER:
                    if not passes_trigger(rel, sent_text):
                        rejected.append(DecisionRecord(
                            sent_id, sent_text, c.head_text, c.tail_text,
                            c.head_type, c.tail_type, c.pattern,
                            rel, conf, margin, topk, "REJECT", "reject_fallback_needs_trigger"
                        ))
                        continue

                # Gate: trigger (always enforce for children and shareholders-ish)
                if rel in NO_FALLBACK_UNLESS_TRIGGER:
                    if not passes_trigger(rel, sent_text):
                        rejected.append(DecisionRecord(
                            sent_id, sent_text, c.head_text, c.tail_text,
                            c.head_type, c.tail_type, c.pattern,
                            rel, conf, margin, topk, "REJECT", "reject_trigger"
                        ))
                        continue

                accepted.append(DecisionRecord(
                    sent_id, sent_text, c.head_text, c.tail_text,
                    c.head_type, c.tail_type, c.pattern,
                    rel, conf, margin, topk, "ACCEPT", "accept"
                ))

        # -------------------------
        # POST-ACCEPT DEDUP (fixes duplicates)
        # Prefer:
        #   1) non-fallback patterns over fallback
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
            # lower is better because fallback=1 is worse, and -conf smaller means higher conf
            if cand_rank < cur_rank:
                best[key] = r

        accepted_dedup = list(best.values())
        accepted_dedup.sort(key=lambda r: (-r.conf, r.pred_rel, r.head, r.tail))

        print("\nAccepted triples:")
        for r in accepted_dedup:
            print(f"({r.head}, {r.pred_rel}, {r.tail})  conf={r.conf:.6f}  via={r.pattern}")
        print(f"\nTotal accepted: {len(accepted_dedup)}")
        print(f"Total rejected: {len(rejected)}")

        if args.log_all:
            print_grouped(rejected, limit=args.log_limit)

        # -------------------------
        # Neo4j write (wipe at start, as requested)
        # -------------------------
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
            write_triples_neo4j(driver, database=db, triples=triples_for_db)
            driver.close()
            print("Neo4j write complete.")

            # Browser link (Aura-style)
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