#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import spacy
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from candidate_pairs import propose_pairs
from config import (
    FALLBACK_EXTRA_CONF,
    FALLBACK_PATTERN_PREFIX,
    NO_FALLBACK_UNLESS_TRIGGER,
    REL_THRESH,
    TRIGGER_RELATIONS,
    DEFAULT_VERIFIER_MODEL,
)
from gating import get_rel_threshold, passes_schema, passes_trigger
from logging_utils import print_grouped
from model_re import device_auto, id_to_label, predict_one
from neo4j_io import GraphDatabase, read_neo4j_creds, wipe_neo4j, write_triples_neo4j
from records import DecisionRecord
from spacy_utils import mark_two_spans
from text_utils import normalize_quotes
from thresholds import load_thresholds_json
from verifier import build_hypothesis, verifier_entailment_prob, load_verifier


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
