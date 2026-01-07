import spacy
from typing import Dict, List, Optional, Tuple, Set

nlp = spacy.load("en_core_web_trf")

TEXTS = [
    "Paul and Anna live in Melbourne.",
    "Paul live in Melbourne, so does Anna.",
    "Both Paul and Anna live together in Melbourne.",
    "Paul and Anna live in Melbourne and Vietnam.",
    "Apple acquired Beats for $3 billion in 2014.",
    "Beats was acquired by Apple for $3 billion in 2014.",
]

REL_PREP_MAP = {
    "in": "in",
    "at": "in",
    "on": "on",
    "of": "of",
    "for": "for",
    "from": "from",
    "to": "to",
    "by": "by",
    "with": "with",
}

EVENT_VERBS = {"acquire", "buy", "purchase", "merge", "acquisition"}


def build_token_to_ent_text(sent) -> Dict[int, str]:
    ent_map: Dict[int, str] = {}
    for ent in sent.ents:
        for tok in ent:
            ent_map[tok.i] = ent.text
    return ent_map


def get_ent_or_token(ent_map: Dict[int, str], tok) -> str:
    return ent_map.get(tok.i, tok.text)


def expand_conj(token) -> List:
    """
    Given a token, return a list containing:
      - the token itself
      - any tokens connected via coordination (conj)
    Example: "Paul and Anna" -> [Paul, Anna]
    """
    items = [token]
    for child in token.children:
        if child.dep_ == "conj":
            items.append(child)
    return items


def find_subjects(verb_tok, ent_map: Dict[int, str]) -> Tuple[List[str], Optional[str]]:
    """
    Return (subjects, subject_dep_type) where dep_type is nsubj or nsubjpass.
    Handles conjunction: "Paul and Anna live ..."
    """
    for child in verb_tok.children:
        if child.dep_ in ("nsubj", "nsubjpass"):
            subj_tokens = expand_conj(child)
            subjects = [get_ent_or_token(ent_map, t) for t in subj_tokens]
            # de-duplicate while preserving order
            seen: Set[str] = set()
            subjects_unique = []
            for s in subjects:
                if s not in seen:
                    seen.add(s)
                    subjects_unique.append(s)
            return subjects_unique, child.dep_
    return [], None


def find_direct_objects(verb_tok, ent_map: Dict[int, str]) -> List[str]:
    """
    Return list of direct objects (dobj), expanding conjunction if present.
    Example: "acquired Beats and Pixar" -> [Beats, Pixar]
    """
    for child in verb_tok.children:
        if child.dep_ == "dobj":
            obj_tokens = expand_conj(child)
            objs = [get_ent_or_token(ent_map, t) for t in obj_tokens]
            # de-duplicate
            seen: Set[str] = set()
            out = []
            for o in objs:
                if o not in seen:
                    seen.add(o)
                    out.append(o)
            return out
    return []


def find_prep_pobj_pairs(head_tok, ent_map: Dict[int, str]) -> List[Tuple[str, str]]:
    """
    Find (prep, pobj) pairs under a predicate, expanding conjunctions on pobj.
    Example: "in Melbourne and Vietnam" -> [("in","Melbourne"), ("in","Vietnam")]
    """
    pairs: List[Tuple[str, str]] = []
    for child in head_tok.children:
        if child.dep_ == "prep":
            prep = child.lemma_.lower()
            for grand in child.children:
                if grand.dep_ == "pobj":
                    pobj_tokens = expand_conj(grand)
                    for t in pobj_tokens:
                        obj_text = get_ent_or_token(ent_map, t)
                        pairs.append((prep, obj_text))
    return pairs


def extract_relations_flat(sent) -> List[dict]:
    """
    Extract flat triples with conjunction handling.
    Patterns:
      A) active SVO:      nsubj + VERB + dobj
      A2) passive by:     nsubjpass + VERB + (by -> pobj)  => reverse to semantic SVO
      B) predicate+prep:  subject + verb_prep + pobj
    """
    ent_map = build_token_to_ent_text(sent)
    triples: List[dict] = []

    for tok in sent:
        if tok.pos_ != "VERB":
            continue

        subjects, subj_dep = find_subjects(tok, ent_map)
        dobjs = find_direct_objects(tok, ent_map)
        prep_pairs = find_prep_pobj_pairs(tok, ent_map)

        # Pattern A: active SVO (supports multiple subjects/objects)
        if subj_dep == "nsubj" and subjects and dobjs:
            for s in subjects:
                for o in dobjs:
                    triples.append({
                        "subject": s,
                        "relation": tok.lemma_,
                        "object": o,
                        "pattern": "A_active_SVO",
                    })

        # Pattern A2: passive "by" agent, reverse to semantic SVO
        # "Beats was acquired by Apple and Google" -> (Apple acquire Beats), (Google acquire Beats)
        if subj_dep == "nsubjpass" and subjects:
            # subjects here are the passive subjects (things being acted on), e.g. Beats
            passive_targets = subjects
            agents: List[str] = []
            for prep, obj_text in prep_pairs:
                if prep == "by":
                    agents.append(obj_text)

            # agents might also have conjunction already expanded in prep_pairs
            if agents:
                for agent in agents:
                    for target in passive_targets:
                        triples.append({
                            "subject": agent,
                            "relation": tok.lemma_,
                            "object": target,
                            "pattern": "A_passive_by",
                        })

        # Pattern B: predicate + prep/pobj relations (supports multiple subjects and multiple pobj)
        # Example: "Paul and Anna live in Melbourne and Vietnam"
        if subjects:
            for s in subjects:
                for prep, obj_text in prep_pairs:
                    rel_suffix = REL_PREP_MAP.get(prep, prep)
                    triples.append({
                        "subject": s,
                        "relation": f"{tok.lemma_}_{rel_suffix}",
                        "object": obj_text,
                        "pattern": "B_predicate_prep",
                    })

    return triples


def build_event_nodes(triples: List[dict]) -> Dict[str, List[dict]]:
    """
    Event-centric conversion for certain verbs (EVENT_VERBS).
    With conjunction handling, there may be multiple core (A,B) pairs.
    We'll create an event per (verb, subject, object) core triple, then attach modifiers
    that share the same subject and verb (e.g., acquire_for, acquire_in).
    """
    edges: List[dict] = []
    events: List[dict] = []
    used = [False] * len(triples)

    event_counter = 0

    for i, t in enumerate(triples):
        rel = t["relation"]
        base = rel.split("_")[0]

        if base not in EVENT_VERBS:
            continue

        # Core triple is exactly (subject, base, object)
        if rel == base:
            event_counter += 1
            event_id = f"{base}_event_{event_counter}"
            events.append({"event_id": event_id, "type": base})

            edges.append({"source": event_id, "relation": "subject", "target": t["subject"]})
            edges.append({"source": event_id, "relation": "object", "target": t["object"]})
            used[i] = True

            # Attach modifiers: base_for, base_in, etc. from same subject
            for j, t2 in enumerate(triples):
                if used[j]:
                    continue
                if t2["subject"] != t["subject"]:
                    continue
                if not t2["relation"].startswith(base + "_"):
                    continue

                suffix = t2["relation"].split("_", 1)[1]
                edges.append({"source": event_id, "relation": suffix, "target": t2["object"]})
                used[j] = True

    # Keep remaining triples as edges
    for i, t in enumerate(triples):
        if used[i]:
            continue
        edges.append({"source": t["subject"], "relation": t["relation"], "target": t["object"]})

    return {"events": events, "edges": edges}


def extract(text: str, use_event_nodes: bool = True) -> Dict[str, List[dict]]:
    doc = nlp(text)
    all_triples: List[dict] = []

    for sent in doc.sents:
        all_triples.extend(extract_relations_flat(sent))

    if use_event_nodes:
        return build_event_nodes(all_triples)

    return {
        "events": [],
        "edges": [{"source": t["subject"], "relation": t["relation"], "target": t["object"]} for t in all_triples],
    }


if __name__ == "__main__":
    for text in TEXTS:
        print("\n" + "=" * 80)
        print("TEXT:", text)
        out = extract(text, use_event_nodes=True)

        if out["events"]:
            print("\nEvents:")
            for e in out["events"]:
                print(e)

        print("\nEdges:")
        for e in out["edges"]:
            print(e)