import spacy
from typing import Dict, List, Optional, Tuple

nlp = spacy.load("en_core_web_trf")

TEXTS = [
    "Apple acquired Beats for $3 billion in 2014.",
    "Barack Obama was born in Hawaii.",
    "Google is headquartered in Mountain View.",
]

# Prepositions we often treat as semantic relations in KG edges
# (you can expand later)
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

def build_token_to_ent_text(sent) -> Dict[int, str]:
    """
    Map each token index inside an entity span to the entity text.
    This lets us recover canonical entity strings during pattern matching.
    """
    ent_map: Dict[int, str] = {}
    for ent in sent.ents:
        for tok in ent:
            ent_map[tok.i] = ent.text
    return ent_map

def get_ent_or_token(ent_map: Dict[int, str], tok) -> str:
    """
    Return entity text if token is part of an entity; else token text.
    This lets the extractor still work when objects aren't recognized as NEs.
    """
    return ent_map.get(tok.i, tok.text)

def find_prep_objects(predicate_tok, ent_map: Dict[int, str]) -> List[Tuple[str, str]]:
    """
    For a predicate token (verb/adj), find (prep, pobj) pairs under it.
    Returns list of (prep_lemma, object_text).
    """
    pairs = []
    for child in predicate_tok.children:
        if child.dep_ == "prep":
            prep = child.lemma_.lower()
            for grand in child.children:
                if grand.dep_ == "pobj":
                    obj_text = get_ent_or_token(ent_map, grand)
                    pairs.append((prep, obj_text))
    return pairs

def extract_relations(text: str) -> List[dict]:
    """
    Dependency-based relation extraction baseline.

    Outputs a list of relations:
      - {"subject": ..., "relation": ..., "object": ..., "pattern": ...}

    Pattern types included:
      A) active:    nsubj + VERB + dobj
      B) passive:   nsubjpass + VERB + prep->pobj   (e.g., born in)
      C) copular/adjectival predicate: nsubj + (ADJ/VERB) + prep->pobj  (e.g., headquartered in)
    """
    doc = nlp(text)
    relations: List[dict] = []

    for sent in doc.sents:
        ent_map = build_token_to_ent_text(sent)

        # ---------- Pattern A: active verb SVO ----------
        for tok in sent:
            if tok.pos_ != "VERB":
                continue

            subj = None
            dobj = None

            for child in tok.children:
                if child.dep_ == "nsubj":
                    subj = get_ent_or_token(ent_map, child)
                elif child.dep_ == "dobj":
                    dobj = get_ent_or_token(ent_map, child)

            if subj and dobj:
                relations.append({
                    "subject": subj,
                    "relation": tok.lemma_,
                    "object": dobj,
                    "pattern": "A_active_SVO",
                })

            # ---------- Pattern B: passive verb + prep pobj ----------
            # e.g. "X was born in Y"
            subjpass = None
            for child in tok.children:
                if child.dep_ == "nsubjpass":
                    subjpass = get_ent_or_token(ent_map, child)

            if subjpass:
                prep_pairs = find_prep_objects(tok, ent_map)
                for prep, obj_text in prep_pairs:
                    rel = REL_PREP_MAP.get(prep, prep)
                    relations.append({
                        "subject": subjpass,
                        "relation": f"{tok.lemma_}_{rel}",   # born_in, base_in, etc.
                        "object": obj_text,
                        "pattern": "B_passive_prep",
                    })

        # ---------- Pattern C: copular/adjectival predicate + prep pobj ----------
        # e.g. "Google is headquartered in Mountain View."
        # spaCy typically makes "headquartered" the ROOT (ADJ) and "Google" nsubj
        for tok in sent:
            if tok.pos_ not in ("ADJ", "VERB"):
                continue

            # Must have a nominal subject
            subj = None
            for child in tok.children:
                if child.dep_ == "nsubj":
                    subj = get_ent_or_token(ent_map, child)

            if not subj:
                continue

            # Prepositional objects under this predicate
            prep_pairs = find_prep_objects(tok, ent_map)
            for prep, obj_text in prep_pairs:
                rel = REL_PREP_MAP.get(prep, prep)
                relations.append({
                    "subject": subj,
                    "relation": f"{tok.lemma_}_{rel}",  # headquartered_in
                    "object": obj_text,
                    "pattern": "C_predicate_prep",
                })

    return relations

if __name__ == "__main__":
    for text in TEXTS:
        print("\n" + "=" * 80)
        print("TEXT:", text)
        rels = extract_relations(text)
        print("Relations:")
        for r in rels:
            print(r)