from __future__ import annotations

from typing import Dict, List, Tuple

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
