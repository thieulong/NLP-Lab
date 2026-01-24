from __future__ import annotations

from typing import Tuple

def coarse_ent_type(spacy_label: str) -> str:
    # Keep a stable coarse mapping; you can refine later if needed.
    if spacy_label in {"PERSON", "ORG", "GPE", "LOC", "NORP", "FAC"}:
        return spacy_label
    return spacy_label

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
