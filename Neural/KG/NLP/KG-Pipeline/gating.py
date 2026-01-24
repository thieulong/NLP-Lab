from __future__ import annotations

from config import REL_SCHEMA, REL_TRIGGERS, REL_THRESH


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
