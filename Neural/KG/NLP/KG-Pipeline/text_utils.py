from __future__ import annotations

def normalize_quotes(s: str) -> str:
    return s.replace("’", "'").replace("“", '"').replace("”", '"')
