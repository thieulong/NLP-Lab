from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

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

