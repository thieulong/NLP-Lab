from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

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

