from __future__ import annotations

from typing import Dict, List

from records import DecisionRecord

def print_grouped(rejected: List[DecisionRecord], limit: int = 10) -> None:
    if not rejected:
        print("\nNo rejected candidates logged.")
        return

    print("\n" + "=" * 100)
    print("REJECTED CANDIDATES BY REASON")
    print("=" * 100)

    groups: Dict[str, List[DecisionRecord]] = {}
    for r in rejected:
        groups.setdefault(r.reason, []).append(r)

    for reason, items in sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True):
        print(f"{reason:32s} count={len(items)}")

    for reason, items in sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True):
        print("\n" + "-" * 100)
        print(f"REASON: {reason} (showing up to {limit})")
        for r in items[:limit]:
            print(f"[SENT {r.sent_id}] {r.sentence}")
            print(f"PAIR: {r.head} ({r.head_type}) -> {r.tail} ({r.tail_type})  via={r.pattern}")
            print(f"PRED: {r.pred_rel}  conf={r.conf:.6f}  margin={None if r.margin is None else round(r.margin,6)}")
            if getattr(r, "verifier_entailment", None) is not None:
                print(f"VERIFIER: entail={r.verifier_entailment:.4f}")
            if r.topk:
                print("TOPK:", [(lab, round(p, 6)) for lab, p in r.topk])
            print()
