# Neural/RE/analyze_export.py
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


def read_jsonl(p: Path) -> List[Dict[str, Any]]:
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges_jsonl", type=str, default="Neural/RE/exports/edges.jsonl")
    ap.add_argument("--topn", type=int, default=15)
    ap.add_argument("--low_conf_n", type=int, default=15)
    ap.add_argument("--high_conf_n", type=int, default=10)
    args = ap.parse_args()

    p = Path(args.edges_jsonl)
    rows = read_jsonl(p)

    print(f"Loaded edges: {len(rows)} from {p}")

    rel_counts = Counter(r["relation"] for r in rows)
    print("\nTop relations:")
    for rel, c in rel_counts.most_common(args.topn):
        print(f"  {rel:45s}  {c:6d}  ({c/len(rows):.3f})")

    confs = [float(r["confidence"]) for r in rows]
    print(f"\nConfidence stats:")
    print(f"  min={min(confs):.4f}  mean={sum(confs)/len(confs):.4f}  max={max(confs):.4f}")

    # Show lowest confidence examples (most likely borderline / noisy)
    rows_sorted = sorted(rows, key=lambda r: float(r["confidence"]))
    print(f"\nLowest-confidence {args.low_conf_n} edges:")
    for r in rows_sorted[: args.low_conf_n]:
        print("-" * 100)
        print(f"conf={r['confidence']}  rel={r['relation']}")
        print(f"head={r['head']} | tail={r['tail']}")
        print(f"text={r['text']}")
        print(f"topk={r['topk']}")

    # Show some highest confidence examples (sanity check)
    print(f"\nHighest-confidence {args.high_conf_n} edges:")
    for r in rows_sorted[-args.high_conf_n :]:
        print("-" * 100)
        print(f"conf={r['confidence']}  rel={r['relation']}")
        print(f"head={r['head']} | tail={r['tail']}")
        print(f"text={r['text']}")

    # Optional: ambiguity bucket based on top1-top2
    gaps = []
    for r in rows:
        topk = r.get("topk", [])
        if isinstance(topk, list) and len(topk) >= 2:
            gaps.append(float(topk[0][1]) - float(topk[1][1]))
    if gaps:
        gaps_sorted = sorted(gaps)
        print("\nTop1-Top2 gap stats:")
        print(f"  min={gaps_sorted[0]:.4f}  p50={gaps_sorted[len(gaps_sorted)//2]:.4f}  max={gaps_sorted[-1]:.4f}")

if __name__ == "__main__":
    main()
