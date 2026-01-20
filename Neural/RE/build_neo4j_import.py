#!/usr/bin/env python3
"""
Build Neo4j-ready nodes.csv and edges.csv from exported edges.jsonl.

Input:
  - Neural/RE/exports_schema_on/edges.jsonl   (recommended)

Output folder (created if missing):
  - Neural/RE/neo4j-kg/nodes.csv
  - Neural/RE/neo4j-kg/edges.csv

Key features:
  - Canonical node IDs using option 1: "{TYPE}:{NAME}"
  - Relation normalization to Neo4j-safe rel types (UPPER_SNAKE)
  - Direction gating per relation:
      - enforce canonical direction
      - optionally flip direction when allowed and types match
  - Deduplicate/aggregate identical triples:
      groups by (src_id, rel_type, dst_id) and aggregates support_count + confidence stats

Run:
  python Neural/RE/build_neo4j_import.py \
    --input_jsonl Neural/RE/exports_schema_on/edges.jsonl \
    --out_dir Neural/RE/neo4j-kg
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any


# -----------------------
# Helpers
# -----------------------

def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Bad JSON on line {line_no}: {e}") from e


def neo4j_rel_type(raw_rel: str) -> str:
    """
    Convert raw relation like '/business/company/place_founded'
    into Neo4j relationship type like 'BUSINESS_COMPANY_PLACE_FOUNDED'.
    """
    rel = raw_rel.strip()
    if rel.startswith("/"):
        rel = rel[1:]
    rel = rel.replace("/", "_")
    rel = rel.replace("-", "_")
    rel = rel.upper()
    return rel


def norm_text(s: str) -> str:
    return (s or "").strip()


def norm_type(t: str) -> str:
    # keep it simple; you can map later (e.g., GPE/LOC -> LOCATION)
    return (t or "").strip().upper()


def make_node_id(name: str, ntype: str) -> str:
    """
    Node ID option 1: "{TYPE}:{NAME}"
    If type missing, use "UNK".
    """
    name = norm_text(name)
    ntype = norm_type(ntype) or "UNK"
    return f"{ntype}:{name}"


@dataclass(frozen=True)
class RelRule:
    """
    Canonical direction rule for a relation.
    head_types: allowed types for head node (canonical)
    tail_types: allowed types for tail node (canonical)
    allow_flip: if True, when we see the reversed direction and types match reversed,
                we flip to canonical.
    """
    head_types: Tuple[str, ...]
    tail_types: Tuple[str, ...]
    allow_flip: bool = True


# -----------------------
# Direction + Type Rules
# -----------------------
# IMPORTANT:
# - These are default rules for NYT-style relations.
# - You can expand/edit later.
REL_RULES: Dict[str, RelRule] = {
    "/location/location/contains": RelRule(
        head_types=("GPE", "LOC", "FAC"),
        tail_types=("GPE", "LOC", "FAC"),
        allow_flip=False,  # 'contains' is directional; don't guess flipping
    ),
    "/location/neighborhood/neighborhood_of": RelRule(
        head_types=("GPE", "LOC"),
        tail_types=("GPE", "LOC"),
        allow_flip=False,
    ),
    "/location/administrative_division/country": RelRule(
        head_types=("GPE", "LOC"),
        tail_types=("GPE", "LOC"),
        allow_flip=True,   # city/state <-> country errors are common; safe to flip with types
    ),
    "/location/country/capital": RelRule(
        head_types=("GPE", "LOC"),  # country
        tail_types=("GPE", "LOC"),  # capital city
        allow_flip=True,
    ),
    "/people/person/place_of_birth": RelRule(
        head_types=("PERSON",),
        tail_types=("GPE", "LOC"),
        allow_flip=True,
    ),
    "/people/person/place_lived": RelRule(
        head_types=("PERSON",),
        tail_types=("GPE", "LOC"),
        allow_flip=True,
    ),
    "/people/deceased_person/place_of_death": RelRule(
        head_types=("PERSON",),
        tail_types=("GPE", "LOC"),
        allow_flip=True,
    ),
    "/people/person/nationality": RelRule(
        head_types=("PERSON",),
        tail_types=("NORP", "GPE", "LOC"),
        allow_flip=True,
    ),
    "/business/person/company": RelRule(
        head_types=("PERSON",),
        tail_types=("ORG",),
        allow_flip=True,
    ),
    "/business/company/founders": RelRule(
        head_types=("ORG",),
        tail_types=("PERSON",),
        allow_flip=True,
    ),
    "/business/company/place_founded": RelRule(
        head_types=("ORG",),
        tail_types=("GPE", "LOC"),
        allow_flip=True,
    ),
    "/business/company/major_shareholders": RelRule(
        head_types=("ORG",),
        tail_types=("PERSON", "ORG"),
        allow_flip=True,
    ),
    "/business/company_shareholder/major_shareholder_of": RelRule(
        head_types=("PERSON", "ORG"),
        tail_types=("ORG",),
        allow_flip=True,
    ),
}


def type_ok(ntype: str, allowed: Tuple[str, ...]) -> bool:
    if not allowed:
        return True
    if not ntype:
        return False
    return ntype in allowed


def apply_direction_gating(
    raw_rel: str,
    head: str,
    tail: str,
    head_type: str,
    tail_type: str,
) -> Tuple[Optional[str], str, str, str, str]:
    """
    Enforce canonical direction for raw_rel.

    Returns:
      (reason_if_dropped, head, tail, head_type, tail_type)

    - If relation has rule:
        - If already canonical types: keep.
        - Else if reversed types AND allow_flip: flip.
        - Else: drop with reason.
    - If no rule: keep as-is.
    """
    rule = REL_RULES.get(raw_rel)
    if rule is None:
        return (None, head, tail, head_type, tail_type)

    ht = norm_type(head_type)
    tt = norm_type(tail_type)

    # If types missing, we can't reliably gate direction; keep as-is
    if ht == "UNK" or tt == "UNK" or ht == "" or tt == "":
        return (None, head, tail, head_type, tail_type)

    # Check canonical
    if type_ok(ht, rule.head_types) and type_ok(tt, rule.tail_types):
        return (None, head, tail, ht, tt)

    # Check reversed, optionally flip
    if rule.allow_flip and type_ok(tt, rule.head_types) and type_ok(ht, rule.tail_types):
        return (None, tail, head, tt, ht)

    return ("direction_or_type_mismatch", head, tail, ht, tt)


# -----------------------
# Aggregation
# -----------------------

@dataclass
class EdgeAgg:
    src_id: str
    dst_id: str
    rel_raw: str
    rel_type: str
    support_count: int = 0
    conf_sum: float = 0.0
    conf_max: float = 0.0

    # optionally keep a few evidence snippets (cap to avoid huge files)
    evidence: List[str] = None

    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []

    def add(self, conf: float, evidence_text: str, evidence_cap: int):
        self.support_count += 1
        self.conf_sum += conf
        self.conf_max = max(self.conf_max, conf)
        if evidence_text and len(self.evidence) < evidence_cap:
            self.evidence.append(evidence_text)

    @property
    def conf_mean(self) -> float:
        if self.support_count == 0:
            return 0.0
        return self.conf_sum / self.support_count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", type=str, required=True, help="edges.jsonl path")
    ap.add_argument("--out_dir", type=str, default="Neural/RE/neo4j-kg", help="output folder")
    ap.add_argument("--evidence_cap", type=int, default=0, help="keep up to N evidence sentences per edge (0 disables)")
    ap.add_argument("--min_support", type=int, default=1, help="drop aggregated edges with support_count < min_support")

    args = ap.parse_args()
    in_path = Path(args.input_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes_out = out_dir / "nodes.csv"
    edges_out = out_dir / "edges.csv"

    # Collect nodes and aggregate edges
    nodes: Dict[str, Dict[str, str]] = {}  # node_id -> {id,name,type}
    edges: Dict[Tuple[str, str, str], EdgeAgg] = {}  # (src_id, rel_type, dst_id) -> agg

    dropped_direction = 0
    total_seen = 0
    total_kept = 0

    for obj in read_jsonl(in_path):
        total_seen += 1

        head = norm_text(obj.get("source") or obj.get("head") or "")
        tail = norm_text(obj.get("target") or obj.get("tail") or "")
        raw_rel = norm_text(obj.get("relation") or "")

        if not head or not tail or not raw_rel:
            continue

        conf = float(obj.get("confidence", 0.0) or 0.0)
        # Types may be present (recommended). If missing, becomes UNK.
        head_type = norm_type(obj.get("head_type", "")) or "UNK"
        tail_type = norm_type(obj.get("tail_type", "")) or "UNK"

        # Direction gating (and possible flipping)
        reason, head2, tail2, ht2, tt2 = apply_direction_gating(
            raw_rel=raw_rel,
            head=head,
            tail=tail,
            head_type=head_type,
            tail_type=tail_type,
        )
        if reason is not None:
            dropped_direction += 1
            continue

        # Build node IDs (option 1)
        src_id = make_node_id(head2, ht2)
        dst_id = make_node_id(tail2, tt2)

        # Register nodes
        if src_id not in nodes:
            nodes[src_id] = {"id": src_id, "name": head2, "type": ht2}
        if dst_id not in nodes:
            nodes[dst_id] = {"id": dst_id, "name": tail2, "type": tt2}

        rel_type = neo4j_rel_type(raw_rel)

        key = (src_id, rel_type, dst_id)
        if key not in edges:
            edges[key] = EdgeAgg(
                src_id=src_id,
                dst_id=dst_id,
                rel_raw=raw_rel,
                rel_type=rel_type,
            )

        evidence_text = ""
        if args.evidence_cap and args.evidence_cap > 0:
            # sentence field varies across your scripts; try a few keys
            evidence_text = norm_text(obj.get("sentence") or obj.get("text") or "")

        edges[key].add(conf=conf, evidence_text=evidence_text, evidence_cap=args.evidence_cap)
        total_kept += 1

    # Write nodes.csv
    with nodes_out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "name", "type"])
        w.writeheader()
        for node in nodes.values():
            w.writerow(node)

    # Write edges.csv (aggregated)
    with edges_out.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["src_id", "dst_id", "rel_type", "rel_raw", "support_count", "conf_max", "conf_mean", "evidence"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for agg in edges.values():
            if agg.support_count < args.min_support:
                continue

            evidence_joined = ""
            if args.evidence_cap and args.evidence_cap > 0 and agg.evidence:
                evidence_joined = " || ".join(agg.evidence)

            w.writerow({
                "src_id": agg.src_id,
                "dst_id": agg.dst_id,
                "rel_type": agg.rel_type,
                "rel_raw": agg.rel_raw,
                "support_count": agg.support_count,
                "conf_max": f"{agg.conf_max:.6f}",
                "conf_mean": f"{agg.conf_mean:.6f}",
                "evidence": evidence_joined,
            })

    print("\nDone.")
    print(f"Input: {in_path}")
    print(f"Out:   {out_dir}")
    print(f"Seen:  {total_seen}")
    print(f"Kept:  {total_kept}")
    print(f"Dropped (direction/type gating): {dropped_direction}")
    print(f"Nodes: {len(nodes)}")
    print(f"Edges (aggregated): {len(edges)}")
    print(f"Wrote: {nodes_out}")
    print(f"Wrote: {edges_out}")


if __name__ == "__main__":
    main()