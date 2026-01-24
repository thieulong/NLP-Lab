from __future__ import annotations

import os
import re
from typing import List, Tuple

try:
    from neo4j import GraphDatabase  # type: ignore
except Exception:
    GraphDatabase = None  # type: ignore

def rel_to_neo4j_type(rel: str) -> str:
    """
    Neo4j relationship types must match: [A-Z_][A-Z0-9_]*
    We'll derive a stable type and also store the original 'rel' as a property.
    """
    r = rel.strip()
    if not r:
        return "RELATED_TO"
    # keep only alnum and underscores
    r = r.strip("/")
    r = re.sub(r"[^0-9A-Za-z_]+", "_", r)
    r = re.sub(r"_+", "_", r).strip("_")
    if not r:
        return "RELATED_TO"
    r = r.upper()
    if not re.match(r"^[A-Z_]", r):
        r = "_" + r
    return r

def read_neo4j_creds(args) -> Tuple[str, str, str, str]:
    uri = args.neo4j_uri or os.getenv("NEO4J_URI", "")
    user = (
        args.neo4j_user
        or os.getenv("NEO4J_USER", "")
        or os.getenv("NEO4J_USERNAME", "")
    )
    pwd = args.neo4j_password or os.getenv("NEO4J_PASSWORD", "")
    db = args.neo4j_database or os.getenv("NEO4J_DATABASE", "")
    return uri, user, pwd, db


def wipe_neo4j(driver, database: str = "") -> None:
    query = "MATCH (n) DETACH DELETE n"
    with driver.session(database=database or None) as sess:
        sess.run(query)


def write_triples_neo4j(
    driver,
    *,
    triples: List[Tuple[str, str, str, float, str, str, str, str]],
    database: str = "",
) -> None:
    """
    triples: (head, rel, tail, conf, sentence, pattern, head_type, tail_type)
    """
    cypher = """
    MERGE (h:Entity {name: $h})
      ON CREATE SET h.created_at = timestamp()
    SET h.type = $h_type

    MERGE (t:Entity {name: $t})
      ON CREATE SET t.created_at = timestamp()
    SET t.type = $t_type

    WITH h, t
    CALL apoc.merge.relationship(
      h,
      $rel_type,
      {rel: $rel},        // identity properties
      {conf: $conf, evidence: $evidence, pattern: $pattern, updated_at: timestamp()},
      t
    ) YIELD rel
    RETURN rel
    """
    # Note: this uses APOC. Neo4j Aura typically has APOC core available.
    # If APOC is not available, replace with dynamic string relationship creation (more awkward).
    with driver.session(database=database or None) as sess:
        for h, rel, t, conf, evidence, pattern, h_type, t_type in triples:
            sess.run(
                cypher,
                {
                    "h": h,
                    "t": t,
                    "h_type": h_type,
                    "t_type": t_type,
                    "rel": rel,
                    "rel_type": rel_to_neo4j_type(rel),
                    "conf": float(conf),
                    "evidence": evidence,
                    "pattern": pattern,
                },
            )

