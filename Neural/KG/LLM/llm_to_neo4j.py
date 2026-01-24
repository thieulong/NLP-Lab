#!/usr/bin/env python3
"""
LLM-only KG builder (Ollama -> triples -> Neo4j)

Fixes vs previous version:
- Robust JSON extraction:
  - retries on parse failure (ask LLM to re-emit valid JSON)
  - salvages truncated JSON arrays by keeping only complete objects
- Stronger prompt to prevent schema violations (entity type vs relation)
- Keeps benchmark fair: ONLY NYT relations, no no_relation.

Env vars (preferred):
  export NEO4J_URI="neo4j+s://23e03260.databases.neo4j.io"
  export NEO4J_USERNAME="neo4j"
  export NEO4J_PASSWORD="..."

Run:
  python Neural/KG/llm_to_neo4j.py
  python Neural/KG/llm_to_neo4j.py --interactive
  python Neural/KG/llm_to_neo4j.py --no_neo4j
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Tuple, Optional

import requests
from neo4j import GraphDatabase


# -----------------------------
# Ollama config
# -----------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b-instruct"


# -----------------------------
# NYT relation label set
# -----------------------------
ALLOWED_RELATIONS: List[str] = [
    "/location/location/contains",
    "/location/administrative_division/country",
    "/people/person/nationality",
    "/people/person/place_lived",
    "/business/person/company",
    "/location/neighborhood/neighborhood_of",
    "/people/person/place_of_birth",
    "/people/deceased_person/place_of_death",
    "/business/company/founders",
    "/location/country/administrative_divisions",
    "/location/country/capital",
    "/people/person/children",
    "/business/company/place_founded",
    "/business/company/major_shareholders",
    "/business/company_shareholder/major_shareholder_of",
    "/business/company/advisors",
    "/people/ethnicity/geographic_distribution",
    "/sports/sports_team_location/teams",
    "/sports/sports_team/location",
]

ALLOWED_ENTITY_TYPES = ["PERSON", "ORG", "GPE", "LOC", "NORP", "FAC", "OTHER"]


# -----------------------------
# Helpers
# -----------------------------
def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_json_array_strict(text: str) -> str:
    """
    Extract the first top-level JSON array from a string.
    Raises if cannot find matching ']'.
    """
    text = _strip_fences(text)
    start = text.find("[")
    if start == -1:
        raise ValueError("No JSON array '[' found in LLM output.")

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    raise ValueError("Could not find matching closing ']' for JSON array.")


def salvage_truncated_json_array(text: str) -> Optional[str]:
    """
    Best-effort salvage when the model outputs a truncated JSON array.

    Strategy:
    - Find the first '['
    - Scan characters tracking:
        array_depth ([]) and object_depth ({})
        string state to ignore brackets inside strings
    - Record the last position where we had:
        array_depth == 1 and object_depth == 0
      i.e., just finished a complete object inside the array.
    - Cut there and append ']'.

    Returns:
      - salvaged JSON array string OR None if cannot salvage.
    """
    text = _strip_fences(text)
    start = text.find("[")
    if start == -1:
        return None

    arr_depth = 0
    obj_depth = 0
    in_str = False
    esc = False

    last_good_end = None

    for i in range(start, len(text)):
        ch = text[i]

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue

        if ch == "[":
            arr_depth += 1
        elif ch == "]":
            arr_depth -= 1
            if arr_depth == 0:
                # We found a complete array anyway
                return text[start : i + 1]
        elif ch == "{":
            obj_depth += 1
        elif ch == "}":
            obj_depth -= 1
            # after closing an object, we might be at a safe boundary
            if arr_depth == 1 and obj_depth == 0:
                last_good_end = i

    if last_good_end is None:
        return None

    # Include everything up to that object end, then close the array.
    salvaged = text[start : last_good_end + 1].rstrip()

    # Remove trailing commas if any (common in truncated outputs)
    salvaged = re.sub(r",\s*$", "", salvaged)

    return salvaged + "]"


def rel_to_neo4j_type(rel: str) -> str:
    s = rel.strip().strip("/")
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "RELATED_TO"
    return s.upper()


def get_neo4j_creds(args) -> Tuple[str, str, str]:
    uri = args.neo4j_uri or os.getenv("NEO4J_URI", "")
    user = args.neo4j_user or os.getenv("NEO4J_USERNAME", "")
    pwd = args.neo4j_password or os.getenv("NEO4J_PASSWORD", "")
    if not uri or not user or not pwd:
        raise RuntimeError(
            "Missing Neo4j credentials. Set env vars NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD "
            "or pass --neo4j_uri/--neo4j_user/--neo4j_password."
        )
    return uri, user, pwd


def call_ollama(prompt: str, temperature: float = 0.0, num_predict: int = 1600) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": float(temperature),
            "num_predict": int(num_predict),
        },
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=240)
    resp.raise_for_status()
    data = resp.json()
    return (data.get("response", "") or "").strip()


def build_prompt(paragraph: str) -> str:
    allowed_rel_str = "\n".join(f"- {r}" for r in ALLOWED_RELATIONS)
    allowed_type_str = ", ".join(ALLOWED_ENTITY_TYPES)

    # Very explicit guardrails: prevent mixing relation labels into entity type fields.
    return f"""You are a strict information extraction system.

You MUST return a SINGLE valid JSON array and nothing else.

TASK:
- Read the paragraph.
- Extract ONLY relation triples explicitly supported by the text.
- Use ONLY the allowed relation labels listed below.
- There is NO "no_relation". If not supported, do not output a triple.

CRITICAL FIELD RULES:
- head.type and tail.type MUST be one of: {allowed_type_str}
- head.type and tail.type MUST NEVER be a relation label.
- relation MUST be one of the allowed relation labels.

OUTPUT JSON schema:
[
  {{
    "head": {{"text": "...", "type": "PERSON|ORG|GPE|LOC|NORP|FAC|OTHER"}},
    "relation": "<allowed relation label>",
    "tail": {{"text": "...", "type": "PERSON|ORG|GPE|LOC|NORP|FAC|OTHER"}},
    "evidence": "short quote from the paragraph (<= 18 words)",
    "confidence": 0.0 to 1.0
  }},
  ...
]

DE-DUPLICATION:
- Avoid duplicates (same head.text + relation + tail.text). Keep the highest confidence.

Allowed relation labels:
{allowed_rel_str}

Paragraph:
\"\"\"{paragraph}\"\"\"
"""


def build_repair_prompt(bad_output: str) -> str:
    """
    If the model output is truncated or invalid, ask it to re-emit clean JSON.
    """
    allowed_rel_str = "\n".join(f"- {r}" for r in ALLOWED_RELATIONS)
    allowed_type_str = ", ".join(ALLOWED_ENTITY_TYPES)

    bad_output = _strip_fences(bad_output)
    bad_output = bad_output[:2500]  # keep it short; just show the start

    return f"""Your previous output was invalid or truncated JSON.

You MUST rewrite the entire answer as a SINGLE valid JSON array that conforms to the schema.

CRITICAL:
- No Markdown fences.
- No commentary text.
- Return ONLY the JSON array.
- head.type and tail.type MUST be one of: {allowed_type_str}
- relation MUST be one of the allowed relation labels below.
- Do not invent relation labels.
- No "no_relation"; omit unsupported triples.

Allowed relation labels:
{allowed_rel_str}

Invalid/truncated output (for reference):
{bad_output}
"""


def dedup_triples(triples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for t in triples:
        h = (t.get("head") or {}).get("text", "").strip()
        r = (t.get("relation") or "").strip()
        ta = (t.get("tail") or {}).get("text", "").strip()
        key = (h, r, ta)
        if not h or not r or not ta:
            continue
        if r not in ALLOWED_RELATIONS:
            continue
        conf = float(t.get("confidence", 0.0) or 0.0)
        if key not in best or conf > float(best[key].get("confidence", 0.0) or 0.0):
            best[key] = t
    return list(best.values())


def validate_triples(triples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for t in triples:
        if not isinstance(t, dict):
            continue
        head = t.get("head")
        tail = t.get("tail")
        rel = t.get("relation")
        if not isinstance(head, dict) or not isinstance(tail, dict) or not isinstance(rel, str):
            continue

        ht = str(head.get("text", "")).strip()
        tt = str(tail.get("text", "")).strip()
        rt = rel.strip()

        if not ht or not tt or not rt:
            continue
        if rt not in ALLOWED_RELATIONS:
            continue

        htype = str(head.get("type", "OTHER")).strip().upper()
        ttype = str(tail.get("type", "OTHER")).strip().upper()

        # Prevent the exact bug you saw: type accidentally becomes a relation label.
        if htype not in ALLOWED_ENTITY_TYPES:
            htype = "OTHER"
        if ttype not in ALLOWED_ENTITY_TYPES:
            ttype = "OTHER"

        evidence = str(t.get("evidence", "")).strip()
        try:
            conf = float(t.get("confidence", 0.0))
        except Exception:
            conf = 0.0
        conf = max(0.0, min(1.0, conf))

        cleaned.append({
            "head": {"text": ht, "type": htype},
            "relation": rt,
            "tail": {"text": tt, "type": ttype},
            "evidence": evidence,
            "confidence": conf,
        })

    return dedup_triples(cleaned)


# -----------------------------
# Neo4j
# -----------------------------
def neo4j_wipe(driver) -> None:
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")


def neo4j_constraints(driver) -> None:
    with driver.session() as session:
        session.run("CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")


def neo4j_insert(driver, triples: List[Dict[str, Any]]) -> None:
    with driver.session() as session:
        for t in triples:
            h = t["head"]["text"]
            ht = t["head"]["type"]
            r_raw = t["relation"]
            r_type = rel_to_neo4j_type(r_raw)
            ta = t["tail"]["text"]
            tt = t["tail"]["type"]
            conf = float(t.get("confidence", 0.0))
            ev = t.get("evidence", "")

            cypher = f"""
            MERGE (h:Entity {{name: $h}})
            ON CREATE SET h.type = $ht
            ON MATCH  SET h.type = coalesce(h.type, $ht)
            MERGE (t:Entity {{name: $t}})
            ON CREATE SET t.type = $tt
            ON MATCH  SET t.type = coalesce(t.type, $tt)
            MERGE (h)-[r:{r_type}]->(t)
            SET r.rel_raw = $rel_raw,
                r.confidence = $conf,
                r.evidence = $evidence
            """
            session.run(
                cypher,
                h=h, ht=ht,
                t=ta, tt=tt,
                rel_raw=r_raw,
                conf=conf,
                evidence=ev,
            )


# -----------------------------
# Default paragraph
# -----------------------------
DEFAULT_TEXT = (
    "Apple was founded in Cupertino in 1976 by Steve Jobs and Steve Wozniak, a moment that helped establish Silicon Valley as a global technology hub. "
    "After joining Apple, Tim Cook later became the company’s chief executive and moved to California, where he has lived for many years. "
    "Tesla was founded in Silicon Valley, and Elon Musk eventually became a major shareholder and public face of the company. "
    "New York contains Manhattan, and the Upper West Side is a well-known neighborhood within Manhattan that has long attracted writers and artists. "
    "Russia’s capital is Moscow, while Kazakhstan’s capital is Astana, reflecting the political centers of the two former Soviet republics. "
    "During a recent interview, Elon Musk joked that artificial intelligence might someday run entire governments, a claim that sparked debate online. "
    "Steve Jobs admired Pablo Picasso and once said that creativity comes from connecting experiences rather than following rules. "
    "Although Apple and Tesla are often compared by investors, the two companies operate in very different industries and markets."
)


def uri_to_browser_url(uri: str) -> str:
    if uri.startswith("neo4j+s://"):
        return "https://" + uri[len("neo4j+s://"):]
    if uri.startswith("neo4j://"):
        return "http://" + uri[len("neo4j://"):]
    return uri


def parse_llm_json_with_retries(
    paragraph: str,
    temperature: float,
    num_predict: int,
    max_retries: int = 2,
) -> List[Dict[str, Any]]:
    """
    1) Call LLM with strict prompt
    2) Try strict JSON extraction
    3) If fails, salvage truncated output
    4) If still fails, retry with repair prompt (max_retries)
    """
    prompt = build_prompt(paragraph)
    out = call_ollama(prompt, temperature=temperature, num_predict=num_predict)

    for attempt in range(max_retries + 1):
        # First try strict extraction
        try:
            json_str = extract_json_array_strict(out)
            raw = json.loads(json_str)
            if not isinstance(raw, list):
                raise ValueError("LLM did not return a JSON array.")
            return raw
        except Exception:
            # Try salvage
            salvaged = salvage_truncated_json_array(out)
            if salvaged is not None:
                try:
                    raw = json.loads(salvaged)
                    if isinstance(raw, list):
                        return raw
                except Exception:
                    pass

        if attempt == max_retries:
            # Give up after retries
            raise ValueError("LLM output could not be parsed even after retries/salvage.")

        # Retry with a repair prompt
        repair_prompt = build_repair_prompt(out)
        out = call_ollama(repair_prompt, temperature=0.0, num_predict=max(num_predict, 1800))

    raise ValueError("Unreachable.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=str, default="")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--num_predict", type=int, default=1600)
    ap.add_argument("--max_retries", type=int, default=2)

    # Neo4j options
    ap.add_argument("--no_neo4j", action="store_true", help="Do not write to Neo4j (LLM extraction only).")
    ap.add_argument("--neo4j_uri", type=str, default="")
    ap.add_argument("--neo4j_user", type=str, default="")       # maps to NEO4J_USERNAME
    ap.add_argument("--neo4j_password", type=str, default="")

    args = ap.parse_args()

    def run_once(paragraph: str) -> None:
        paragraph = paragraph.strip()
        if not paragraph:
            return

        print("\n[LLM] Extracting triples via Ollama...")

        try:
            raw = parse_llm_json_with_retries(
                paragraph,
                temperature=args.temperature,
                num_predict=args.num_predict,
                max_retries=args.max_retries,
            )
        except Exception as e:
            print("\n[ERROR] Could not parse LLM JSON after salvage/retries.")
            raise

        triples = validate_triples(raw)

        print(f"\n[LLM] Triples accepted after validation/dedup: {len(triples)}")
        for t in sorted(triples, key=lambda x: x.get("confidence", 0.0), reverse=True):
            h = t["head"]["text"]
            r = t["relation"]
            ta = t["tail"]["text"]
            conf = t.get("confidence", 0.0)
            print(f"({h}, {r}, {ta})  conf={conf:.3f}")

        if args.no_neo4j:
            print("\n[Neo4j] Skipped (--no_neo4j).")
            return

        uri, user, pwd = get_neo4j_creds(args)
        driver = GraphDatabase.driver(uri, auth=(user, pwd))

        try:
            neo4j_constraints(driver)

            # Always wipe at the beginning of each run
            print("\n[Neo4j] Wiping database (beginning of this run)...")
            neo4j_wipe(driver)

            print("[Neo4j] Inserting triples...")
            neo4j_insert(driver, triples)
        finally:
            driver.close()

        print("\n[Neo4j] Done.")
        print("Neo4j browser (best effort):", uri_to_browser_url(uri))

    if args.interactive:
        print("\nEnter a paragraph (blank line to exit). Neo4j will be wiped each run unless --no_neo4j.")
        while True:
            inp = input("\n> ").strip()
            if not inp:
                break
            run_once(inp)
    else:
        text = args.text.strip() if args.text else DEFAULT_TEXT
        run_once(text)


if __name__ == "__main__":
    main()