# Neural/llm_ner.py

import json
import re
import requests
from typing import List, Dict

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b-instruct"

ALLOWED_LABELS = [
    "PERSON",
    "NORP",
    "FAC",
    "ORG",
    "GPE",
    "LOC",
    "PRODUCT",
    "EVENT",
    "WORK_OF_ART",
    "LAW",
    "LANGUAGE",
    "DATE",
    "TIME",
    "MONEY",
    "QUANTITY",
    "ORDINAL",
    "CARDINAL",
]

def _extract_json_array(text: str) -> str:
    """
    Extract the first JSON array from the model output.
    """
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        return text

    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        raise ValueError("No JSON array found in LLM output")

    return match.group(0)


def llm_ner(text: str) -> List[Dict[str, str]]:
    """
    LLM-based NER using Ollama.
    Returns: [{ "text": ..., "label": ... }]
    """

    prompt = f"""
Extract named entities from the text below.

Return ONLY a JSON array.
Each item must be an object with:
- "text": the entity string exactly as it appears
- "label": one of {ALLOWED_LABELS}

Do not include explanations.
Do not include extra text.

Text:
{text}
""".strip()

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,   # critical for fair benchmarking
            "num_predict": 512
        }
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()

    result = response.json()
    output_text = result.get("response", "").strip()

    json_str = _extract_json_array(output_text)
    data = json.loads(json_str)

    cleaned = []
    for item in data:
        if not isinstance(item, dict):
            continue

        ent_text = str(item.get("text", "")).strip()
        ent_label = str(item.get("label", "")).strip()

        if ent_text and ent_label in ALLOWED_LABELS:
            cleaned.append({
                "text": ent_text,
                "label": ent_label
            })

    return cleaned


if __name__ == "__main__":
    test = "Apple acquired Beats for $3 billion in 2014."
    print(json.dumps(llm_ner(test), indent=2, ensure_ascii=False))