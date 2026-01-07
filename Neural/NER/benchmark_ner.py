# Neural/benchmark_ner.py

import nltk
import spacy
from nltk.tokenize import word_tokenize
from transformers import pipeline

from llm_ner import llm_ner

TEXTS = [
    "Mary saw the cat sit on the mat in New York.",
    "Apple acquired Beats for $3 billion in 2014.",
    "Barack Obama was born in Hawaii.",
]

LABEL_MAP_TO_SPACY = {
    # HF CoNLL style
    "PER": "PERSON",
    "ORG": "ORG",
    "LOC": "LOC",
    "MISC": "MISC",

    # Common variants
    "ORGANIZATION": "ORG",
    "LOCATION": "LOC",
}

def norm_label(label: str) -> str:
    lab = (label or "").strip().upper()
    return LABEL_MAP_TO_SPACY.get(lab, lab)

def nltk_ner(text: str):
    tokens = word_tokenize(text)
    pos = nltk.pos_tag(tokens)
    tree = nltk.ne_chunk(pos)

    entities = []
    for subtree in tree:
        if hasattr(subtree, "label"):
            entities.append({
                "text": " ".join(t[0] for t in subtree),
                "label": norm_label(subtree.label()),
            })
    return entities

def spacy_ner(text: str, nlp):
    doc = nlp(text)
    return [{"text": ent.text, "label": norm_label(ent.label_)} for ent in doc.ents]

def hf_ner(text: str, ner_pipe):
    ents = ner_pipe(text)
    return [
        {
            "text": e["word"],
            "label": norm_label(e["entity_group"]),
            "score": round(float(e["score"]), 4),
        }
        for e in ents
    ]

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_trf")
    hf_pipe = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

    for text in TEXTS:
        print("\n" + "=" * 80)
        print("TEXT:", text)

        print("\nNLTK NER:")
        print(nltk_ner(text))

        print("\nspaCy NER:")
        print(spacy_ner(text, nlp))

        print("\nHF (BERT) NER:")
        print(hf_ner(text, hf_pipe))

        print("\nLLM (Qwen2.5) NER:")
        try:
            llm_out = llm_ner(text)
            llm_out = [{"text": x["text"], "label": norm_label(x["label"])} for x in llm_out]
            print(llm_out)
        except Exception as e:
            print("LLM NER failed:", e)