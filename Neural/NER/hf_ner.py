# Neural/hf_ner.py

from transformers import pipeline

TEXT = "Apple acquired Beats for $3 billion in 2014."

def hf_ner(text: str):
    """
    Named Entity Recognition using a Hugging Face pretrained transformer.
    """
    ner = pipeline(
        "ner",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple",  # important for clean spans
    )

    entities = ner(text)
    results = []

    for ent in entities:
        results.append({
            "text": ent["word"],
            "label": ent["entity_group"],
            "score": round(ent["score"], 4),
            "start": ent["start"],
            "end": ent["end"],
        })

    return results

if __name__ == "__main__":
    ents = hf_ner(TEXT)
    print("Detected entities:")
    for e in ents:
        print(e)