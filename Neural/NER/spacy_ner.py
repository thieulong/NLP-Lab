import spacy

TEXT = "Apple acquired Beats for $3 billion in 2014."

def spacy_ner(text: str):
    """
    Named Entity Recognition using spaCy pretrained transformer model.
    """
    nlp = spacy.load("en_core_web_trf")
    doc = nlp(text)

    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
        })

    return entities

if __name__ == "__main__":
    ents = spacy_ner(TEXT)
    print("Detected entities:")
    for e in ents:
        print(e)