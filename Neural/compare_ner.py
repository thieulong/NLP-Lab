import nltk
import spacy
from nltk.tokenize import word_tokenize

TEXTS = [
    "Mary saw the cat sit on the mat in New York.",
    "Apple acquired Beats for $3 billion in 2014.",
    "Barack Obama was born in Hawaii.",
]

def nltk_ner(text: str):
    tokens = word_tokenize(text)
    pos = nltk.pos_tag(tokens)
    tree = nltk.ne_chunk(pos)

    entities = []
    for subtree in tree:
        if hasattr(subtree, "label"):
            entities.append({
                "text": " ".join(t[0] for t in subtree),
                "label": subtree.label()
            })
    return entities

def spacy_ner(text: str, nlp):
    doc = nlp(text)
    return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_trf")

    for text in TEXTS:
        print("\nTEXT:", text)
        print("NLTK NER:")
        print(nltk_ner(text))
        print("spaCy NER:")
        print(spacy_ner(text, nlp))