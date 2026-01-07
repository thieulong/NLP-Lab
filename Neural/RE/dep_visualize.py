import spacy

nlp = spacy.load("en_core_web_trf")

TEXTS = [
    "Apple acquired Beats for $3 billion in 2014.",
    "Barack Obama was born in Hawaii.",
    "Google is headquartered in Mountain View.",
]

def show_sentence(sent):
    print("\nSENTENCE:", sent.text)

    # Entities
    if sent.ents:
        print("Entities:")
        for ent in sent.ents:
            print(f"  - {ent.text!r} [{ent.label_}]")
    else:
        print("Entities: (none)")

    # Token table
    print("\nTokens (index, text, lemma, pos, dep, head):")
    for tok in sent:
        print(
            f"{tok.i:>3}  {tok.text:<15}  lemma={tok.lemma_:<12}  "
            f"pos={tok.pos_:<6}  dep={tok.dep_:<10}  head={tok.head.text}"
        )

    # Dependency arcs (head -> child)
    print("\nDependency arcs (head --dep--> child):")
    for tok in sent:
        for child in tok.children:
            print(f"  {tok.text:<15} --{child.dep_:<10}--> {child.text}")

def main():
    for text in TEXTS:
        print("\n" + "=" * 80)
        print("TEXT:", text)
        doc = nlp(text)
        for sent in doc.sents:
            show_sentence(sent)

if __name__ == "__main__":
    main()