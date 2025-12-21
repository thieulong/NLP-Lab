import re
import nltk
from nltk.sem import extract_rels, rtuple
from nltk.corpus import ieer

RELATION_PATTERN = re.compile(r".*\bin\b.*")

def extract_location_relations():
    """
    Extract ORG-in-LOC relations using NLTK's relation extractor.
    """
    results = []

    for doc in ieer.parsed_docs("NYT_19980315"):
        for rel in extract_rels(
            "ORG",
            "LOC",
            doc,
            corpus="ieer",
            pattern=RELATION_PATTERN,
        ):
            results.append(rtuple(rel))

    return results

if __name__ == "__main__":
    relations = extract_location_relations()
    print("Extracted relations:")
    for r in relations[:10]:
        print(r)