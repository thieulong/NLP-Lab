import nltk
from nltk.tokenize import word_tokenize

SENTENCE = "Apple acquired Beats for $3 billion in 2014."

def named_entity_detection(sentence: str):
    """
    Detect named entities using NLTK's NE chunker.
    Returns a chunk tree.
    """
    tokens = word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)
    tree = nltk.ne_chunk(pos_tags)
    return tree

if __name__ == "__main__":
    ne_tree = named_entity_detection(SENTENCE)
    print("Named Entity Tree:")
    print(ne_tree)