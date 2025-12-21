import nltk
from nltk.tokenize import word_tokenize

SENTENCE = "Mary saw the cat sit on the mat."

def pos_tag(sentence: str) -> list[tuple[str, str]]:
    """
    Assign part-of-speech tags to tokens.
    """
    tokens = word_tokenize(sentence)
    return nltk.pos_tag(tokens)

if __name__ == "__main__":
    tagged = pos_tag(SENTENCE)
    print("POS-tagged sentence:")
    print(tagged)