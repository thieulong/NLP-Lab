from nltk.tokenize import word_tokenize

SENTENCE = "Mary saw the cat sit on the mat."

def tokenize(sentence: str) -> list[str]:
    """
    Tokenize a sentence into words and punctuation.
    """
    return word_tokenize(sentence)

if __name__ == "__main__":
    tokens = tokenize(SENTENCE)
    print("Tokens:")
    print(tokens)