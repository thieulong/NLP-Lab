import nltk
from nltk.tokenize import sent_tokenize

TEXT = "Mary saw the cat sit on the mat. The cat was happy. It purred loudly."

def sentence_segmentation(text: str) -> list[str]:
    """
    Split raw text into sentences.
    """
    return sent_tokenize(text)

if __name__ == "__main__":
    sentences = sentence_segmentation(TEXT)
    print("Sentences:")
    for s in sentences:
        print("-", s)