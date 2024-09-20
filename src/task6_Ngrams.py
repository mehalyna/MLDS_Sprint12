from nltk import ngrams
from collections import Counter
import nltk

nltk.download('punkt')


def generate_ngrams(text, n):
    """
    Generates N-grams from the input text.

    Args:
    text (str): The input text string.
    n (int): The size of the N-grams (e.g., n=2 for bigrams, n=3 for trigrams).

    Returns:
    list: A list of N-grams.
    """
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Generate N-grams
    n_grams = list(ngrams(tokens, n))
    return n_grams


def count_ngrams(ngrams):
    """
    Counts the frequency of each N-gram in the list.

    Args:
    ngrams (list): A list of N-grams.

    Returns:
    dict: A dictionary where keys are N-grams and values are their frequencies.
    """
    # Count the frequency of N-grams
    ngram_counts = Counter(ngrams)
    return dict(ngram_counts)


text = "I love NLP and learning NLP techniques"

# Generate bigrams
bigrams = generate_ngrams(text, 2)
print("Bigrams:", bigrams)

# Count bigrams
bigram_counts = count_ngrams(bigrams)
print("Bigram counts:", bigram_counts)
