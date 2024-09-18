import re
from nltk.tokenize import word_tokenize


def clean_and_tokenize(text):
    """
    Cleans the input text by removing punctuation and converting it to lowercase,
    then tokenizes it into individual words.

    Args:
    text (str): The input text string.

    Returns:
    list: A list of tokens (words).
    """
    # Remove punctuation and convert to lowercase
    cleaned_text = re.sub(r'[^\w\s]', '', text.lower())
    # Tokenize the text
    tokens = word_tokenize(cleaned_text)
    return tokens


if __name__ == "__main__":
    print(clean_and_tokenize("Cleans the input text by removing punctuation and converting it to lowercase, \
    then tokenizes it into individual words."))

