import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def calculate_tfidf(texts):
    """
    Converts a list of text strings into a TF-IDF representation.

    Args:
    texts (list): A list of text strings.

    Returns:
    DataFrame: A DataFrame representing the TF-IDF model.
    """
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    return df


if __name__ == "__main__":
    texts = ["Converts a list of text strings into a Bag of Words representation", \
             "The words tha changed the strings",\
             "Your duty words of the heart",
             "The Bag full of strings"]
    print(calculate_tfidf(texts))
    