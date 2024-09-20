from gensim.models import Word2Vec
import nltk

# Download required NLTK resources
nltk.download('punkt')


# Define example sentences
sentences = [
    ['i', 'love', 'nlp'],
    ['nlp', 'is', 'fun'],
    ['deep', 'learning', 'is', 'a', 'subset', 'of', 'machine', 'learning'],
    ['machine', 'learning', 'is', 'fun'],
    ['nlp', 'can', 'be', 'challenging'],
    ['artificial', 'intelligence', 'is', 'related', 'to', 'nlp']
]


# Function to train Word2Vec model
def train_word2vec(sentences):
    """
    Trains a Word2Vec model on a list of tokenized sentences.

    Args:
    sentences (list of list of str): A list of tokenized sentences.

    Returns:
    Word2Vec: A trained Word2Vec model.
    """
    # Initialize and train the Word2Vec model
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    return model


# Function to find similar words
def find_similar_words(model, word):
    """
    Finds the most similar words to a given word using a trained Word2Vec model.

    Args:
    model (Word2Vec): A trained Word2Vec model.
    word (str): The word to find similar words for.

    Returns:
    list: A list of tuples with similar words and their similarity scores.
    """
    return model.wv.most_similar(word)


# Train Word2Vec model on the sentences
model = train_word2vec(sentences)


similar_words = find_similar_words(model, 'nlp')
print("Words similar to 'nlp':")
for word, score in similar_words:
    print(f"{word}: {score:.4f}")

similar_words = find_similar_words(model, 'learning')
print("\nWords similar to 'learning':")
for word, score in similar_words:
    print(f"{word}: {score:.4f}")
