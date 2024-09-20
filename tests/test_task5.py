from src.task5_word2vec import find_similar_words, train_word2vec


def test_train_word2vec_valid_model():
    sentences = [
        ['i', 'love', 'nlp'],
        ['nlp', 'is', 'fun'],
        ['deep', 'learning', 'is', 'a', 'subset', 'of', 'machine', 'learning'],
        ['machine', 'learning', 'is', 'fun']
    ]
    model = train_word2vec(sentences)

    assert model is not None, "Word2Vec model training failed"


def test_train_word2vec_valid_word_found():
    sentences = [
        ['i', 'love', 'nlp'],
        ['nlp', 'is', 'fun'],
        ['deep', 'learning', 'is', 'a', 'subset', 'of', 'machine', 'learning'],
        ['machine', 'learning', 'is', 'fun']
    ]
    model = train_word2vec(sentences)

    assert 'nlp' in model.wv, "Word 'nlp' not found in vocabulary"


def test_find_similar_words_similar_word_found():
    sentences = [
        ['i', 'love', 'nlp'],
        ['nlp', 'is', 'fun'],
        ['deep', 'learning', 'is', 'a', 'subset', 'of', 'machine', 'learning'],
        ['machine', 'learning', 'is', 'fun']
    ]
    model = train_word2vec(sentences)

    similar_words = find_similar_words(model, 'nlp')

    assert len(similar_words) > 0, "No similar words found"


def test_find_similar_words_list():
    sentences = [
        ['i', 'love', 'nlp'],
        ['nlp', 'is', 'fun'],
        ['deep', 'learning', 'is', 'a', 'subset', 'of', 'machine', 'learning'],
        ['machine', 'learning', 'is', 'fun']
    ]
    model = train_word2vec(sentences)

    similar_words = find_similar_words(model, 'nlp')

    assert isinstance(similar_words, list), "Expected a list of similar words"

