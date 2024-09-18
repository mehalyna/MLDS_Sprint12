from src.task2_stemm_lemma import stem_and_lemmatize


def test_stem():
    tokens = ['running', 'jumps', 'easily', 'cats']
    result = stem_and_lemmatize(tokens)
    assert result['stemmed'] == ['run', 'jump', 'easili', 'cat'], "Stemming failed"


def test_lemmatize():
    tokens = ['running', 'jumps', 'easily', 'cats']
    result = stem_and_lemmatize(tokens)
    assert result['lemmatized'] == ['running', 'jump', 'easily', 'cat'], "Lemmatization failed"

