from src.task3_BoW import bag_of_words


def test_bag_of_words_column_match():
    texts = ["NLP is fun", "NLP is challenging"]
    result = bag_of_words(texts)
    expected_columns = {'nlp', 'is', 'fun', 'challenging'}
    assert set(result.columns) == expected_columns, "BoW column mismatch"


def test_bag_of_words_shape_match():
    texts = ["NLP is fun", "NLP is challenging"]
    result = bag_of_words(texts)
    assert result.shape == (2, 4), "BoW shape mismatch"

