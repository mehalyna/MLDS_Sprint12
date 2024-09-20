from src.task4_TFIDF import calculate_tfidf


def test_calculate_tfidf_correct_terms():
    texts = ["NLP is amazing", "Learning NLP"]
    result = calculate_tfidf(texts)
    assert 'nlp' in result.columns, "TF-IDF did not include expected term 'nlp'"


def test_calculate_tfidf_correct_number_rows():
    texts = ["NLP is amazing", "Learning NLP"]
    result = calculate_tfidf(texts)
    assert result.shape[0] == 2, "TF-IDF incorrect number of rows"
