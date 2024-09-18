from src.task1_text_cleaning import clean_and_tokenize


def test_clean_and_tokenize():
    text = "Hello, World! NLP is amazing."
    expected = ['hello', 'world', 'nlp', 'is', 'amazing']
    result = clean_and_tokenize(text)
    assert result == expected, f"Expected {expected} but got {result}"


