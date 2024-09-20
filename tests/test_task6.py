from src.task6_Ngrams import count_ngrams, generate_ngrams


def test_generate_ngrams():
    text = "I love NLP and learning NLP techniques"
    n = 2
    expected_bigrams = [
        ('I', 'love'),
        ('love', 'NLP'),
        ('NLP', 'and'),
        ('and', 'learning'),
        ('learning', 'NLP'),
        ('NLP', 'techniques')
    ]
    result = generate_ngrams(text, n)
    assert result == expected_bigrams, f"Expected {expected_bigrams} but got {result}"


def test_count_ngrams():
    ngrams = [
        ('I', 'love'),
        ('love', 'NLP'),
        ('NLP', 'and'),
        ('and', 'learning'),
        ('learning', 'NLP'),
        ('NLP', 'techniques')
    ]
    expected_counts = {
        ('I', 'love'): 1,
        ('love', 'NLP'): 1,
        ('NLP', 'and'): 1,
        ('and', 'learning'): 1,
        ('learning', 'NLP'): 1,
        ('NLP', 'techniques'): 1
    }
    result = count_ngrams(ngrams)
    assert result == expected_counts, f"Expected {expected_counts} but got {result}"

