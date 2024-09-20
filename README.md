# Sprint12 Free Text and NLP in Data Science

---

## **Task 1: Text Cleaning and Tokenization**

Write a function `clean_and_tokenize(text)` that takes a string of text as input, performs text cleaning (removing punctuation and converting to lowercase), and tokenizes the text into words. Return the list of tokens.

---

## **Task 2: Stemming and Lemmatization**

Create a function `stem_and_lemmatize(tokens)` that takes a list of tokens and applies stemming and lemmatization. Return a dictionary with the original tokens, stemmed tokens, and lemmatized tokens.

---

## **Task 3: Implementing Bag of Words (BoW)**

Implement a function `bag_of_words(texts)` that takes a list of text strings and returns a BoW representation as a DataFrame with words as columns and their frequencies in each text as rows.

---

## **Task 4: Calculating TF-IDF**

Create a function `calculate_tfidf(texts)` that takes a list of text strings and returns their TF-IDF representation as a DataFrame.

---

## **Task 5: Training a Word2Vec Model for Text Representation**

Create functions`train_word2vec(sentences)` that trains a Word2Vec model on a list of tokenized sentences. The function should return the trained Word2Vec model and demonstrate how to find the most similar words to a given word using the model.
Define the function `find_similar_words(model, word)` which finds words similar to the given word using the trained Word2Vec model.

---

### **Task: N-gram Generation and Frequency Counting**

**Problem Definition:**
Write a function `generate_ngrams(text, n)` that takes a string of text and an integer `n` as inputs and returns a list of N-grams. Additionally, write a function `count_ngrams(ngrams)` that counts the frequency of each N-gram and returns the result as a dictionary.

---
