# pylint: disable=W1401,C0103
"""This module transforms text to vectors"""
import joblib
import numpy as np
from scipy import sparse as sp_sparse
from sklearn.feature_extraction.text import TfidfVectorizer


# Bag of words
def bag_of_words(x_train, x_val, x_test, words_counts):
    """
    x_train: the training data
    x_val: the actual data
    x_test: the testing data
    words_counts: the number of words

    returns the bags of data
    """
    DICT_SIZE = 5000
    INDEX_TO_WORDS = sorted(words_counts, key=words_counts.get, reverse=True)[
        :DICT_SIZE
    ]
    WORDS_TO_INDEX = {word: i for i, word in enumerate(INDEX_TO_WORDS)}
    # ALL_WORDS = WORDS_TO_INDEX.keys()

    x_train_mybag = sp_sparse.vstack(
        [
            sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE))
            for text in x_train
        ]
    )
    x_val_mybag = sp_sparse.vstack(
        [
            sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE))
            for text in x_val
        ]
    )
    x_test_mybag = sp_sparse.vstack(
        [
            sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE))
            for text in x_test
        ]
    )
    print("X_train shape ", x_train_mybag.shape)
    print("X_val shape ", x_val_mybag.shape)
    print("X_test shape ", x_test_mybag.shape)

    # row = X_train_mybag[10].toarray()[0]
    # non_zero_elements_count = (row>0).sum()
    return x_train_mybag, x_val_mybag, x_test_mybag


def my_bag_of_words(text, words_to_index, dict_size):
    """
    text: a string
    dict_size: size of the dictionary

    return a vector which is a bag-of-words representation of 'text'
    """
    result_vector = np.zeros(dict_size)

    for word in text.split():
        if word in words_to_index:
            result_vector[words_to_index[word]] += 1
    return result_vector


def test_my_bag_of_words():
    """
    tests the bag of words
    """
    words_to_index = {"hi": 0, "you": 1, "me": 2, "are": 3}
    examples = ["hi how are you"]
    answers = [[1, 1, 0, 1]]
    for ex, ans in zip(examples, answers):
        if (my_bag_of_words(ex, words_to_index, 4) != ans).any():
            return f"Wrong answer for the case: {ex}"
    return "Basic tests are passed."


# TFIDF
def initialize_tfidf_vectorizer(x_train, x_val, x_test):
    """
    x_train: the training data
    x_val: the actual data
    x_test: the testing data

    returns the data for tfidf
    """
    x_train_tfidf, x_val_tfidf, x_test_tfidf, tfidf_vocab = tfidf_features(
        x_train, x_val, x_test
    )
    tfidf_reversed_vocab = {i: word for word, i in tfidf_vocab.items()}
    return x_train_tfidf, x_val_tfidf, x_test_tfidf, tfidf_vocab, tfidf_reversed_vocab


def tfidf_features(x_train, x_val, x_test):
    """
    x_train, x_val, x_test — samples
    return TF-IDF vectorized representation of each sample and vocabulary
    """
    # Create TF-IDF vectorizer with a proper parameters choice
    # Fit the vectorizer on the train set
    # Transform the train, test, and val sets and return the result

    tfidf_vectorizer = TfidfVectorizer(
        min_df=5, max_df=0.9, ngram_range=(1, 2), token_pattern="(\S+)"
    )

    x_train = tfidf_vectorizer.fit_transform(x_train)
    x_val = tfidf_vectorizer.transform(x_val)
    x_test = tfidf_vectorizer.transform(x_test)

    joblib.dump(tfidf_vectorizer, 'output/tfidf_vectorizer.joblib')

    return x_train, x_val, x_test, tfidf_vectorizer.vocabulary_
