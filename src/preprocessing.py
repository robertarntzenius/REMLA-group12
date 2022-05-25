# pylint: disable=E0012,W1401,R5503
"""This module preprocesses the data for the model"""
import re
from ast import literal_eval

import nltk
import pandas as pd
from nltk.corpus import stopwords

nltk.download("stopwords")


def init_preprocessing():
    """
    initialize the preprocessing
    """
    train = read_data("data/train.tsv")
    validation = read_data("data/validation.tsv")
    test = pd.read_csv("data/test.tsv", sep="\t")["title"]

    x_train, y_train = train["title"].values, train["tags"].values
    x_val, y_val = validation["title"].values, validation["tags"].values
    x_test = test["title"].values

    # Text prepare
    x_train = [text_prepare(x) for x in x_train]
    x_val = [text_prepare(x) for x in x_val]
    x_test = [text_prepare(x) for x in x_test]
    return x_train, y_train, x_val, y_val, x_test


def read_data(filename):
    """
    filename: the name of the file

    return the data from the file
    """
    data = pd.read_csv(filename, sep="\t")["tags"]
    data["tags"] = data["tags"].apply(literal_eval)
    return data


# TextPrepare


REPLACE_BY_SPACE_RE = re.compile("[/(){}\[\]\|@,;]")
BAD_SYMBOLS_RE = re.compile("[^0-9a-z #+_]")
STOPWORDS = set(stopwords.words("english"))


def text_prepare(text):
    """
    text: a string

    return: modified initial string
    """
    text = text.lower()  # lowercase text
    text = re.sub(
        REPLACE_BY_SPACE_RE, " ", text
    )  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = re.sub(
        BAD_SYMBOLS_RE, "", text
    )  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = " ".join(
        [word for word in text.split() if not word in STOPWORDS]
    )  # delete stopwords from text
    return text


def test_text_prepare():
    """
    preprocess some test text
    """
    examples = [
        "SQL Server - any equivalent of Excel's CHOOSE function?",
        "How to free c++ memory vector<int> * arr?",
    ]
    answers = [
        "sql server equivalent excels choose function",
        "free c++ memory vectorint arr",
    ]
    for ex, ans in zip(examples, answers):
        if text_prepare(ex) != ans:
            return f"Wrong answer for the case: {ex}"
    return "Basic tests are passed."


# print(test_text_prepare())


def get_prepared_questions():
    """
    get the preprocessed questions
    """
    prepared_questions = []
    with open("data/text_prepare_tests.tsv", encoding="utf-8") as file:
        for line in file:
            line = text_prepare(line.strip())
            prepared_questions.append(line)
        text_prepare_results = "\n".join(prepared_questions)
    return text_prepare_results


# WordsTagsCount
def words_tags_count(x_train, y_train):
    """
    x_train: the training data
    y_train: the training labels

    return the count of words and tags
    """

    # Dictionary of all tags from train corpus with their counts.
    tags_counts = {}
    # Dictionary of all words from train corpus with their counts.
    words_counts = {}

    for sentence in x_train:
        for word in sentence.split():
            if word in words_counts:
                words_counts[word] += 1
            else:
                words_counts[word] = 1

    for tags in y_train:
        for tag in tags:
            if tag in tags_counts:
                tags_counts[tag] += 1
            else:
                tags_counts[tag] = 1
    return tags_counts, words_counts


# print(tags_counts)
# print(words_counts)

# print(sorted(words_counts, key=words_counts.get, reverse=True)[:3])


def get_most_common_tags_or_words(x_train, y_train, get_tags):
    """
    x_train: the training data
    y_train: the training labels
    get_tags: whether to return the tags_counts

    return the most common tags or commo words
    """
    tags_counts, words_counts = words_tags_count(x_train, y_train)
    if get_tags:
        most_common_tags = sorted(
            tags_counts.items(), key=lambda x: x[1], reverse=True
        )[:3]
        return most_common_tags
    most_common_words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[
        :3
    ]
    return most_common_words
