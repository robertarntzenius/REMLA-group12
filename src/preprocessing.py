import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from ast import literal_eval
import pandas as pd
import numpy as np
import re

def init_preprocessing():

    train = read_data('data/train.tsv')
    validation = read_data('data/validation.tsv')
    test = pd.read_csv('data/test.tsv', sep='\t')

    X_train, y_train = train['title'].values, train['tags'].values
    X_val, y_val = validation['title'].values, validation['tags'].values
    X_test = test['title'].values

    # Text prepare
    X_train = [text_prepare(x) for x in X_train]
    X_val = [text_prepare(x) for x in X_val]
    X_test = [text_prepare(x) for x in X_test]
    return X_train, y_train, X_val, y_val, X_test

def read_data(filename):
    data = pd.read_csv(filename, sep='\t')
    data['tags'] = data['tags'].apply(literal_eval)
    return data


# TextPrepare


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))


def text_prepare(text):
    """
        text: a string

        return: modified initial string
    """
    text = text.lower() # lowercase text
    text = re.sub(REPLACE_BY_SPACE_RE, " ", text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = re.sub(BAD_SYMBOLS_RE, "", text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = " ".join([word for word in text.split() if not word in STOPWORDS]) # delete stopwords from text
    return text


def test_text_prepare():
    examples = ["SQL Server - any equivalent of Excel's CHOOSE function?",
                "How to free c++ memory vector<int> * arr?"]
    answers = ["sql server equivalent excels choose function",
               "free c++ memory vectorint arr"]
    for ex, ans in zip(examples, answers):
        if text_prepare(ex) != ans:
            return "Wrong answer for the case: '%s'" % ex
    return 'Basic tests are passed.'


# print(test_text_prepare())

def get_prepared_questions():
    prepared_questions = []
    for line in open('data/text_prepare_tests.tsv', encoding='utf-8'):
        line = text_prepare(line.strip())
        prepared_questions.append(line)
    text_prepare_results = '\n'.join(prepared_questions)
    return text_prepare_results


# WordsTagsCount
def words_tags_count(X_train, y_train):

    # Dictionary of all tags from train corpus with their counts.
    tags_counts = {}
    # Dictionary of all words from train corpus with their counts.
    words_counts = {}

    for sentence in X_train:
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


def get_most_common_tags(X_train, y_train):
    tags_counts, words_counts = words_tags_count(X_train, y_train)
    most_common_tags = sorted(tags_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    return most_common_tags


def get_most_common_words(X_train, y_train):
    tags_counts, words_counts = words_tags_count(X_train, y_train)
    most_common_words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    return most_common_words
