# pylint: disable=E0012,W1514,R5503,R5504,E0401,W0612,R0914,R0801
"""This module evaluates the robustness of the model"""
# fmt: off
import json
from ast import literal_eval

import nlpaug.augmenter.char as nac
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score

import multilabel
import preprocessing
import transform_text_to_vector as transform

# fmt: on


def main():
    """
    Train the model and test for robustness
    """
    train = pd.read_csv("data/train.tsv", sep="\t")
    train["tags"] = train["tags"].apply(literal_eval)
    validation = pd.read_csv("data/validation.tsv", sep="\t")
    validation["tags"] = validation["tags"].apply(literal_eval)
    test = pd.read_csv("data/test.tsv", sep="\t")

    x_train, y_train = train["title"].values, train["tags"].values
    x_val, y_val = validation["title"].values, validation["tags"].values
    x_test = test["title"].values

    tags_counts, words_counts = preprocessing.words_tags_count(x_train, y_train)

    aug = nac.KeyboardAug(aug_word_max=1)
    x_val_mod = []
    for title in x_val:
        title_mod = aug.augment(title)
        x_val_mod.append(title_mod)

    (
        x_train_mybag,
        x_val_mybag,
        x_test_mybag,
    ) = transform.bag_of_words(x_train, x_val_mod, x_test, words_counts)
    (
        x_train_tfidf,
        x_val_tfidf,
        x_test_tfidf,
        tfidf_vocab,
        tfidf_reversed_vocab,
    ) = transform.initialize_tfidf_vectorizer(x_train, x_val_mod, x_test)

    mlb, y_train, y_val = multilabel.init_multilabel_classifier(
        y_train, y_val, tags_counts
    )

    (
        y_val_predicted_labels_mybag,
        y_val_predicted_scores_mybag,
    ) = multilabel.multilabel_bag_of_words(x_train_mybag, y_train, x_val_mybag)
    (
        y_val_predicted_labels_tfidf,
        y_val_predicted_scores_tfidf,
    ) = multilabel.multilabel_tfidf(x_train_tfidf, y_train, x_val_tfidf)

    print_robustness_evaluation_scores_bag_of_words(y_val, y_val_predicted_labels_mybag)
    print_robustness_evaluation_scores_tfidf(y_val, y_val_predicted_labels_tfidf)


def print_robustness_evaluation_scores(y_val, predicted, is_bag_of_words):
    """
    y_val: the actual tag
    predicted: the predicted tag
    is_bag_of_words: whether it will print the scores for the bag of words

    return nothing, just print the accuracy
    """
    accuracy = accuracy_score(y_val, predicted)
    f1score = f1_score(y_val, predicted, average="weighted")
    precision = average_precision_score(y_val, predicted, average="macro")
    print("Accuracy score: ", accuracy)
    print("F1 score: ", f1score)
    print("Average precision score: ", precision)
    if is_bag_of_words:
        with open("reports/bag-of-words-robustness-metrics.json", "w") as file:
            json.dump(
                {"accuracy": accuracy, "F1 score": f1score, "precision": precision},
                file,
            )
    else:
        with open("reports/tfidf-robustness-metrics.json", "w") as file:
            json.dump(
                {"accuracy": accuracy, "F1 score": f1score, "precision": precision},
                file,
            )


def print_robustness_evaluation_scores_bag_of_words(
    y_val, y_val_predicted_labels_mybag
):
    """
    y_val: the actual tag
    y_val_predicted_labels_mybag: the predicted bag of tags

    return nothing, just call a print method
    """
    print("Bag-of-words")
    print_robustness_evaluation_scores(y_val, y_val_predicted_labels_mybag, True)


def print_robustness_evaluation_scores_tfidf(y_val, y_val_predicted_labels_tfidf):
    """
    y_val: the actual tag
    y_val_predicted_labels_tfidf: the predicted tags of tfidf

    return nothing, just call a print method
    """
    print("Tfidf")
    print_robustness_evaluation_scores(y_val, y_val_predicted_labels_tfidf, False)


if __name__ == "__main__":
    main()
