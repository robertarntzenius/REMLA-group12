# pylint: disable=W1514
"""This module evaluates the model"""
# fmt: off
import json

from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             roc_auc_score)

# fmt: on


def print_evaluation_scores(y_val, predicted):
    """
    y_val: the actual tag
    predicted: the predicted tag

    return nothing, just print the accuracy
    """
    accuracy = accuracy_score(y_val, predicted)
    f1score = f1_score(y_val, predicted, average="weighted")
    precision = average_precision_score(y_val, predicted, average="macro")
    print("Accuracy score: ", accuracy)
    print("F1 score: ", f1score)
    print("Average precision score: ", precision)
    with open("metrics.json", "w") as file:
        json.dump(
            {"accuracy": accuracy, "F1 score": f1score, "precision": precision}, file
        )


def print_evaluation_scores_bag_of_words(y_val, y_val_predicted_labels_mybag):
    """
    y_val: the actual tag
    y_val_predicted_labels_mybag: the predicted bag of tags

    return nothing, just call a print method
    """
    print("Bag-of-words")
    print_evaluation_scores(y_val, y_val_predicted_labels_mybag)


def print_evaluation_scores_tfidf(y_val, y_val_predicted_labels_tfidf):
    """
    y_val: the actual tag
    y_val_predicted_labels_tfidf: the predicted tags of tfidf

    return nothing, just call a print method
    """
    print("Tfidf")
    print_evaluation_scores(y_val, y_val_predicted_labels_tfidf)


def print_roc_auc_score_bag_of_words(y_val, y_val_predicted_scores_mybag):
    """
    y_val: the actual tag
    y_val_predicted_scores_mybag: the predicted bag of scores

    return nothing, just print something
    """
    print(
        "Roc-auc: ",
        roc_auc_score(y_val, y_val_predicted_scores_mybag, multi_class="ovo"),
    )


def print_roc_auc_score_tfidf(y_val, y_val_predicted_scores_tfidf):
    """
    y_val: the actual tag
    y_val_predicted_scores_tfidf: the predicted scores of tfidf

    return nothing, just print something
    """
    print(
        "Roc-auc: ",
        roc_auc_score(y_val, y_val_predicted_scores_tfidf, multi_class="ovo"),
    )


# coefficients = [0.1, 1, 10, 100]
# penalties = ['l1', 'l2']

# for coefficient in coefficients:
#     for penalty in penalties:
#         classifier_tfidf = train_classifier(
#         X_train_tfidf, y_train, penalty=penalty, C=coefficient)
#         y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
#         y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)
#         print("Coefficient: {}, Penalty: {}".format(coefficient, penalty))
#         print_evaluation_scores(y_val, y_val_predicted_labels_tfidf)


# classifier_tfidf = train_classifier(X_train_tfidf, y_train, penalty='l2', C=10)
# y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
# y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)
#
#
# test_predictions = classifier_tfidf.predict(X_test_tfidf)######### YOUR CODE HERE #############
# test_pred_inversed = mlb.inverse_transform(test_predictions)
#
# test_predictions_for_submission = '\n'.join('%i\t%s' % (i, ','.join(row))
# for i, row in enumerate(test_pred_inversed))
