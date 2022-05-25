"""This module uses the model to predict the tags"""
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


def init_multilabel_classifier(y_train, y_val, tags_counts):
    """
    y_train: training labels
    y_val: actual tags
    tags_count: different tags

    return a MulitLabelBinarizer and the fitted training and actual tags
    """
    mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
    y_train = mlb.fit_transform(y_train)
    y_val = mlb.fit_transform(y_val)
    return mlb, y_train, y_val


def multilabel_bag_of_words(x_train_mybag, y_train, x_val_mybag):
    """
    x_train_mybag: the data to train on
    y_train: training labels
    x_val_mybag: bag of data to predict for

    return the predicted labels and scores
    """
    classifier_mybag = train_classifier(x_train_mybag, y_train)
    y_val_predicted_labels_mybag = classifier_mybag.predict(x_val_mybag)
    y_val_predicted_scores_mybag = classifier_mybag.decision_function(x_val_mybag)
    return y_val_predicted_labels_mybag, y_val_predicted_scores_mybag


def multilabel_tfidf(x_train_tfidf, y_train, x_val_tfidf):
    """
    x_train_tfidf: the data to train on
    y_train: training labels
    x_val_tfidf: bag of data to predict for

    return the predicted labels and scores for tfidf
    """
    classifier_tfidf = train_classifier(x_train_tfidf, y_train)
    y_val_predicted_labels_tfidf = classifier_tfidf.predict(x_val_tfidf)
    y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(x_val_tfidf)
    return y_val_predicted_labels_tfidf, y_val_predicted_scores_tfidf


def train_classifier(x_train, y_train, penalty="l1", cln=1):
    """
    x_train, y_train — training data

    return: trained classifier
    """

    # Create and fit LogisticRegression wraped into OneVsRestClassifier.

    clf = LogisticRegression(penalty=penalty, C=cln, dual=False, solver="liblinear")
    clf = OneVsRestClassifier(clf)
    clf.fit(x_train, y_train)

    return clf


def test_classifier(y_val_predicted_labels_tfidf, y_val, x_val, mlb):
    """
    y_val_predicted_labels_tfidf: predicted labels for tfidf
    y_val: actual labels
    x_val: data to predict
    mlb: the MultiLabelBinarizer

    return nothing, just print the predicted labels
    """
    y_val_pred_inversed = mlb.inverse_transform(y_val_predicted_labels_tfidf)
    y_val_inversed = mlb.inverse_transform(y_val)
    for i in range(3):
        y_val_inversed_joined = ",".join(y_val_inversed[i])
        y_val_pred_inversed_joined = ",".join(y_val_pred_inversed[i])
        print(
            f"Title:{x_val[i]}\nTrue labels:{y_val_inversed_joined}\n"
            f"Predicted labels:{y_val_pred_inversed_joined}\n\n"
        )
