from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier


def init_multilabel_classifier(y_train, y_val, tags_counts):
    mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
    y_train = mlb.fit_transform(y_train)
    y_val = mlb.fit_transform(y_val)
    return mlb, y_train, y_val


def multilabel_bag_of_words(X_train_mybag, y_train, X_val_mybag):
    classifier_mybag = train_classifier(X_train_mybag, y_train)
    y_val_predicted_labels_mybag = classifier_mybag.predict(X_val_mybag)
    y_val_predicted_scores_mybag = classifier_mybag.decision_function(X_val_mybag)
    return y_val_predicted_labels_mybag, y_val_predicted_scores_mybag


def multilabel_tfidf(X_train_tfidf, y_train, X_val_tfidf):
    classifier_tfidf = train_classifier(X_train_tfidf, y_train)
    y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
    y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)
    return y_val_predicted_labels_tfidf, y_val_predicted_scores_tfidf


def train_classifier(X_train, y_train, penalty='l1', C=1):
    """
      X_train, y_train — training data

      return: trained classifier
    """

    # Create and fit LogisticRegression wraped into OneVsRestClassifier.

    clf = LogisticRegression(penalty=penalty, C=C, dual=False, solver='liblinear')
    clf = OneVsRestClassifier(clf)
    clf.fit(X_train, y_train)

    return clf


def test_classifier(y_val_predicted_labels_tfidf, y_val, X_val, mlb):
    y_val_pred_inversed = mlb.inverse_transform(y_val_predicted_labels_tfidf)
    y_val_inversed = mlb.inverse_transform(y_val)
    for i in range(3):
        print('Title:\t{}\nTrue labels:\t{}\nPredicted labels:\t{}\n\n'.format(
            X_val[i],
            ','.join(y_val_inversed[i]),
            ','.join(y_val_pred_inversed[i])
        ))


